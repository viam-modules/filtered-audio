from typing import (
    ClassVar,
    Mapping,
    Sequence,
    Tuple,
    cast,
    List,
    AsyncGenerator,
    Any,
    Optional,
)
import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
import webrtcvad
from typing_extensions import Self
import numpy as np

from viam.components.audio_in import AudioIn, AudioResponse as AudioChunk
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict
from viam.streams import StreamWithIterator

from .fuzzy_matcher import FuzzyWakeWordMatcher
from .oww import setup_oww
from .vosk import (
    setup_vosk,
    AUDIO_SAMPLE_RATE_HZ,
    DEFAULT_VOSK_MODEL,
    DEFAULT_GRAMMAR_CONFIDENCE,
)
from vosk import Model as VoskModel, KaldiRecognizer

# Default configuration values
DEFAULT_VAD_AGGRESSIVENESS = 3  # 0-3, higher = less sensitive
MAX_BUFFER_SIZE_BYTES = 480000  # ~15 seconds at 16kHz
DEFAULT_SILENCE_DURATION_MS = (
    900  # milliseconds of silence before ending a speech segment
)
DEFAULT_MIN_SPEECH_DURATION_MS = 300  # min length of speech to process


class WakeWordFilter(AudioIn, EasyResource):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam", "filtered-audio"), "wake-word-filter"
    )

    # Instance variables
    logger: logging.Logger
    wake_words: List[str]
    vad: webrtcvad.Vad
    vosk_model: VoskModel
    recognizer: KaldiRecognizer
    executor: ThreadPoolExecutor
    is_shutting_down: bool
    microphone_client: AudioIn
    fuzzy_matcher: Optional[FuzzyWakeWordMatcher]
    silence_duration_ms: int
    min_speech_duration_ms: int
    detection_running: bool
    grammar_confidence: float
    use_grammar: bool
    detection_engine: str
    oww_model: Optional[Any]
    oww_model_name: Optional[str]
    oww_threshold: float

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        instance = super().new(config, dependencies)

        attrs = struct_to_dict(config.attributes)
        microphone = str(attrs.get("source_microphone", ""))
        # Get source microphone
        if microphone:
            mic = dependencies[AudioIn.get_resource_name(microphone)]
            instance.microphone_client = cast(AudioIn, mic)
        else:
            instance.logger.error(
                "wake-word-filter must have a source microphone, no microphone found"
            )
            raise RuntimeError(
                "wake-word-filter must have a source microphone, no microphone found"
            )

        wake_words = attrs.get("wake_words", [])

        # Handle both single string and list
        if isinstance(wake_words, str):
            instance.wake_words = [wake_words.lower()] if wake_words else []
        elif isinstance(wake_words, list):
            instance.wake_words = [str(w).lower() for w in wake_words]
        else:
            instance.wake_words = []

        vad_aggressiveness = int(
            attrs.get("vad_aggressiveness", DEFAULT_VAD_AGGRESSIVENESS)
        )

        instance.silence_duration_ms = int(
            attrs.get("silence_duration_ms", DEFAULT_SILENCE_DURATION_MS)
        )
        instance.logger.info(f"VAD Silence duration: {instance.silence_duration_ms}ms")

        instance.min_speech_duration_ms = int(
            attrs.get("min_speech_ms", DEFAULT_MIN_SPEECH_DURATION_MS)
        )
        instance.logger.info(
            f"min speech segment duration: {instance.min_speech_duration_ms}ms"
        )

        # Initialize WebRTC VAD (used by both engines)
        instance.vad = webrtcvad.Vad(vad_aggressiveness)
        instance.logger.info(
            f"WebRTC VAD initialized with aggressiveness: {vad_aggressiveness}"
        )

        # Detection engine selection
        instance.detection_engine = str(attrs.get("detection_engine", "vosk"))
        instance.logger.info(f"Detection engine: {instance.detection_engine}")

        # Initialize OWW defaults
        instance.oww_model = None
        instance.oww_model_name = None
        instance.oww_threshold = 0.5

        if instance.detection_engine == "openwakeword":
            setup_oww(
                instance,
                oww_model_path=str(attrs.get("oww_model_path", "")),
                oww_threshold=float(attrs.get("oww_threshold", 0.5)),
            )
        else:
            setup_vosk(
                instance,
                vosk_model=str(attrs.get("vosk_model", DEFAULT_VOSK_MODEL)),
                use_grammar=bool(attrs.get("use_grammar", True)),
                grammar_confidence=float(
                    attrs.get("vosk_grammar_confidence", DEFAULT_GRAMMAR_CONFIDENCE)
                ),
                fuzzy_threshold=int(attrs["fuzzy_threshold"])
                if attrs.get("fuzzy_threshold") is not None
                else None,
            )

        # Create thread pool for processing (non-blocking)
        instance.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="wakeword"
        )
        instance.is_shutting_down = False
        instance.logger.debug("Thread pool executor created")

        # Detection pause state (for muting during TTS playback)
        instance.detection_running = True

        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        deps: List[str] = []
        attrs = struct_to_dict(config.attributes)
        mic: Any = attrs.get("source_microphone", "")

        if mic == "":
            raise ValueError("source_microphone attribute is required")
        if not isinstance(mic, str):
            raise ValueError("source_microphone attribute must be a string")
        deps.append(mic)

        wake_words: Any = attrs.get("wake_words", [])
        _engine = attrs.get("detection_engine", "vosk")
        if _engine != "openwakeword" and not wake_words:
            raise ValueError(
                "wake_words is required when using the vosk detection engine"
            )

        # Validate wake_words type if provided
        if wake_words:
            if isinstance(wake_words, str):
                # Single string is valid
                pass
            elif isinstance(wake_words, list):
                # List must contain only strings
                for word in wake_words:
                    if not isinstance(word, str):
                        raise ValueError(
                            f"All wake_words must be strings, got {type(word).__name__}"
                        )
            else:
                raise ValueError(
                    f"wake_words must be a string or list of strings, got {type(wake_words).__name__}"
                )

        # Validate VAD aggressiveness
        vad_aggressiveness: Any = attrs.get("vad_aggressiveness", None)
        if vad_aggressiveness is not None:
            if (
                not isinstance(vad_aggressiveness, (int, float))
                or vad_aggressiveness % 1 != 0
            ):
                raise ValueError("vad_aggressiveness must be a whole number")
        if vad_aggressiveness is not None and not 0 <= vad_aggressiveness <= 3:
            raise ValueError(
                f"vad_aggressiveness must be 0-3, got {vad_aggressiveness}"
            )

        # Validate fuzzy threshold
        fuzzy_threshold: Any = attrs.get("fuzzy_threshold", None)
        if fuzzy_threshold is not None:
            if (
                not isinstance(fuzzy_threshold, (int, float))
                or fuzzy_threshold % 1 != 0
            ):
                raise ValueError("fuzzy_threshold must be a whole number")
        if fuzzy_threshold is not None and not 0 <= fuzzy_threshold <= 5:
            raise ValueError(f"fuzzy_threshold must be 0-5, got {fuzzy_threshold}")

        # Validate silence_duration_ms
        silence_duration_ms: Any = attrs.get("silence_duration_ms", None)
        if silence_duration_ms is not None:
            if (
                not isinstance(silence_duration_ms, (int, float))
                or silence_duration_ms % 1 != 0
            ):
                raise ValueError("silence_duration_ms must be a whole number")
            if silence_duration_ms <= 0:
                raise ValueError("silence_duration_ms must be positive")

        # Validate min_speech_ms
        min_speech_ms: Any = attrs.get("min_speech_ms", None)
        if min_speech_ms is not None:
            if not isinstance(min_speech_ms, (int, float)) or min_speech_ms % 1 != 0:
                raise ValueError("min_speech_ms must be a whole number")
            if min_speech_ms <= 0:
                raise ValueError("min_speech_ms must be positive")

        # Validate use_grammar
        use_grammar: Any = attrs.get("use_grammar", None)
        if use_grammar is not None:
            if not isinstance(use_grammar, bool):
                raise ValueError("use_grammar must be a boolean")

        # Validate grammar_confidence
        grammar_confidence: Any = attrs.get("vosk_grammar_confidence", None)
        if grammar_confidence is not None:
            if not isinstance(grammar_confidence, (int, float)):
                raise ValueError("vosk_grammar_confidence must be a number")
            if grammar_confidence < 0.0 or grammar_confidence > 1.0:
                raise ValueError(
                    f"vosk_grammar_confidence must be 0.0-1.0, got {grammar_confidence}"
                )

        # Validate detection_engine
        detection_engine: Any = attrs.get("detection_engine", "vosk")
        if detection_engine not in ("vosk", "openwakeword"):
            raise ValueError(
                f"detection_engine must be 'vosk' or 'openwakeword', got '{detection_engine}'"
            )

        # Validate openWakeWord-specific config
        if detection_engine == "openwakeword":
            oww_model_path: Any = attrs.get("oww_model_path", "")
            if not isinstance(oww_model_path, str) or not oww_model_path:
                raise ValueError(
                    "oww_model_path must be a non-empty string"
                    if not isinstance(oww_model_path, str)
                    else "oww_model_path is required when detection_engine is 'openwakeword'"
                )

            oww_threshold: Any = attrs.get("oww_threshold", None)
            if oww_threshold is not None:
                if not isinstance(oww_threshold, (int, float)):
                    raise ValueError("oww_threshold must be a number")
                if oww_threshold < 0.0 or oww_threshold > 1.0:
                    raise ValueError(
                        f"oww_threshold must be 0.0-1.0, got {oww_threshold}"
                    )

        return deps, []

    async def _vosk_process_segment(
        self, speech_chunk_buffer: List[AudioChunk], speech_buffer: bytearray
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Vosk: run wake word inference on the full buffered speech segment.

        Unlike OWW (which processes each frame in real-time), Vosk runs once
        on the complete segment after VAD detects end-of-speech. A future
        optimization could feed frames incrementally via AcceptWaveform() for
        lower latency.

        Args:
            speech_chunk_buffer: chunks to yield if wake word detected
            speech_buffer: accumulated raw PCM for the full speech segment
        """
        if not speech_chunk_buffer:
            return

        # Don't process if we're shutting down
        if self.is_shutting_down:
            self.logger.debug("Skipping speech processing due to shutdown")
            return

        try:
            wake_word_detected = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                self._vosk_check_for_wake_word,
                bytes(speech_buffer),
            )

            if wake_word_detected:
                self.logger.info(
                    f"Wake word detected! Yielding {len(speech_chunk_buffer)} chunks ({len(speech_buffer)} bytes)"
                )
                for chunk in speech_chunk_buffer:
                    yield chunk
                # Yield empty chunk to signal segment end
                empty_response = AudioChunk()
                empty_response.audio.audio_data = b""
                yield empty_response
                self.logger.debug("Sent empty chunk to signal segment end")
        except RuntimeError as e:
            if "shutdown" in str(e).lower():
                self.logger.debug("Executor shutdown during processing, ignoring")
                return
            raise

    def _validate_mic_properties(self, mic_props: Any) -> None:
        """Raise ValueError if mic sample rate or channel count is incompatible."""
        if mic_props.sample_rate_hz != AUDIO_SAMPLE_RATE_HZ:
            raise ValueError(
                f"Wake word filter requires 16000 Hz audio, "
                f"but source microphone provides {mic_props.sample_rate_hz} Hz. "
                f"Please configure source microphone to output 16000 Hz PCM16 audio."
            )
        if mic_props.num_channels != 1:
            raise ValueError(
                f"Wake word filter requires mono (1 channel) audio, "
                f"but source microphone provides {mic_props.num_channels} channels. "
                f"Please configure source microphone for mono audio."
            )

    async def _finalize_segment(
        self,
        speech_chunk_buffer: List[AudioChunk],
        speech_buffer: bytearray,
        oww_detected: bool,
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Finalize a completed speech segment for the active detection engine.

        OWW: detection already happened per-frame, so just check the result
        and yield buffered chunks if the wake word was detected, then reset
        the OWW model state for the next segment.

        Vosk: detection hasn't run yet — run inference on the full buffered
        segment now, then yield chunks if the wake word was found.
        """
        if self.detection_engine == "openwakeword":
            if oww_detected:
                self.logger.info(
                    "OWW: Yielding %d chunks (%d bytes)",
                    len(speech_chunk_buffer),
                    len(speech_buffer),
                )
                for chunk in speech_chunk_buffer:
                    yield chunk
                empty_response = AudioChunk()
                empty_response.audio.audio_data = b""
                yield empty_response
            else:
                buf = self.oww_model.prediction_buffer.get(self.oww_model_name)
                max_score = max(buf) if buf else 0.0
                self.logger.debug(
                    "OWW: No detection (max_score=%.3f, threshold=%.2f)",
                    max_score,
                    self.oww_threshold,
                )
            self.oww_model.reset()
        else:
            # Vosk: run inference on the complete buffered segment
            async for chunk in self._vosk_process_segment(
                speech_chunk_buffer, speech_buffer
            ):
                yield chunk

    async def get_audio(
        self, codec: str, duration_seconds: float, previous_timestamp_ns: int, **kwargs
    ) -> StreamWithIterator:
        """
        Stream audio, yielding buffered audio chunks when a wake word detected.

        Uses WebRTC VAD to detect speech, then Vosk or OWW to detect wake words

        Args:
            codec: Audio codec (should be "pcm16")
            duration_seconds: Duration (use 0 for continuous)
            previous_timestamp_ns: Previous timestamp (use 0 to start from now)

        Yields:
            AudioResponse chunks
        """
        if not self.microphone_client:
            raise ValueError("no source microphone found")

        if codec.lower() != "pcm16":
            self.logger.error(f"Unsupported codec: {codec}. Only PCM16 is supported.")
            raise ValueError(
                f"Wake word filter only supports PCM16 codec, got: {codec}"
            )

        async def audio_generator() -> AsyncGenerator[AudioChunk, None]:
            self.logger.info(
                f"Starting speech detection with VAD... (duration_seconds={duration_seconds})"
            )

            # Check mic properties
            mic_props = await self.microphone_client.get_properties()
            self.logger.debug(
                f"Microphone properties - Sample rate: {mic_props.sample_rate_hz} Hz, Channels: {mic_props.num_channels}"
            )

            self._validate_mic_properties(mic_props)

            mic_stream = await self.microphone_client.get_audio(
                codec, duration_seconds, previous_timestamp_ns
            )
            self.logger.info(
                f"Microphone stream started (requested duration: {duration_seconds}s)"
            )

            # --- Shared buffers (used by both engines) ---
            speech_chunk_buffer: list[AudioChunk] = []  # chunks to yield on detection
            speech_buffer = bytearray()  # raw PCM accumulated during speech segment
            audio_buffer = bytearray()  # working buffer for slicing into VAD frames
            is_speech_active = False
            silence_frames = 0
            speech_frames = 0
            frame_duration_ms = 30
            max_silence_frames = self.silence_duration_ms // frame_duration_ms
            min_speech_frames = self.min_speech_duration_ms // frame_duration_ms

            # --- OWW-only state ---
            # Frames should be multiples of 80ms, higher frame size
            # = better efficiency but longer latency
            # 16kHz * 0.080s = 1280 samples * 2 bytes = 2560 bytes
            oww_detected = False
            oww_audio_buffer = bytearray()
            OWW_CHUNK_SIZE = 2560

            def reset_buffers():
                """Clear all buffers and reset speech state."""
                nonlocal is_speech_active, silence_frames, speech_frames, oww_detected
                speech_chunk_buffer.clear()
                speech_buffer.clear()
                audio_buffer.clear()
                oww_audio_buffer.clear()
                is_speech_active = False
                silence_frames = 0
                speech_frames = 0
                oww_detected = False

            def oww_run_inference() -> bool:
                """Drain oww_audio_buffer in chunks and run OWW inference on each."""
                while len(oww_audio_buffer) >= OWW_CHUNK_SIZE:
                    oww_chunk = bytes(oww_audio_buffer[:OWW_CHUNK_SIZE])
                    del oww_audio_buffer[:OWW_CHUNK_SIZE]
                    audio_int16 = np.frombuffer(oww_chunk, dtype=np.int16)
                    prediction = self.oww_model.predict(audio_int16)
                    score = prediction.get(self.oww_model_name, 0.0)
                    if score >= self.oww_threshold:
                        self.logger.info(
                            "Wake word detected (score=%.3f >= %.3f)",
                            score,
                            self.oww_threshold,
                        )
                        return True
                return False

            def oww_process_frame(frame):
                """Feed audio frame into OWW buffer and run inference."""
                nonlocal oww_detected
                # wake word already detected earlier in this segment
                if oww_detected:
                    return
                oww_audio_buffer.extend(frame)
                if oww_run_inference():
                    oww_detected = True

            async for audio_chunk in mic_stream:
                # Exit stream if shutting down
                if self.is_shutting_down:
                    self.logger.info("Stream ending due to shutdown")
                    break

                # Skip processing when detection is paused (e.g., during TTS)
                if not self.detection_running:
                    # Clear any buffered speech to avoid stale data
                    if speech_chunk_buffer or speech_buffer:
                        reset_buffers()
                    continue

                audio_data = audio_chunk.audio.audio_data

                if not audio_data:
                    continue

                # WebRTC VAD requires specific frame sizes (10, 20, or 30ms)
                # At 16kHz: 30ms = 480 samples = 960 bytes
                frame_size = 960
                should_process = False

                # Track if we've added this chunk to the buffer yet
                chunk_added = False

                # Add new audio data to working buffer
                audio_buffer.extend(audio_data)

                # Process complete 30ms frames
                while len(audio_buffer) >= frame_size:
                    frame = audio_buffer[:frame_size]

                    # Check if frame contains speech
                    try:
                        # Note: WebRTC VAD only accepts 16-bit mono PCM audio,
                        # sampled at 8000, 16000, or 32000 Hz.
                        # A frame must be either 10, 20, or 30 ms in duration
                        is_speech = self.vad.is_speech(frame, mic_props.sample_rate_hz)
                    except Exception as e:
                        self.logger.error(f"VAD error: {e}")
                        is_speech = False

                    if is_speech:
                        if not is_speech_active:
                            self.logger.debug("Speech segment started")
                            is_speech_active = True
                        silence_frames = 0
                        speech_frames += 1

                        # Check buffer limit before adding frame
                        if len(speech_buffer) >= MAX_BUFFER_SIZE_BYTES:
                            should_process = True
                            break

                        # Buffer frame (both engines accumulate speech here;
                        # Vosk runs inference on the full segment at end-of-speech)
                        if not chunk_added:
                            speech_chunk_buffer.append(audio_chunk)
                            chunk_added = True
                        speech_buffer.extend(frame)

                        # OWW: feed each frame for real-time per-frame detection
                        if self.detection_engine == "openwakeword":
                            oww_process_frame(frame)
                    else:
                        if is_speech_active:
                            # Check buffer limit before adding frame
                            if len(speech_buffer) >= MAX_BUFFER_SIZE_BYTES:
                                should_process = True
                                break

                            # Buffer silence frames during active speech segment
                            if not chunk_added:
                                speech_chunk_buffer.append(audio_chunk)
                                chunk_added = True
                            speech_buffer.extend(frame)

                            # OWW: feed silence frames too — needs continuous audio
                            if self.detection_engine == "openwakeword":
                                oww_process_frame(frame)

                            silence_frames += 1

                            if silence_frames >= max_silence_frames:
                                should_process = True
                                audio_buffer = audio_buffer[
                                    frame_size:
                                ]  # Remove processed frame
                                break

                    # Remove the frame we just processed
                    audio_buffer = audio_buffer[frame_size:]

                # If speech segment ended or buffer limit reached, check for wake word
                if should_process:
                    if speech_frames >= min_speech_frames:
                        self.logger.debug(
                            "Speech segment ended (%d frames, %d bytes)",
                            speech_frames,
                            len(speech_buffer),
                        )
                        async for chunk in self._finalize_segment(
                            speech_chunk_buffer, speech_buffer, oww_detected
                        ):
                            yield chunk
                    else:
                        self.logger.debug(
                            "Ignoring false positive: only %d frames", speech_frames
                        )
                        if self.detection_engine == "openwakeword":
                            self.oww_model.reset()

                    reset_buffers()

                # Secondary check (shouldn't be needed but safety)
                if len(speech_buffer) >= MAX_BUFFER_SIZE_BYTES:
                    self.logger.debug("Processing speech segment")
                    async for chunk in self._finalize_segment(
                        speech_chunk_buffer, speech_buffer, oww_detected
                    ):
                        yield chunk
                    reset_buffers()

            # Process any remaining buffered audio when stream ends
            if speech_chunk_buffer and speech_frames >= min_speech_frames:
                self.logger.debug(
                    "Stream ended with %d bytes buffered, processing",
                    len(speech_buffer),
                )
                async for chunk in self._finalize_segment(
                    speech_chunk_buffer, speech_buffer, oww_detected
                ):
                    yield chunk
            elif speech_chunk_buffer:
                self.logger.debug(
                    "Stream ended: ignoring buffered audio (only %d frames, likely false positive)",
                    speech_frames,
                )
                if self.detection_engine == "openwakeword":
                    self.oww_model.reset()

        return StreamWithIterator(audio_generator())

    def _vosk_check_for_wake_word(self, audio_bytes: bytes) -> bool:
        """
        Check if any wake word is in audio.

        Args:
            audio_bytes: Raw PCM16 audio data

        Returns:
            bool: True if any wake word detected
        """
        try:
            self.recognizer.AcceptWaveform(audio_bytes)
            result = json.loads(self.recognizer.FinalResult())
            text = result.get("text", "").lower()

            # Check confidence to reduce false positives from grammar forcing
            self.logger.debug("Vosk result: %s", result)
            if "result" in result and result["result"]:
                avg_conf = sum(w.get("conf", 1.0) for w in result["result"])
                avg_conf /= len(result["result"])
                self.logger.debug("Vosk confidence: %.2f", avg_conf)
                if avg_conf < self.grammar_confidence:
                    self.logger.debug(
                        "Rejecting low confidence: '%s' (conf=%.2f < %.2f)",
                        text,
                        avg_conf,
                        self.grammar_confidence,
                    )
                    return False

            if text:
                self.logger.debug(f"Recognized: '{text}'")
            else:
                self.logger.debug("Vosk returned empty text, no speech recognized")
                return False

            for wake_word in self.wake_words:
                if self.fuzzy_matcher and not self.use_grammar:
                    match_details = self.fuzzy_matcher.match(text, wake_word)
                    if match_details:
                        self.logger.info(
                            f"Wake word '{wake_word}' detected (fuzzy: "
                            f"'{match_details['matched_text']}', "
                            f"distance={match_details['distance']})"
                        )
                        return True
                else:
                    # checking whole return phrase for wake word
                    pattern = rf"\b{re.escape(wake_word)}\b"
                    match = re.search(pattern, text)
                    self.logger.debug(
                        "Checking wake_word='%s' pattern='%s' against text='%s' -> %s",
                        wake_word,
                        pattern,
                        text,
                        match,
                    )
                    if match:
                        self.logger.info("Wake word '%s' detected", wake_word)
                        return True

            self.logger.debug("No wake word match found")
            return False
        except Exception as e:
            self.logger.error(f"Vosk error: {e}", exc_info=True)
            return False

    async def close(self) -> None:
        # Signal shutdown to prevent new tasks
        self.is_shutting_down = True

        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
            self.logger.debug("Thread pool executor shut down")

        if self.oww_model is not None:
            self.oww_model.reset()
            self.oww_model = None

    async def do_command(
        self, command: Mapping[str, Any], **kwargs
    ) -> Mapping[str, Any]:
        """
        Handle commands for the wake word filter.

        Supported commands:
        - pause_detection: Pause wake word detection
        - resume_detection: Resume wake word detection

        Examples:
            {"pause_detection": None}  # pause detection
            {"resume_detection": None}    # resume detection
        """
        if "pause_detection" in command:
            self.detection_running = False
            self.logger.info("Detection paused")
            return {"status": "paused"}

        elif "resume_detection" in command:
            self.detection_running = True
            self.logger.info("Detection resumed")
            return {"status": "resumed"}

        else:
            raise ValueError(f"Unknown command keys: {list(command.keys())}")

    async def get_geometries(self, **kwargs) -> List[Any]:
        raise NotImplementedError()

    async def get_properties(self, **kwargs) -> AudioIn.Properties:
        # Return properties from underlying microphone
        props = await self.microphone_client.get_properties()
        # Ensure we only report PCM16 support
        return AudioIn.Properties(
            supported_codecs=["pcm16"],
            sample_rate_hz=props.sample_rate_hz,
            num_channels=props.num_channels,
        )
