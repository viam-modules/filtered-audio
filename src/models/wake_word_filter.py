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
from ._speech_segment import _SpeechState, _SpeechSegment, _SegmentThresholds
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
import webrtcvad
from typing_extensions import Self

from viam.components.audio_in import AudioIn, AudioResponse as AudioChunk
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict
from viam.streams import StreamWithIterator

from .fuzzy_matcher import FuzzyWakeWordMatcher
from .oww import setup_oww, oww_check_for_wake_word
from .vosk import (
    setup_vosk,
    vosk_process_segment,
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

# WebRTC VAD requires 30ms frames: 480 samples * 2 bytes = 960 bytes at 16kHz
FRAME_DURATION_MS = 30
FRAME_SIZE_BYTES = 960


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

        # Validate detection_engine
        detection_engine: Any = attrs.get("detection_engine", "vosk")
        if detection_engine not in ("vosk", "openwakeword"):
            raise ValueError(
                f"detection_engine must be 'vosk' or 'openwakeword', got '{detection_engine}'"
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

        # Validate Vosk-specific config
        if detection_engine == "vosk":
            wake_words: Any = attrs.get("wake_words", [])
            if not wake_words:
                raise ValueError(
                    "wake_words is required when using the vosk detection engine"
                )

            if wake_words:
                if isinstance(wake_words, str):
                    pass
                elif isinstance(wake_words, list):
                    for word in wake_words:
                        if not isinstance(word, str):
                            raise ValueError(
                                f"All wake_words must be strings, got {type(word).__name__}"
                            )
                else:
                    raise ValueError(
                        f"wake_words must be a string or list of strings, got {type(wake_words).__name__}"
                    )

            fuzzy_threshold: Any = attrs.get("fuzzy_threshold", None)
            if fuzzy_threshold is not None:
                if (
                    not isinstance(fuzzy_threshold, (int, float))
                    or fuzzy_threshold % 1 != 0
                ):
                    raise ValueError("fuzzy_threshold must be a whole number")
            if fuzzy_threshold is not None and not 0 <= fuzzy_threshold <= 5:
                raise ValueError(f"fuzzy_threshold must be 0-5, got {fuzzy_threshold}")

            use_grammar: Any = attrs.get("use_grammar", None)
            if use_grammar is not None:
                if not isinstance(use_grammar, bool):
                    raise ValueError("use_grammar must be a boolean")

            grammar_confidence: Any = attrs.get("vosk_grammar_confidence", None)
            if grammar_confidence is not None:
                if not isinstance(grammar_confidence, (int, float)):
                    raise ValueError("vosk_grammar_confidence must be a number")
                if grammar_confidence < 0.0 or grammar_confidence > 1.0:
                    raise ValueError(
                        f"vosk_grammar_confidence must be 0.0-1.0, got {grammar_confidence}"
                    )

        # Validate openWakeWord-specific config
        elif detection_engine == "openwakeword":
            oww_model_path: Any = attrs.get("oww_model_path", "")
            if not isinstance(oww_model_path, str) or not oww_model_path:
                raise ValueError(
                    "oww_model_path must be a non-empty string"
                    if not isinstance(oww_model_path, str)
                    else "oww_model_path is required when detection_engine is 'openwakeword'"
                )

            path_for_ext = oww_model_path.split("?")[0]  # strip query params if URL
            if path_for_ext.endswith(".tflite") and sys.platform != "linux":
                raise ValueError(
                    "tflite models are only supported on Linux. "
                    "Please use an .onnx model instead."
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

            await self._validate_mic_properties()

            mic_stream = await self.microphone_client.get_audio(
                codec, duration_seconds, previous_timestamp_ns
            )
            self.logger.info(
                f"Microphone stream started (requested duration: {duration_seconds}s)"
            )

            config = _SegmentThresholds(
                max_silence_frames=self.silence_duration_ms // FRAME_DURATION_MS,
                min_speech_frames=self.min_speech_duration_ms // FRAME_DURATION_MS,
            )

            speech_segment = _SpeechSegment()
            state = _SpeechState.IDLE
            vad_audio_buffer = bytearray()

            async for audio_chunk in mic_stream:
                if self.is_shutting_down:
                    self.logger.info("Stream ending due to shutdown")
                    break

                # Skip processing when detection is paused (e.g., during TTS)
                if not self.detection_running:
                    if (
                        speech_segment.speech_chunk_buffer
                        or speech_segment.speech_buffer
                    ):
                        speech_segment.reset()
                        state = _SpeechState.IDLE
                    vad_audio_buffer.clear()
                    continue

                audio_data = audio_chunk.audio.audio_data
                if not audio_data:
                    continue

                vad_audio_buffer.extend(audio_data)

                while len(vad_audio_buffer) >= FRAME_SIZE_BYTES:
                    frame = bytes(vad_audio_buffer[:FRAME_SIZE_BYTES])
                    del vad_audio_buffer[:FRAME_SIZE_BYTES]

                    segment_complete, state = self._process_vad_frame(
                        speech_segment, state, frame, audio_chunk, config
                    )
                    if segment_complete:
                        async for chunk in self._finalize_segment(
                            speech_segment, config
                        ):
                            yield chunk
                        state = _SpeechState.IDLE
                        break

            # Process any remaining buffered audio when stream ends
            if state in (_SpeechState.ACTIVE, _SpeechState.TRAILING):
                if speech_segment.speech_frames >= config.min_speech_frames:
                    self.logger.debug(
                        "Stream ended with %d bytes buffered, processing",
                        len(speech_segment.speech_buffer),
                    )
                    async for chunk in self._run_detection(
                        speech_segment.speech_chunk_buffer,
                        speech_segment.speech_buffer,
                        speech_segment.oww_detected,
                    ):
                        yield chunk
                else:
                    self.logger.debug(
                        "Stream ended: ignoring buffered audio (only %d frames, likely false positive)",
                        speech_segment.speech_frames,
                    )
                    if self.detection_engine == "openwakeword":
                        self.oww_model.reset()

        return StreamWithIterator(audio_generator())

    async def _validate_mic_properties(self) -> AudioIn.Properties:
        """Fetch mic properties, log them, and raise ValueError if incompatible."""
        mic_props = await self.microphone_client.get_properties()
        self.logger.debug(
            f"Microphone properties - Sample rate: {mic_props.sample_rate_hz} Hz, Channels: {mic_props.num_channels}"
        )
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
        return mic_props

    async def _run_detection(
        self,
        speech_chunk_buffer: List[AudioChunk],
        speech_buffer: bytearray,
        oww_detected: bool,
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Run wake word detection on a buffered speech segment and yield matching chunks.

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
            async for chunk in vosk_process_segment(
                self, speech_chunk_buffer, speech_buffer
            ):
                yield chunk

    async def _finalize_segment(
        self, speech_segment: _SpeechSegment, config: _SegmentThresholds
    ) -> AsyncGenerator[AudioChunk, None]:
        """Finalize the current segment and reset to IDLE."""
        if speech_segment.speech_frames >= config.min_speech_frames:
            self.logger.debug(
                "Speech segment ended (%d frames, %d bytes)",
                speech_segment.speech_frames,
                len(speech_segment.speech_buffer),
            )
            async for chunk in self._run_detection(
                speech_segment.speech_chunk_buffer,
                speech_segment.speech_buffer,
                speech_segment.oww_detected,
            ):
                yield chunk
        else:
            self.logger.debug(
                "Ignoring false positive: only %d frames", speech_segment.speech_frames
            )
            if self.detection_engine == "openwakeword":
                self.oww_model.reset()
        speech_segment.reset()

    def _process_vad_frame(
        self,
        speech_segment: _SpeechSegment,
        state: _SpeechState,
        frame: bytes,
        audio_chunk: AudioChunk,
        config: _SegmentThresholds,
    ) -> tuple[bool, _SpeechState]:
        """Classify one VAD frame, update state machine, and buffer audio.

        Returns:
            (segment_complete, new_state)
        """
        try:
            is_speech = self.vad.is_speech(frame, AUDIO_SAMPLE_RATE_HZ)
        except Exception as e:
            self.logger.error(f"VAD error: {e}")
            is_speech = False

        # OWW receives every frame (including silence) for continuous inference
        if self.detection_engine == "openwakeword":
            if not speech_segment.oww_detected:
                speech_segment.oww_audio_buffer.extend(frame)
                speech_segment.oww_detected = oww_check_for_wake_word(
                    self, speech_segment.oww_audio_buffer
                )

        if state == _SpeechState.IDLE:
            if is_speech:
                self.logger.debug("Speech segment started")
                state = _SpeechState.ACTIVE
                speech_segment.speech_frames = 1
                speech_segment.speech_chunk_buffer.append(audio_chunk)
                speech_segment.speech_buffer.extend(frame)
        else:
            # ACTIVE or TRAILING: buffer every frame, but only add the chunk once
            # (one audio_chunk may contain multiple VAD frames)
            if (
                not speech_segment.speech_chunk_buffer
                or speech_segment.speech_chunk_buffer[-1] is not audio_chunk
            ):
                speech_segment.speech_chunk_buffer.append(audio_chunk)
            speech_segment.speech_buffer.extend(frame)

            if state == _SpeechState.ACTIVE:
                if is_speech:
                    speech_segment.speech_frames += 1
                else:
                    # Speech stopped — start counting silence
                    state = _SpeechState.TRAILING
                    speech_segment.silence_frames = 1

            elif state == _SpeechState.TRAILING:
                if is_speech:
                    # Speech resumed — back to ACTIVE
                    state = _SpeechState.ACTIVE
                    speech_segment.speech_frames += 1
                    speech_segment.silence_frames = 0
                else:
                    speech_segment.silence_frames += 1

            if (
                speech_segment.silence_frames >= config.max_silence_frames
                or len(speech_segment.speech_buffer) >= MAX_BUFFER_SIZE_BYTES
            ):
                return True, state

        return False, state

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
