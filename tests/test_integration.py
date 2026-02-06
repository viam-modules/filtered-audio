"""
Integration tests for wake word filter.

These tests use real Vosk models and VAD to test the full pipeline.
"""

import struct
from typing import AsyncGenerator, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.wake_word_filter import (
    AUDIO_SAMPLE_RATE_HZ,
    MAX_BUFFER_SIZE_BYTES,
    WakeWordFilter,
)
from viam.components.audio_in import AudioResponse as AudioChunk


def generate_silence(
    duration_ms: int, sample_rate: int = AUDIO_SAMPLE_RATE_HZ
) -> bytes:
    """Generate silent audio (zeros) for the given duration."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


def create_audio_chunk(audio_data: bytes) -> AudioChunk:
    """Create an AudioChunk with the given audio data."""
    chunk = AudioChunk()
    chunk.audio.audio_data = audio_data
    return chunk


async def audio_stream_from_chunks(
    chunks: List[bytes],
) -> AsyncGenerator[AudioChunk, None]:
    """Create an async generator from a list of audio byte chunks."""
    for chunk_data in chunks:
        yield create_audio_chunk(chunk_data)


@pytest.fixture
def mock_microphone():
    """Create a mock microphone with configurable audio stream."""
    mic = AsyncMock()
    mic.get_properties.return_value = Mock(
        sample_rate_hz=AUDIO_SAMPLE_RATE_HZ, num_channels=1
    )
    return mic


@pytest.fixture
def integration_env():
    """
    Set up environment for integration tests with real Vosk model.

    This fixture patches only the minimum needed (microphone dependency lookup)
    while using real Vosk, VAD, and recognition components.
    """
    with (
        patch(
            "src.models.wake_word_filter.AudioIn.get_resource_name", return_value="mic1"
        ),
    ):
        yield


class TestVADIntegration:
    """Integration tests for VAD (Voice Activity Detection)."""

    def test_vad_detects_silence(self):
        """Test that VAD correctly identifies silence."""
        import webrtcvad

        vad = webrtcvad.Vad(3)  # Most aggressive
        silence = generate_silence(30)  # 30ms frame
        assert vad.is_speech(silence, AUDIO_SAMPLE_RATE_HZ) is False

    def test_vad_frame_size_requirements(self):
        """Test VAD works with valid frame sizes (10, 20, 30ms)."""
        import webrtcvad

        vad = webrtcvad.Vad(3)

        for frame_ms in [10, 20, 30]:
            silence = generate_silence(frame_ms)
            result = vad.is_speech(silence, AUDIO_SAMPLE_RATE_HZ)
            assert isinstance(result, bool)

    def test_vad_detects_gtts_speech(self):
        """Test VAD detects actual TTS-generated speech."""
        import webrtcvad
        from tests.fixtures.speech_audio import get_speech_audio

        vad = webrtcvad.Vad(2)  # Moderate aggressiveness
        speech = get_speech_audio("hello robot turn on the lights")

        # Count speech frames in 30ms chunks
        frame_bytes = int(AUDIO_SAMPLE_RATE_HZ * 0.030 * 2)  # 30ms, 2 bytes/sample
        speech_frame_count = 0
        total_frames = 0

        for i in range(0, len(speech) - frame_bytes, frame_bytes):
            frame = speech[i : i + frame_bytes]
            total_frames += 1
            if vad.is_speech(frame, AUDIO_SAMPLE_RATE_HZ):
                speech_frame_count += 1

        # TTS audio should have significant speech content
        speech_ratio = speech_frame_count / total_frames if total_frames > 0 else 0
        assert speech_ratio > 0.3, f"Expected >30% speech frames, got {speech_ratio:.1%}"

    def test_vad_distinguishes_speech_from_silence(self):
        """Test VAD can distinguish between speech and silence."""
        import webrtcvad
        from tests.fixtures.speech_audio import get_speech_audio

        vad = webrtcvad.Vad(2)
        frame_bytes = int(AUDIO_SAMPLE_RATE_HZ * 0.030 * 2)

        # Check silence
        silence = generate_silence(300)  # 300ms
        silence_speech_count = sum(
            1
            for i in range(0, len(silence) - frame_bytes, frame_bytes)
            if vad.is_speech(silence[i : i + frame_bytes], AUDIO_SAMPLE_RATE_HZ)
        )

        # Check speech
        speech = get_speech_audio("robot")
        speech_speech_count = sum(
            1
            for i in range(0, len(speech) - frame_bytes, frame_bytes)
            if vad.is_speech(speech[i : i + frame_bytes], AUDIO_SAMPLE_RATE_HZ)
        )

        assert silence_speech_count == 0, "Silence should have no speech frames"
        assert speech_speech_count > 0, "Speech should have some speech frames"


class TestVoskIntegration:
    """Integration tests for Vosk speech recognition."""

    @pytest.fixture
    def vosk_model(self):
        """Load the real Vosk model for testing."""
        from vosk import Model as VoskModel
        from src.models.vosk import get_vosk_model

        logger = Mock()
        model_path = get_vosk_model("vosk-model-small-en-us-0.15", logger)
        return VoskModel(model_path)

    def test_vosk_model_loads(self, vosk_model):
        """Test that Vosk model loads successfully."""
        assert vosk_model is not None

    def test_vosk_recognizer_processes_audio(self, vosk_model):
        """Test that Vosk recognizer can process audio without errors."""
        from vosk import KaldiRecognizer
        import json

        recognizer = KaldiRecognizer(vosk_model, AUDIO_SAMPLE_RATE_HZ)
        silence = generate_silence(1000)  # 1 second

        recognizer.AcceptWaveform(silence)
        result = json.loads(recognizer.FinalResult())

        assert "text" in result

    def test_vosk_grammar_mode(self, vosk_model):
        """Test Vosk with grammar constraint."""
        from vosk import KaldiRecognizer
        import json

        grammar = json.dumps(["robot", "hello"])
        recognizer = KaldiRecognizer(vosk_model, AUDIO_SAMPLE_RATE_HZ, grammar)
        recognizer.SetWords(True)

        silence = generate_silence(500)
        recognizer.AcceptWaveform(silence)
        result = json.loads(recognizer.FinalResult())

        assert "text" in result

    def test_vosk_recognizes_speech(self, vosk_model):
        """Test Vosk recognizes actual speech audio."""
        from vosk import KaldiRecognizer
        import json
        from tests.fixtures.speech_audio import get_speech_audio

        recognizer = KaldiRecognizer(vosk_model, AUDIO_SAMPLE_RATE_HZ)
        speech = get_speech_audio("hello robot")

        recognizer.AcceptWaveform(speech)
        result = json.loads(recognizer.FinalResult())

        assert "text" in result
        # Should recognize something (may not be exact due to TTS quality)
        assert len(result["text"]) > 0

    def test_vosk_recognizes_wake_word(self, vosk_model):
        """Test Vosk recognizes the wake word 'robot' in speech."""
        from vosk import KaldiRecognizer
        import json
        from tests.fixtures.speech_audio import get_speech_audio

        recognizer = KaldiRecognizer(vosk_model, AUDIO_SAMPLE_RATE_HZ)
        speech = get_speech_audio("robot turn on the lights")

        recognizer.AcceptWaveform(speech)
        result = json.loads(recognizer.FinalResult())

        assert "text" in result
        text = result["text"].lower()
        # Check if 'robot' or similar is recognized
        assert any(word in text for word in ["robot", "robots", "robert"])


class TestSpeechRecognition:
    """Integration tests using actual speech audio."""

    @pytest.fixture
    def vosk_model(self):
        """Load the real Vosk model for testing."""
        from vosk import Model as VoskModel
        from src.models.vosk import get_vosk_model

        logger = Mock()
        model_path = get_vosk_model("vosk-model-small-en-us-0.15", logger)
        return VoskModel(model_path)

    def test_recognizes_hello(self, vosk_model):
        """Test recognition of 'hello'."""
        from vosk import KaldiRecognizer
        import json
        from tests.fixtures.speech_audio import get_speech_audio

        recognizer = KaldiRecognizer(vosk_model, AUDIO_SAMPLE_RATE_HZ)
        speech = get_speech_audio("hello")

        recognizer.AcceptWaveform(speech)
        result = json.loads(recognizer.FinalResult())

        assert "hello" in result["text"].lower()

    def test_recognizes_hey_robot(self, vosk_model):
        """Test recognition of 'hey robot'."""
        from vosk import KaldiRecognizer
        import json
        from tests.fixtures.speech_audio import get_speech_audio

        recognizer = KaldiRecognizer(vosk_model, AUDIO_SAMPLE_RATE_HZ)
        speech = get_speech_audio("hey robot")

        recognizer.AcceptWaveform(speech)
        result = json.loads(recognizer.FinalResult())

        text = result["text"].lower()
        # TTS + Vosk may produce variations
        assert "robot" in text or "robert" in text

    def test_vad_detects_speech_audio(self):
        """Test that VAD detects actual speech as speech."""
        import webrtcvad
        from tests.fixtures.speech_audio import get_speech_audio

        vad = webrtcvad.Vad(1)  # Less aggressive for TTS audio
        speech = get_speech_audio("hello robot")

        # Check VAD on 30ms frames
        frame_size = int(AUDIO_SAMPLE_RATE_HZ * 0.030 * 2)  # 30ms at 16kHz, 2 bytes/sample
        speech_detected = False

        for i in range(0, len(speech) - frame_size, frame_size):
            frame = speech[i : i + frame_size]
            if vad.is_speech(frame, AUDIO_SAMPLE_RATE_HZ):
                speech_detected = True
                break

        assert speech_detected, "VAD should detect speech in TTS audio"


class TestWakeWordFilterIntegration:
    """Integration tests for the full WakeWordFilter pipeline."""

    @pytest.fixture
    def wake_word_filter(self, mock_microphone, integration_env):
        """Create a WakeWordFilter with real Vosk model."""
        config = Mock()
        config.name = "test-filter"

        with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
            mock_struct.return_value = {
                "source_microphone": "mic1",
                "wake_words": ["robot"],
                "vad_aggressiveness": 3,
                "silence_duration_ms": 300,
                "min_speech_ms": 100,
            }

            dependencies = {"mic1": mock_microphone}

            # Create the filter with real components
            filter_instance = WakeWordFilter.new(config, dependencies)
            filter_instance.microphone_client = mock_microphone

            yield filter_instance

            # Cleanup (sync)
            filter_instance.is_shutting_down = True
            filter_instance.executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_get_audio_rejects_wrong_sample_rate(
        self, wake_word_filter, mock_microphone
    ):
        """Test that get_audio validates microphone sample rate."""
        mock_microphone.get_properties.return_value = Mock(
            sample_rate_hz=48000, num_channels=1
        )

        with pytest.raises(ValueError, match="16000 Hz"):
            stream = await wake_word_filter.get_audio("pcm16", 0, 0)
            async for _ in stream:
                pass

    @pytest.mark.asyncio
    async def test_get_audio_rejects_stereo(self, wake_word_filter, mock_microphone):
        """Test that get_audio validates mono audio requirement."""
        mock_microphone.get_properties.return_value = Mock(
            sample_rate_hz=16000, num_channels=2
        )

        with pytest.raises(ValueError, match="mono"):
            stream = await wake_word_filter.get_audio("pcm16", 0, 0)
            async for _ in stream:
                pass

    @pytest.mark.asyncio
    async def test_pause_resume_detection(self, wake_word_filter):
        """Test do_command pause and resume functionality."""
        # Initially detection should be running
        assert wake_word_filter.detection_running is True

        # Pause detection
        result = await wake_word_filter.do_command({"pause_detection": None})
        assert result == {"status": "paused"}
        assert wake_word_filter.detection_running is False

        # Resume detection
        result = await wake_word_filter.do_command({"resume_detection": None})
        assert result == {"status": "resumed"}
        assert wake_word_filter.detection_running is True

    @pytest.mark.asyncio
    async def test_silence_does_not_yield_audio(
        self, wake_word_filter, mock_microphone
    ):
        """Test that silence-only input doesn't yield any audio chunks."""
        # Create stream of silence chunks
        silence_chunks = [generate_silence(100) for _ in range(10)]

        async def silence_stream():
            for chunk in silence_chunks:
                yield create_audio_chunk(chunk)
            # Signal end
            wake_word_filter.is_shutting_down = True

        mock_microphone.get_audio.return_value = silence_stream()

        stream = await wake_word_filter.get_audio("pcm16", 0, 0)
        chunks_received = []

        async for chunk in stream:
            # Skip the initial empty chunk (stream ready signal)
            if chunk.audio.audio_data == b"" and len(chunks_received) == 0:
                continue
            chunks_received.append(chunk)

        # Should not yield any speech chunks for pure silence
        assert len(chunks_received) == 0


class TestCheckForWakeWordIntegration:
    """Integration tests for wake word detection with real Vosk."""

    @pytest.fixture
    def wake_word_filter_with_vosk(self, mock_microphone, integration_env):
        """Create WakeWordFilter with real Vosk for wake word testing."""
        config = Mock()
        config.name = "test-filter"

        with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
            mock_struct.return_value = {
                "source_microphone": "mic1",
                "wake_words": ["hello", "robot"],
                "vad_aggressiveness": 3,
                "use_grammar": False,  # Use full transcription for testing
            }

            dependencies = {"mic1": mock_microphone}
            return WakeWordFilter.new(config, dependencies)

    def test_check_for_wake_word_with_silence(self, wake_word_filter_with_vosk):
        """Test that silence doesn't trigger wake word detection."""
        silence = generate_silence(1000)  # 1 second of silence

        result = wake_word_filter_with_vosk._check_for_wake_word(silence)

        assert result is False

    def test_check_for_wake_word_returns_bool(self, wake_word_filter_with_vosk):
        """Test that _check_for_wake_word always returns a boolean."""
        from tests.fixtures.speech_audio import get_speech_audio

        # Test with various audio inputs
        test_inputs = [
            generate_silence(500),
            get_speech_audio("hello"),
            get_speech_audio("robot"),
        ]

        for audio in test_inputs:
            result = wake_word_filter_with_vosk._check_for_wake_word(audio)
            assert isinstance(result, bool)


class TestFuzzyMatcherIntegration:
    """Integration tests for fuzzy wake word matching."""

    @pytest.fixture
    def filter_with_fuzzy(self, mock_microphone, integration_env):
        """Create WakeWordFilter with fuzzy matching enabled."""
        config = Mock()
        config.name = "test-filter"

        with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
            mock_struct.return_value = {
                "source_microphone": "mic1",
                "wake_words": ["hey robot"],
                "fuzzy_threshold": 2,
                "use_grammar": False,
            }

            dependencies = {"mic1": mock_microphone}
            return WakeWordFilter.new(config, dependencies)

    def test_fuzzy_matcher_initialized(self, filter_with_fuzzy):
        """Test that fuzzy matcher is properly initialized."""
        assert filter_with_fuzzy.fuzzy_matcher is not None
        assert filter_with_fuzzy.fuzzy_matcher.threshold == 2

    def test_fuzzy_matcher_matches_similar_phrases(self, filter_with_fuzzy):
        """Test fuzzy matching with similar phrases."""
        matcher = filter_with_fuzzy.fuzzy_matcher

        # Should match (within threshold) - returns dict with match details
        result = matcher.match("the robot say something", "hey robot")
        assert result is not None
        assert result["distance"] <= 2

        result = matcher.match("hey robert what time", "hey robot")
        assert result is not None
        assert result["distance"] <= 2

        # Should not match (exceeds threshold) - returns None
        result = matcher.match("a robot turn on", "hey robot")
        assert result is None


class TestBufferLimits:
    """Integration tests for buffer size limits."""

    def test_max_buffer_size_constant(self):
        """Verify MAX_BUFFER_SIZE_BYTES is set correctly."""
        assert MAX_BUFFER_SIZE_BYTES == 480000  # ~15 seconds at 16kHz

    def test_buffer_size_calculation(self):
        """Test that buffer size calculations are correct."""
        # 16kHz * 2 bytes per sample = 32000 bytes per second
        bytes_per_second = AUDIO_SAMPLE_RATE_HZ * 2
        assert bytes_per_second == 32000

        # MAX_BUFFER_SIZE_BYTES should be ~15 seconds
        max_seconds = MAX_BUFFER_SIZE_BYTES / bytes_per_second
        assert 14 <= max_seconds <= 16


class TestDoCommandIntegration:
    """Integration tests for do_command functionality."""

    @pytest.fixture
    def filter_instance(self, mock_microphone, integration_env):
        """Create a basic WakeWordFilter instance."""
        config = Mock()
        config.name = "test-filter"

        with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
            mock_struct.return_value = {
                "source_microphone": "mic1",
                "wake_words": ["robot"],
            }

            dependencies = {"mic1": mock_microphone}
            return WakeWordFilter.new(config, dependencies)

    @pytest.mark.asyncio
    async def test_unknown_command_raises_error(self, filter_instance):
        """Test that unknown commands raise ValueError."""
        with pytest.raises(ValueError, match="Unknown command"):
            await filter_instance.do_command({"unknown_command": None})

    @pytest.mark.asyncio
    async def test_pause_resume_cycle(self, filter_instance):
        """Test multiple pause/resume cycles."""
        for _ in range(3):
            # Pause
            result = await filter_instance.do_command({"pause_detection": None})
            assert result["status"] == "paused"
            assert filter_instance.detection_running is False

            # Resume
            result = await filter_instance.do_command({"resume_detection": None})
            assert result["status"] == "resumed"
            assert filter_instance.detection_running is True
