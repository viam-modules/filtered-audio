"""
Integration tests for wake word filter.

These tests verify the full get_audio() pipeline with real Vosk and VAD.
"""

import struct
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.wake_word_filter import (
    AUDIO_SAMPLE_RATE_HZ,
    WakeWordFilter,
)
from tests.fixtures.speech_audio import get_speech_audio
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
    """Patch microphone dependency lookup for integration tests."""
    with (
        patch(
            "src.models.wake_word_filter.AudioIn.get_resource_name",
            return_value="mic1",
        ),
        patch(
            "src.models.vosk.os.getenv",
            return_value="/tmp/module-data",
        ),
    ):
        yield


class TestGetAudioIntegration:
    """Integration tests for the full get_audio() pipeline."""

    @pytest.fixture
    def wake_word_filter(self, mock_microphone, integration_env):
        """Create a WakeWordFilter with real Vosk model and VAD."""
        config = Mock()
        config.name = "test-filter"

        with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
            mock_struct.return_value = {
                "source_microphone": "mic1",
                "wake_words": ["okay robot"],
                "vad_aggressiveness": 3,
                "silence_duration_ms": 300,
                "min_speech_ms": 100,
            }

            dependencies = {"mic1": mock_microphone}
            filter_instance = WakeWordFilter.new(config, dependencies)
            filter_instance.microphone_client = mock_microphone

            yield filter_instance

            # Cleanup
            filter_instance.is_shutting_down = True
            filter_instance.executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_silence_does_not_yield_audio(
        self, wake_word_filter, mock_microphone
    ):
        """Test that silence-only input doesn't yield any audio chunks."""
        silence_chunks = [generate_silence(100) for _ in range(10)]

        async def silence_stream():
            for chunk in silence_chunks:
                yield create_audio_chunk(chunk)
            wake_word_filter.is_shutting_down = True

        mock_microphone.get_audio.return_value = silence_stream()

        stream = await wake_word_filter.get_audio("pcm16", 0, 0)
        chunks_received = []

        async for chunk in stream:
            chunks_received.append(chunk)

        assert len(chunks_received) == 0

    @pytest.mark.asyncio
    async def test_wake_word_yields_audio_chunks(
        self, wake_word_filter, mock_microphone
    ):
        """Test full pipeline: wake word audio → VAD → Vosk → yields chunks."""
        wake_word_audio = get_speech_audio("okay robot turn on the lights")

        # Split into chunks (simulating mic stream)
        chunk_size = 3200  # 100ms at 16kHz
        audio_chunks = [
            wake_word_audio[i : i + chunk_size]
            for i in range(0, len(wake_word_audio), chunk_size)
        ]
        # Add silence to trigger segment end
        audio_chunks.extend([generate_silence(100) for _ in range(15)])

        async def mic_stream():
            for chunk_data in audio_chunks:
                yield create_audio_chunk(chunk_data)
            wake_word_filter.is_shutting_down = True

        mock_microphone.get_audio.return_value = mic_stream()

        stream = await wake_word_filter.get_audio("pcm16", 0, 0)
        chunks_received = []
        got_empty_chunk = False

        async for chunk in stream:
            if len(chunk.audio.audio_data) == 0:
                got_empty_chunk = True
            else:
                chunks_received.append(chunk)

        assert len(chunks_received) > 0, "Should yield chunks when wake word detected"
        assert got_empty_chunk, "Should yield empty chunk to signal segment end"

    @pytest.mark.asyncio
    async def test_no_wake_word_yields_no_chunks(
        self, wake_word_filter, mock_microphone
    ):
        """Test full pipeline: wake word audio → VAD → no wake word -> yields no chunks"""
        wake_word_audio = get_speech_audio("turn on the lights")

        # Split into chunks (simulating mic stream)
        chunk_size = 3200  # 100ms at 16kHz
        audio_chunks = [
            wake_word_audio[i : i + chunk_size]
            for i in range(0, len(wake_word_audio), chunk_size)
        ]
        # Add silence to trigger segment end
        audio_chunks.extend([generate_silence(100) for _ in range(15)])

        async def mic_stream():
            for chunk_data in audio_chunks:
                yield create_audio_chunk(chunk_data)
            wake_word_filter.is_shutting_down = True

        mock_microphone.get_audio.return_value = mic_stream()

        stream = await wake_word_filter.get_audio("pcm16", 0, 0)
        chunks_received = []

        async for chunk in stream:
            chunks_received.append(chunk)

        assert len(chunks_received) == 0, (
            "Shouldn't yield chunks when no wake word present"
        )

    @pytest.mark.asyncio
    async def test_silence_does_not_trigger_vosk(
        self, wake_word_filter, mock_microphone
    ):
        """Test that silence-only input never calls Vosk recognition."""
        silence_chunks = [generate_silence(100) for _ in range(20)]

        async def silence_stream():
            for chunk in silence_chunks:
                yield create_audio_chunk(chunk)
            wake_word_filter.is_shutting_down = True

        mock_microphone.get_audio.return_value = silence_stream()

        with patch.object(wake_word_filter, "_vosk_check_for_wake_word") as mock_vosk:
            stream = await wake_word_filter.get_audio("pcm16", 0, 0)
            async for _ in stream:
                pass

            mock_vosk.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_segments_detected_separately(
        self, wake_word_filter, mock_microphone
    ):
        """Test that two wake word segments in one stream both get detected."""
        wake_word_audio = get_speech_audio("okay robot turn on the lights")
        chunk_size = 3200  # 100ms at 16kHz

        def speech_chunks():
            return [
                wake_word_audio[i : i + chunk_size]
                for i in range(0, len(wake_word_audio), chunk_size)
            ]

        silence_gap = [generate_silence(100) for _ in range(15)]

        async def mic_stream():
            # First segment
            for chunk_data in speech_chunks():
                yield create_audio_chunk(chunk_data)
            for chunk_data in silence_gap:
                yield create_audio_chunk(chunk_data)
            # Second segment
            for chunk_data in speech_chunks():
                yield create_audio_chunk(chunk_data)
            for chunk_data in silence_gap:
                yield create_audio_chunk(chunk_data)
            wake_word_filter.is_shutting_down = True

        mock_microphone.get_audio.return_value = mic_stream()

        stream = await wake_word_filter.get_audio("pcm16", 0, 0)
        empty_chunk_count = 0

        async for chunk in stream:
            if len(chunk.audio.audio_data) == 0:
                empty_chunk_count += 1

        assert empty_chunk_count == 2, (
            f"Expected 2 empty sentinel chunks (one per segment), got {empty_chunk_count}"
        )

    @pytest.mark.asyncio
    async def test_paused_detection_ignores_wake_word(
        self, wake_word_filter, mock_microphone
    ):
        """Test that wake words during paused detection yield nothing."""
        wake_word_audio = get_speech_audio("okay robot turn on the lights")
        chunk_size = 3200
        audio_chunks = [
            wake_word_audio[i : i + chunk_size]
            for i in range(0, len(wake_word_audio), chunk_size)
        ]
        audio_chunks.extend([generate_silence(100) for _ in range(15)])

        # Pause detection before streaming
        wake_word_filter.detection_running = False

        async def mic_stream():
            for chunk_data in audio_chunks:
                yield create_audio_chunk(chunk_data)
            wake_word_filter.is_shutting_down = True

        mock_microphone.get_audio.return_value = mic_stream()

        stream = await wake_word_filter.get_audio("pcm16", 0, 0)
        chunks_received = []

        async for chunk in stream:
            chunks_received.append(chunk)

        assert len(chunks_received) == 0, (
            "Should yield nothing when detection is paused"
        )
