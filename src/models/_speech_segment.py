import dataclasses
from enum import Enum, auto


class _SpeechState(Enum):
    IDLE = auto()  # waiting for speech to start
    ACTIVE = auto()  # speech active, buffering frames
    TRAILING = auto()  # silence after speech, waiting for timeout


@dataclasses.dataclass
class _SegmentThresholds:
    """Frame-count thresholds for VAD detection."""

    max_silence_frames: int
    min_speech_frames: int


@dataclasses.dataclass
class _SpeechSegment:
    """Buffers and counters for an active speech segment."""

    speech_frames: int = 0
    silence_frames: int = 0
    oww_detected: bool = False

    # Original AudioChunk objects from the mic stream — yielded downstream on wake word detection
    speech_chunk_buffer: list = dataclasses.field(default_factory=list)

    # Raw PCM bytes of VAD-active frames — passed to Vosk for inference
    speech_buffer: bytearray = dataclasses.field(default_factory=bytearray)

    # Raw PCM bytes fed to OWW — every frame regardless of VAD state (OWW needs continuous audio)
    oww_audio_buffer: bytearray = dataclasses.field(default_factory=bytearray)

    def reset(self) -> None:
        """Clear all buffers and counters."""
        self.speech_frames = 0
        self.silence_frames = 0
        self.oww_detected = False
        self.speech_chunk_buffer.clear()
        self.speech_buffer.clear()
        self.oww_audio_buffer.clear()
