import dataclasses
from enum import Enum, auto


class _SpeechState(Enum):
    IDLE = auto()  # waiting for speech onset
    ACTIVE = auto()  # VAD active, buffering speech frames
    TRAILING = auto()  # silence after speech, waiting for timeout


@dataclasses.dataclass
class _SegmentThresholds:
    """Frame-count thresholds for speech segment detection."""

    max_silence_frames: int
    min_speech_frames: int


@dataclasses.dataclass
class _SpeechSegment:
    """Buffers and counters for an active speech segment."""

    speech_frames: int = 0
    silence_frames: int = 0
    oww_detected: bool = False
    speech_chunk_buffer: list = dataclasses.field(default_factory=list)
    speech_buffer: bytearray = dataclasses.field(default_factory=bytearray)
    audio_buffer: bytearray = dataclasses.field(default_factory=bytearray)
    oww_audio_buffer: bytearray = dataclasses.field(default_factory=bytearray)

    def reset(self) -> None:
        """Clear all buffers and counters."""
        self.speech_frames = 0
        self.silence_frames = 0
        self.oww_detected = False
        self.speech_chunk_buffer.clear()
        self.speech_buffer.clear()
        self.audio_buffer.clear()
        self.oww_audio_buffer.clear()
