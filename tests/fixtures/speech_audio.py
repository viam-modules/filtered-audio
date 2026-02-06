"""Load pre-generated speech audio for integration tests."""

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent
AUDIO_DIR = FIXTURES_DIR / "audio"
SAMPLE_RATE = 16000


def _get_safe_filename(text: str) -> str:
    """Convert text to a safe filename."""
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:50]


def get_speech_audio(text: str) -> bytes:
    """
    Load pre-generated speech audio for the given text.

    Returns PCM16 audio at 16kHz mono.

    Args:
        text: The text to get audio for

    Returns:
        Raw PCM16 audio bytes at 16kHz mono

    Raises:
        FileNotFoundError: If no pre-generated audio exists for this text
    """
    safe_name = _get_safe_filename(text)
    audio_file = AUDIO_DIR / f"{safe_name}.raw"

    if not audio_file.exists():
        available = [f.stem for f in AUDIO_DIR.glob("*.raw")]
        raise FileNotFoundError(
            f"No pre-generated audio for {text!r}. "
            f"Run 'python tests/fixtures/generate_audio.py' to generate it. "
            f"Available: {available}"
        )

    return audio_file.read_bytes()
