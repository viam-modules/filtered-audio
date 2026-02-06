#!/usr/bin/env python3
"""
Pre-generate audio fixtures for CI.

Run this script locally to generate the test audio files before committing.
This allows tests to run in CI without internet access.

Usage:
    python tests/fixtures/generate_audio.py
"""

import io
from pathlib import Path

# Phrases used in integration tests
TEST_PHRASES = [
    "hello",
    "robot",
    "hey robot",
    "hello robot",
    "robot turn on the lights",
    "hello robot turn on the lights",
    "hello how are you today",
]

FIXTURES_DIR = Path(__file__).parent
AUDIO_DIR = FIXTURES_DIR / "audio"
SAMPLE_RATE = 16000


def generate_audio_file(text: str, output_path: Path) -> None:
    """Generate a PCM16 audio file from text using gTTS."""
    from gtts import gTTS
    from pydub import AudioSegment

    print(f"  Generating: {text!r}")

    # Generate speech with gTTS
    tts = gTTS(text=text, lang="en")
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)

    # Convert to PCM16 at 16kHz mono
    audio = AudioSegment.from_mp3(mp3_buffer)
    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)

    # Export as raw PCM
    pcm_buffer = io.BytesIO()
    audio.export(pcm_buffer, format="raw")

    output_path.write_bytes(pcm_buffer.getvalue())
    print(f"    -> {output_path.name} ({output_path.stat().st_size} bytes)")


def main():
    """Generate all test audio files."""
    print("Generating test audio fixtures...")
    print()

    # Create audio directory
    AUDIO_DIR.mkdir(exist_ok=True)

    for phrase in TEST_PHRASES:
        # Create safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in phrase.lower())[:50]
        output_path = AUDIO_DIR / f"{safe_name}.raw"

        generate_audio_file(phrase, output_path)

    print()
    print(f"Generated {len(TEST_PHRASES)} audio files in {AUDIO_DIR}")
    print()
    print("Don't forget to commit the audio/ directory for CI!")


if __name__ == "__main__":
    main()
