# Module filtered-audio

filtered-audio module provides a model to filter audio input from a source microphone based on
wake words.

## Supported Platforms
- **Darwin ARM64**
- **Linux x64**
- **Linux ARM64**

## Models

This module provides the following model(s):

- [`viam:filtered-audio:wake-word-filter`]

## Model viam:filtered-audio:wake-word-filter
### Configuration
The following attribute template can be used to configure this model:

```json
{
  "source_microphone" : <AUDIO_IN NAME>,
  "wake_words": ["<word>"]
}
```

``
#### Configuration Attributes

The following attributes are available for the `viam:filtered-audio:wake-word-filter` model:

| Name          | Type   | Inclusion | Description                |
|---------------|--------|-----------|----------------------------|
| `source_microphone` | string | **Required** | Name of a Viam AudioIn component to recieve and filter audio from
| `wake_words` | string array | **Required** | Wake words to filter speech. All speech segments said after the wake words will be returned from get_audio
| `vosk_model` | string | **Optional** | The name of the VOSK model to use for speech to text. Default: vosk-model-small-en-us-0.15. See [list](https://alphacephei.com/vosk/models) of available models.
| `vad_aggressiveness` | int | **Optional** | Sensitivity of the webRTC VAD (voice activity detection). A higher number is more restrictive in reporting speech, and missed detection rates go up. A lower number is less restrictive but may report background noise as speech. Range: 0-3. Default: 3.
| `fuzzy_threshold` | int | **Optional** | Enable fuzzy wake word matching. The threshold (0-5) is the maximum number of character edits (insertions, deletions, substitutions) allowed between the transcript and wake word. If not set, exact matching is used.

### Source Microphone Requirements

The source microphone **must** provide audio in the following format:

| Requirement | Value | Description |
|-------------|-------|-------------|
| **Codec** | PCM16 | 16-bit PCM audio format |
| **Sample Rate** | 16000 Hz | Required for Vosk model |
| **Channels** | 1 (Mono) | Stereo audio is not supported |

**Example configuration for source microphone:**
```json
{
  "name": "my-microphone",
  "type": "audio_in",
  "model": "...",
  "attributes": {
    "sample_rate": 16000,
    "channels": 1
  }
}
```

**Recommended Source Microphone:** Use the [`viam:system-audio`](https://app.viam.com/module/viam/system-audio) module, which supports resampling and can output 16 kHz mono PCM16 audio from any system microphone.

### Fuzzy Wake Word Matching

The wake word filter supports fuzzy matching using Levenshtein distance (edit distance) via the [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) library. This improves accuracy when speech recognition produces slight variations (e.g., "hey robot" transcribed as "the robot").

#### Enabling Fuzzy Matching

To enable fuzzy wake word matching, add `fuzzy_threshold` to your configuration:

```json
{
  "source_microphone": "mic",
  "wake_words": ["hey robot"],
  "fuzzy_threshold": 2
}
```

#### How It Works

Fuzzy matching compares the wake phrase against the first few words of the transcript. It measures how many character edits (insertions, deletions, substitutions) are needed to transform one into the other. This handles common speech-to-text errors like "the robot" being transcribed instead of "hey robot" (2 character changes: t→h, h→e→y):

| Transcribed | Wake Word | Distance | Match (threshold=2) |
|-------------|-----------|----------|---------------------|
| "the robot say something" | "hey robot" | 2 | ✓ |
| "hey Robert what time" | "hey robot" | 2 | ✓ |
| "a robot turn on lights" | "hey robot" | 3 | ✗ |
| "please hey robot do it" | "hey robot" | - | ✗ (not at start) |

#### Threshold Guidelines

| Threshold | Use Case |
|-----------|----------|
| 1 | Very strict - for short wake words or quiet environments |
| 2-3 | Recommended for most wake words |
| 4-5 | Lenient - for noisy environments (may increase false positives) |
