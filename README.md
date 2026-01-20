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
| `fuzzy_threshold` | int | **Optional** | Enable fuzzy wake word matching with specified Levenshtein distance threshold (0-5). If not set, exact matching is used. See [Fuzzy Wake Word Matching](#fuzzy-wake-word-matching).

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

Fuzzy matching uses word-boundary matching to allow wake words to trigger even when transcribed slightly differently, while preventing false matches:

| Transcribed | Wake Word | Distance | Match |
|-------------|-----------|----------|-------|
| "the robot say something" | "hey robot" | 2 | ✓ |
| "a robot turn on lights" | "hey robot" | 2 | ✓ |
| "hey Robert what time" | "hey robot" | 1 | ✓ |
| "they robotic assistant" | "hey robot" | 5 | ✗ |

The word-boundary approach prevents partial-word matches like "robotic" matching "robot".

#### Threshold Guidelines

| Threshold | Use Case |
|-----------|----------|
| 1 | Very strict - for short wake words or quiet environments |
| 2-3 | Recommended for most wake words |
| 4-5 | Lenient - for noisy environments (may increase false positives) |
