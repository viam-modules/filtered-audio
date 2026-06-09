# Module filtered-audio

filtered-audio module provides a model to filter audio input from a source microphone based on
wake words.

## Supported Platforms
- **Darwin ARM64**
- **Linux x64**
- **Linux ARM64**

## Models

This module provides the following model(s):

- [`viam:filtered-audio:wake-word-filter`](#model-viamfiltered-audiowake-word-filter) — filters audio from a source microphone, only forwarding segments that contain a detected wake word.
- [`viam:filtered-audio:wakeword-miss-sensor`](#model-viamfiltered-audiowakeword-miss-sensor) — captures wake-word near-miss audio + metadata to the Viam Data tab for debugging and training-data collection.

## Model viam:filtered-audio:wake-word-filter
### Configuration
The following attribute template can be used to configure this model:

For `vosk`:
```json
{
  "source_microphone" : <AUDIO_IN NAME>,
  "wake_words": ["<word>"]
}
```

For `openwakeword`:
```json
{
  "source_microphone": <AUDIO_IN NAME>,
  "detection_engine": "openwakeword",
  "oww_model_path": "<path or URL to .onnx model>"
}
```

#### Configuration Attributes

The following attributes are available for the `viam:filtered-audio:wake-word-filter` model:

| Name          | Type   | Inclusion | Description                |
|---------------|--------|-----------|----------------------------|
| `source_microphone` | string | **Required** | Name of a Viam AudioIn component to recieve and filter audio from.
| `detection_engine` | string | **Optional** | Wake word detection engine to use. Options: `vosk`, `openwakeword`. Default: `vosk`.
| `vad_aggressiveness` | int | **Optional** | Sensitivity of the webRTC VAD (voice activity detection). A higher number is more restrictive in reporting speech, and missed detection rates go up. A lower number is less restrictive but may report background noise as speech. Range: 0-3. Default: 3.
| `silence_duration_ms` | int | **Optional** | Milliseconds of continuous silence needed before speech is considered finished. Default: 900
| `min_speech_ms` | int | **Optional** | The minimum length (in milliseconds) a speech segment must be before it is treated as valid speech. Shorter sounds are ignored. Default: 300

#### Vosk Attributes

| Name          | Type   | Inclusion | Description                |
|---------------|--------|-----------|----------------------------|
| `wake_words` | string array | **Required** | Wake words to filter speech. All speech segments said after the wake words will be returned from get_audio.
| `vosk_model` | string | **Optional** | Vosk model to use for speech recognition. Accepts a model name, directory path, or zip file path. Default: `vosk-model-small-en-us-0.15`. See [list](https://alphacephei.com/vosk/models) of available models. For models larger than 1GB, download manually and provide the file path.
| `use_grammar` | bool | **Optional** | When true, Vosk uses grammar-constrained recognition limited to wake words for better accuracy with short wake words. When false, uses full transcription mode which has higher accuracy for longer wake phrases (3+ words). Default: true
| `vosk_grammar_confidence` | float | **Optional** | Minimum confidence threshold (0.0-1.0) for wake word recognition. Lower confidence matches will be rejected. Default: 0.7
| `fuzzy_threshold` | int | **Optional** | Enable fuzzy wake word matching. The threshold (0-5) is the maximum number of character edits (insertions, deletions, substitutions) allowed between the transcript and wake word. If not set, exact matching is used. Note use_grammar must be set to false to use fuzzy matching.

#### OpenWakeWord Attributes

These attributes apply when `detection_engine` is set to `openwakeword`.

| Name          | Type   | Inclusion | Description                |
|---------------|--------|-----------|----------------------------|
| `oww_model_path` | string | **Required** | Path or URL to a custom `.onnx` wakeword model file. Local paths and HTTP/HTTPS URLs are supported. URL models are downloaded and cached in `VIAM_MODULE_DATA`.
| `oww_threshold` | float | **Optional** | Detection confidence threshold (0.0-1.0). A higher value requires more confidence before triggering, reducing false positives. Default: 0.5
| `wakeword_miss_sensor` | string | **Optional** | Resource name of a `viam:filtered-audio:wakeword-miss-sensor` to receive near-miss captures (audio segments where OWW's peak score crossed `near_miss_threshold` but stayed below `oww_threshold`). See [the sensor section](#model-viamfiltered-audiowakeword-miss-sensor). If unset, miss capture is disabled.
| `near_miss_threshold` | float | **Optional** | Lower bound for the "near miss" band (0.0-1.0). A captured segment must satisfy `near_miss_threshold ≤ max_oww_score < oww_threshold`. Required for miss capture to fire; if unset, no captures happen even when the sensor dep is set.


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

### Training a Custom OpenWakeWord Model

To use `detection_engine: openwakeword` you need a custom `.onnx` model trained on your wake word. Use the [openWakeWord automatic training notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb) to generate one.

Once trained, set `oww_model_path` to the local path or a URL pointing to the `.onnx` file.

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

### get_audio()

The wake word filter implements the AudioIn `get_audio()` method:


#### Parameters
- **codec**: Must be `"pcm16"`. Other codecs are not supported.
- **duration_seconds**: Use `0` for continuous streaming
- **previous_timestamp_ns**: Use `0` to start from current time.

#### Stream Behavior

The filter returns a continuous stream that:

1. Monitors continuously for wake words using VAD (Voice Activity Detection) and Vosk speech recognition
2. Only yields chunks when a wake word is detected followed by speech
3. Uses empty chunks to signal speech segment boundaries

**Stream Protocol:**
- **Normal chunks**: Contain audio data (16kHz mono PCM16) for detected speech segments
- **Empty chunks**: Signal the end of a speech segment (`audio_data` has length 0)

After yielding a speech segment and empty chunk, the filter resumes listening for the next wake word automatically.

#### Example Usage

**Basic accumulation and processing:**
```python
# Get continuous stream
audio_stream = await filter.get_audio("pcm16", 0, 0)

segment = bytearray()

async for chunk in audio_stream:
    audio_data = chunk.audio.audio_data

    if len(audio_data) == 0:
        # Empty chunk = segment ended
        if segment:
            process_speech_segment(bytes(segment))
            segment.clear()
    else:
        # Normal chunk - accumulate audio
        segment.extend(audio_data)
```

 Clients should continue consuming chunks even while processing previous segments to avoid stream disconnection.

 See examples/ directory for complete usage examples.

### Do command

The wake word filter supports `do_command()` for pausing and resuming detection. This is useful for voice assistants that need to prevent the filter from detecting its own TTS (text-to-speech) output.

#### Supported Commands

| Command | Description |
|---------|-------------|
| `pause_detection` | Pauses wake word detection. Audio is still consumed but not processed. |
| `resume_detection` | Resumes wake word detection. |

#### Example Usage

```python
# Pause detection before playing TTS audio
await filter.do_command({"pause_detection": None})

await audio_output.play(audio_data)

# Resume detection after TTS finishes
await filter.do_command({"resume_detection": None})
```

## Model viam:filtered-audio:wakeword-miss-sensor

A Viam sensor component that captures wake-word near-miss audio segments from
a `wake-word-filter` instance (OWW only) and uploads them to the Viam Data
tab — binary WAV in the binary store, tabular metadata row joined by
`binary_data_id`. Useful for debugging false negatives
and for building negative/positive training data for a new wake-word
model.

### How it fires

For each completed VAD segment where OWW didn't detect the wake word, the
filter inspects OWW's `prediction_buffer` for the segment's peak score. If
`near_miss_threshold ≤ max_oww_score < oww_threshold`, the filter builds a
reading and calls the sensor's in-process `push_miss(...)`. The sensor
uploads the WAV via `binary_data_capture_upload` and queues the tabular
reading; the next `get_readings()` poll returns it.

When the queue is empty, `get_readings()` raises `NoCaptureToStoreError`,
which the Viam data manager treats as "skip this poll" — no blank rows.

### Configuration

```json
{
  "dataset_ids": ["<viam dataset id>"],
  "component_name": "<override>",
  "max_queue_size": 1000
}
```

| Name             | Type     | Inclusion | Description                                                                                                                |
|------------------|----------|-----------|----------------------------------------------------------------------------------------------------------------------------|
| `dataset_ids`    | string[] | Optional  | Dataset IDs attached to every uploaded WAV — routes captures into a named Viam dataset for training/evaluation export.     |
| `component_name` | string   | Optional  | Overrides the component name in upload metadata. Defaults to the sensor's own resource name.                               |
| `max_queue_size` | int      | Optional  | Soft cap on pending readings. When exceeded, the oldest reading is dropped with a warning. Defaults to `1000`.              |

### Environment variables

The sensor reads `VIAM_API_KEY`, `VIAM_API_KEY_ID`, and `VIAM_MACHINE_PART_ID`
to build a Viam app client for binary uploads. The Viam module manager sets
these automatically. If they're missing (e.g. local dev), the sensor still
runs in "tabular-only" mode — readings are queued with an empty
`binary_data_id`.

### Data manager configuration

The sensor implements `get_readings()` and is meant to be polled by the Viam
data manager. Both **capture and sync** must be enabled on the sensor.

### Tabular reading shape

| Field            | Type        | Notes                                                                                |
|------------------|-------------|--------------------------------------------------------------------------------------|
| `capture_id`     | string      | UUID generated by the filter at miss time.                                           |
| `binary_data_id` | string      | Returned by `binary_data_capture_upload`. Empty if the upload failed.                |
| `wake_word`      | string      | OWW model name (e.g. `"gambit"`) — the wake word this filter was listening for.      |
| `max_oww_score`  | float       | Peak OWW score across the segment.                                                   |
| `oww_threshold`  | float       | The `oww_threshold` that was active when the miss happened.                          |
| `oww_model_path` | string      | Path or URL of the OWW model in use.                                                 |
| `audio_bytes`    | int         | Raw PCM size of the captured segment.                                                |
| `duration_ms`    | float       | Segment duration in milliseconds.                                                    |
| `created_at`     | RFC3339Nano | Set by the filter at miss time.                                                      |
| `captured_at`    | RFC3339Nano | Set by the sensor when the row was appended to the queue.                            |

The audio is always 16 kHz mono PCM16 (the only format the wake-word filter
accepts), so sample rate / channels / encoding are not stored on the row.

### Binary upload tags

Every uploaded WAV is tagged with:

- `wakeword_miss` — coarse "all near-miss captures" filter
- `capture_<capture_id>` — exact one-record lookup of a single miss
- `wake_<wake_word>` — bucket by which wake word was active

### Example configuration

```json
{
  "components": [
    {
      "name": "filter",
      "model": "viam:filtered-audio:wake-word-filter",
      "type": "audio_in",
      "attributes": {
        "source_microphone": "microphone-2",
        "detection_engine": "openwakeword",
        "oww_model_path": "/path/to/gambit.onnx",
        "oww_threshold": 0.8,
        "wakeword_miss_sensor": "miss-sensor",
        "near_miss_threshold": 0.5
      }
    },
    {
      "name": "miss-sensor",
      "model": "viam:filtered-audio:wakeword-miss-sensor",
      "type": "sensor",
      "attributes": {
        "dataset_ids": ["my-wake-training-dataset"]
      }
    }
  ],
  "services": [
    {
      "name": "data_manager-1",
      "type": "data_manager",
      "attributes": {
        "capture_disabled": false,
        "sync_disabled": false,
        "capture_methods": [
          {
            "name": "miss-sensor",
            "method": "Readings",
            "capture_frequency_hz": 1
          }
        ]
      }
    }
  ]
}
```

### Finding a capture

In the Viam Data tab, filter binary records by tag `wakeword_miss` (or by
`capture_<id>` for one specific row's WAV). Or use the CLI:

```bash
viam data export binary ids \
  --org-id <ORG_ID> \
  --destination ./out \
  --binary-data-ids "<the binary_data_id from the row>"
```

To MQL-query the tabular rows:

```javascript
[
  { "$match": { "data.readings.max_oww_score": { "$gte": 0.7 } } }
]
```
