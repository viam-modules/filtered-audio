import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.models.wake_word_filter import WakeWordFilter, FRAME_SIZE_BYTES
from src.models._speech_segment import _SpeechState, _SpeechSegment, _SegmentThresholds


@pytest.fixture
def mock_config():
    """Mock Viam ComponentConfig"""
    config = Mock()
    config.attributes.fields = {
        "wake_words": Mock(string_value="robot"),
        "source_microphone": Mock(string_value="mic1"),
        "vad_aggressiveness": Mock(number_value=3),
    }
    return config


@pytest.fixture
def mock_dependencies():
    """Mock microphone dependency"""
    mic = AsyncMock()
    mic.get_properties.return_value = Mock(sample_rate_hz=16000, num_channels=1)
    return {"mic1": mic}


@pytest.fixture
def mock_env():
    """Mock all WakeWordFilter.new() dependencies"""
    with (
        patch("src.models.wake_word_filter.struct_to_dict") as mock_struct,
        patch("src.models.wake_word_filter.setup_vosk") as mock_setup_vosk,
        patch("src.models.wake_word_filter.webrtcvad.Vad") as mock_vad,
        patch("src.models.wake_word_filter.ThreadPoolExecutor") as mock_executor,
        patch(
            "src.models.wake_word_filter.AudioIn.get_resource_name", return_value="mic1"
        ) as mock_resource,
    ):
        yield {
            "struct_to_dict": mock_struct,
            "setup_vosk": mock_setup_vosk,
            "vad": mock_vad,
            "executor": mock_executor,
            "resource_name": mock_resource,
        }


def create_config(attrs):
    """Helper to create mock config with specific attributes"""
    config = Mock()
    with patch("src.models.wake_word_filter.struct_to_dict", return_value=attrs):
        return config


def test_validate_config_success():
    """Test validate_config with valid configuration"""
    config = Mock()
    config.attributes = Mock()

    # Mock struct_to_dict to return our test config
    with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
        mock_struct.return_value = {
            "source_microphone": "mic1",
            "wake_words": ["robot"],
        }

        deps, errors = WakeWordFilter.validate_config(config)

        assert deps == ["mic1"]
        assert not errors


def test_validate_config_missing_microphone():
    """Test validate_config raises error when source_microphone is missing"""
    config = Mock()
    config.attributes = Mock()

    with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
        mock_struct.return_value = {
            "wake_words": ["robot"]
            # source_microphone missing
        }

        with pytest.raises(ValueError, match="source_microphone attribute is required"):
            WakeWordFilter.validate_config(config)


def test_validate_config_empty_microphone():
    """Test validate_config raises error when source_microphone is empty string"""
    config = Mock()
    config.attributes = Mock()

    with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
        mock_struct.return_value = {"source_microphone": "", "wake_words": ["robot"]}

        with pytest.raises(ValueError, match="source_microphone attribute is required"):
            WakeWordFilter.validate_config(config)


def test_validate_config_missing_wake_words():
    """Test validate_config raises error when wake_words is missing for vosk"""
    config = Mock()
    config.attributes = Mock()

    with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
        mock_struct.return_value = {
            "source_microphone": "mic1",
            "detection_engine": "vosk",
            # wake_words missing
        }

        with pytest.raises(
            ValueError,
            match="wake_words is required when using the vosk detection engine",
        ):
            WakeWordFilter.validate_config(config)


def test_validate_config_empty_wake_words():
    """Test validate_config raises error when wake_words is empty for vosk"""
    config = Mock()
    config.attributes = Mock()

    with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
        mock_struct.return_value = {
            "source_microphone": "mic1",
            "wake_words": [],
            "detection_engine": "vosk",
        }

        with pytest.raises(
            ValueError,
            match="wake_words is required when using the vosk detection engine",
        ):
            WakeWordFilter.validate_config(config)


def test_validate_config_oww_missing_wake_words_ok():
    """Test validate_config does not error when wake_words is missing for openwakeword"""
    config = Mock()
    config.attributes = Mock()

    with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
        mock_struct.return_value = {
            "source_microphone": "mic1",
            "detection_engine": "openwakeword",
            "oww_model_path": "/path/to/model.onnx",
        }

        WakeWordFilter.validate_config(config)  # should not raise


def test_validate_config_oww_empty_wake_words_ok():
    """Test validate_config does not error when wake_words is empty for openwakeword"""
    config = Mock()
    config.attributes = Mock()

    with patch("src.models.wake_word_filter.struct_to_dict") as mock_struct:
        mock_struct.return_value = {
            "source_microphone": "mic1",
            "detection_engine": "openwakeword",
            "oww_model_path": "/path/to/model.onnx",
            "wake_words": [],
        }

        WakeWordFilter.validate_config(config)  # should not raise


def test_validate_config_rejects_non_string_wake_words(mock_env):
    """Test validate_config raises error when wake_words contains non-strings"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot", 123, "computer"],
    }

    with pytest.raises(ValueError, match="All wake_words must be strings"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_string_microphone(mock_env):
    """Test validate_config raises error when microphone is non-string"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": 123,
        "wake_words": ["robot", "computer"],
    }

    with pytest.raises(
        ValueError, match="source_microphone attribute must be a string"
    ):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_invalid_vad_aggressiveness(mock_env):
    """Test validate_config raises error when vad_agressiveness invalid"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot", "computer"],
        "vad_aggressiveness": 4,
    }

    with pytest.raises(ValueError, match="vad_aggressiveness must be 0-3, got 4"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_int_vad_agressiveness(mock_env):
    """Test validate_config raises error when vad_agressiveness not an int"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot", "computer"],
        "vad_aggressiveness": "four",
    }

    with pytest.raises(ValueError, match="vad_aggressiveness must be a whole number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_accepts_float_vad_aggressiveness_without_fraction(mock_env):
    """Test validate_config accepts float vad_aggressiveness if no fractional part (e.g. 2.0)"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "vad_aggressiveness": 2.0,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_rejects_float_vad_aggressiveness_with_fraction(mock_env):
    """Test validate_config rejects float vad_aggressiveness with fractional part (e.g. 2.5)"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "vad_aggressiveness": 2.5,
    }

    with pytest.raises(ValueError, match="vad_aggressiveness must be a whole number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_int_fuzzy_threshold(mock_env):
    """Test validate_config raises error when fuzzy_threshold not an int"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot", "computer"],
        "fuzzy_threshold": "five",
    }

    with pytest.raises(ValueError, match="fuzzy_threshold must be a whole number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_accepts_float_fuzzy_threshold_without_fraction(mock_env):
    """Test validate_config accepts float fuzzy_threshold if no fractional part (e.g. 3.0)"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "fuzzy_threshold": 3.0,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_rejects_float_fuzzy_threshold_with_fraction(mock_env):
    """Test validate_config rejects float fuzzy_threshold with fractional part (e.g. 3.5)"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "fuzzy_threshold": 3.5,
    }

    with pytest.raises(ValueError, match="fuzzy_threshold must be a whole number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_invalid_fuzzy_threshold(mock_env):
    """Test validate_config raises error when fuzzy_threshold not an int"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot", "computer"],
        "fuzzy_threshold": 7,
    }

    with pytest.raises(ValueError, match="fuzzy_threshold must be 0-5, got 7"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_int_silence_duration_ms(mock_env):
    """Test validate_config raises error when silence_duration_ms not an int"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "silence_duration_ms": "five hundred",
    }

    with pytest.raises(ValueError, match="silence_duration_ms must be a whole number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_accepts_float_silence_duration_ms_without_fraction(mock_env):
    """Test validate_config accepts float silence_duration_ms if no fractional part (e.g. 500.0)"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "silence_duration_ms": 500.0,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_rejects_float_silence_duration_ms_with_fraction(mock_env):
    """Test validate_config rejects float silence_duration_ms with fractional part (e.g. 500.5)"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "silence_duration_ms": 500.5,
    }

    with pytest.raises(ValueError, match="silence_duration_ms must be a whole number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_int_min_speech_ms(mock_env):
    """Test validate_config raises error when min_speech_ms not an int"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "min_speech_ms": "three hundred",
    }

    with pytest.raises(ValueError, match="min_speech_ms must be a whole number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_accepts_float_min_speech_ms_without_fraction(mock_env):
    """Test validate_config accepts float min_speech_ms if no fractional part (e.g. 300.0)"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "min_speech_ms": 300.0,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_rejects_float_min_speech_ms_with_fraction(mock_env):
    """Test validate_config rejects float min_speech_ms with fractional part (e.g. 300.5)"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "min_speech_ms": 300.5,
    }

    with pytest.raises(ValueError, match="min_speech_ms must be a whole number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_accepts_valid_silence_duration_ms(mock_env):
    """Test validate_config accepts valid silence_duration_ms"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "silence_duration_ms": 1000,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_accepts_valid_min_speech_ms(mock_env):
    """Test validate_config accepts valid min_speech_ms"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "min_speech_ms": 500,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors



def test_validate_config_rejects_non_bool_use_grammar(mock_env):
    """Test validate_config raises error when use_grammar is not a boolean"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "use_grammar": "true",
    }

    with pytest.raises(ValueError, match="use_grammar must be a boolean"):
        WakeWordFilter.validate_config(config)


def test_validate_config_accepts_use_grammar_true(mock_env):
    """Test validate_config accepts use_grammar=True"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "use_grammar": True,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_accepts_use_grammar_false(mock_env):
    """Test validate_config accepts use_grammar=False"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "use_grammar": False,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_rejects_non_number_grammar_confidence(mock_env):
    """Test validate_config raises error when vosk_grammar_confidence not a number"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "vosk_grammar_confidence": "zero point seven",
    }

    with pytest.raises(ValueError, match="vosk_grammar_confidence must be a number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_grammar_confidence_too_low(mock_env):
    """Test validate_config raises error when vosk_grammar_confidence below 0"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "vosk_grammar_confidence": -0.1,
    }

    with pytest.raises(ValueError, match="vosk_grammar_confidence must be 0.0-1.0"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_grammar_confidence_too_high(mock_env):
    """Test validate_config raises error when vosk_grammar_confidence above 1"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "vosk_grammar_confidence": 1.5,
    }

    with pytest.raises(ValueError, match="vosk_grammar_confidence must be 0.0-1.0"):
        WakeWordFilter.validate_config(config)


def test_validate_config_accepts_valid_grammar_confidence(mock_env):
    """Test validate_config accepts valid vosk_grammar_confidence"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "vosk_grammar_confidence": 0.8,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


# Tests for WakeWordFilter.new()


def test_new_with_single_wake_word_string(mock_env):
    """Test new() parses single wake word string correctly"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": "robot",
        "vad_aggressiveness": 3,
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.wake_words == ["robot"]


def test_new_with_wake_word_list(mock_env):
    """Test new() parses wake word list correctly"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["hey robot", "OK Robot", "Computer"],
        "vad_aggressiveness": 3,
    }

    dependencies["mic1"] = mic
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.wake_words == ["hey robot", "ok robot", "computer"]


def test_new_uses_default_vad_aggressiveness(mock_env):
    """Test new() uses default VAD aggressiveness of 3"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        # vad_aggressiveness not specified
    }

    dependencies["mic1"] = mic
    WakeWordFilter.new(config, dependencies)

    # Verify VAD was initialized with default value 3
    mock_env["vad"].assert_called_once_with(3)


def test_new_uses_custom_vad_aggressiveness(mock_env):
    """Test new() uses custom VAD aggressiveness when provided"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "vad_aggressiveness": 1,
    }

    dependencies["mic1"] = mic
    WakeWordFilter.new(config, dependencies)

    # Verify VAD was initialized with custom value 1
    mock_env["vad"].assert_called_once_with(1)


def test_new_loads_vosk_model(mock_env):
    """Test new() calls setup_vosk for vosk engine"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "vosk_model": "vosk-model-en-us-0.22",
    }

    dependencies["mic1"] = mic
    WakeWordFilter.new(config, dependencies)

    # Verify setup_vosk was called
    mock_env["setup_vosk"].assert_called_once()


def test_new_sets_microphone_client(mock_env):
    """Test new() sets microphone_client from dependencies"""
    config = Mock()
    mic = AsyncMock()
    dependencies = {}

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "vosk_model": "vosk-model-en-us-0.22",
    }

    dependencies["mic1"] = mic
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.microphone_client == mic


def test_new_uses_default_silence_duration_ms(mock_env):
    """Test new() uses default silence_duration_ms when not specified"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    # Default is 900ms
    assert instance.silence_duration_ms == 900


def test_new_uses_custom_silence_duration_ms(mock_env):
    """Test new() uses custom silence_duration_ms when provided"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "silence_duration_ms": 1200,
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.silence_duration_ms == 1200


def test_new_uses_default_min_speech_duration_ms(mock_env):
    """Test new() uses default min_speech_duration_ms when not specified"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    # Default is 300ms
    assert instance.min_speech_duration_ms == 300


def test_new_uses_custom_min_speech_duration_ms(mock_env):
    """Test new() uses custom min_speech_ms when provided"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "min_speech_ms": 500,
    }

    dependencies = {"mic1": mic}
    instance = WakeWordFilter.new(config, dependencies)

    assert instance.min_speech_duration_ms == 500


# # Error case tests for WakeWordFilter.new()


def test_new_fails_when_setup_vosk_fails(mock_env):
    """Test new() raises error when setup_vosk fails"""
    config = Mock()
    mic = AsyncMock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
    }

    # Make setup_vosk raise an error
    mock_env["setup_vosk"].side_effect = RuntimeError("Model not found")

    dependencies = {"mic1": mic}

    with pytest.raises(RuntimeError, match="Model not found"):
        WakeWordFilter.new(config, dependencies)


def test_new_calls_setup_vosk_with_attrs(mock_env):
    """Test new() calls setup_vosk with the correct vosk_model kwarg"""
    config = Mock()
    mic = AsyncMock()

    attrs = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "vosk_model": "vosk-model-en-us-0.22",
    }
    mock_env["struct_to_dict"].return_value = attrs

    dependencies = {"mic1": mic}
    WakeWordFilter.new(config, dependencies)

    # Verify setup_vosk was called with the instance and correct vosk_model kwarg
    call_args = mock_env["setup_vosk"].call_args
    assert call_args[1]["vosk_model"] == "vosk-model-en-us-0.22"


def test_new_handles_missing_microphone_in_dependencies(mock_env):
    """Test new() raises error when microphone not in dependencies"""
    config = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
    }

    dependencies = {}  # Empty - no microphone!

    with pytest.raises(KeyError):
        WakeWordFilter.new(config, dependencies)


def test_validate_config_accepts_vosk_engine(mock_env):
    """Test validate_config accepts detection_engine='vosk'"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "detection_engine": "vosk",
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_accepts_openwakeword_engine(mock_env):
    """Test validate_config accepts detection_engine='openwakeword'"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "detection_engine": "openwakeword",
        "oww_model_path": "/path/to/model.onnx",
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_rejects_invalid_detection_engine(mock_env):
    """Test validate_config raises error for invalid detection_engine"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "detection_engine": "whisper",
    }

    with pytest.raises(
        ValueError, match="detection_engine must be 'vosk' or 'openwakeword'"
    ):
        WakeWordFilter.validate_config(config)


def test_validate_config_defaults_to_vosk_engine(mock_env):
    """Test validate_config defaults to vosk when detection_engine not specified"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


# Tests for oww_model_path validation


def test_validate_config_requires_oww_model_path_for_openwakeword(mock_env):
    """Test validate_config raises error when oww_model_path missing for openwakeword engine"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "detection_engine": "openwakeword",
    }

    with pytest.raises(ValueError, match="oww_model_path is required"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_empty_oww_model_path(mock_env):
    """Test validate_config raises error when oww_model_path is empty string"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "detection_engine": "openwakeword",
        "oww_model_path": "",
    }

    with pytest.raises(ValueError, match="oww_model_path is required"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_string_oww_model_path(mock_env):
    """Test validate_config raises error when oww_model_path is not a string"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "detection_engine": "openwakeword",
        "oww_model_path": 123,
    }

    with pytest.raises(ValueError, match="oww_model_path must be a non-empty string"):
        WakeWordFilter.validate_config(config)


# Tests for oww_threshold validation


def test_validate_config_accepts_valid_oww_threshold(mock_env):
    """Test validate_config accepts valid oww_threshold"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "detection_engine": "openwakeword",
        "oww_model_path": "/path/to/model.onnx",
        "oww_threshold": 0.7,
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_rejects_oww_threshold_too_low(mock_env):
    """Test validate_config raises error when oww_threshold below 0"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "detection_engine": "openwakeword",
        "oww_model_path": "/path/to/model.onnx",
        "oww_threshold": -0.1,
    }

    with pytest.raises(ValueError, match="oww_threshold must be 0.0-1.0"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_oww_threshold_too_high(mock_env):
    """Test validate_config raises error when oww_threshold above 1"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "detection_engine": "openwakeword",
        "oww_model_path": "/path/to/model.onnx",
        "oww_threshold": 1.5,
    }

    with pytest.raises(ValueError, match="oww_threshold must be 0.0-1.0"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_number_oww_threshold(mock_env):
    """Test validate_config raises error when oww_threshold is not a number"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "detection_engine": "openwakeword",
        "oww_model_path": "/path/to/model.onnx",
        "oww_threshold": "high",
    }

    with pytest.raises(ValueError, match="oww_threshold must be a number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_oww_threshold_not_validated_for_vosk(mock_env):
    """Test validate_config ignores oww_threshold when engine is vosk"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "detection_engine": "vosk",
        "oww_threshold": 999,  # Invalid but should be ignored
    }

    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


@pytest.mark.asyncio
async def test_get_audio_rejects_non_pcm16_codec():
    """Test get_audio raises error for non-PCM16 codec"""
    wake_word_filter = Mock()
    wake_word_filter.microphone_client = AsyncMock()
    wake_word_filter.logger = Mock()

    with pytest.raises(ValueError, match="PCM16"):
        await WakeWordFilter.get_audio(wake_word_filter, "mp3", 0, 0)


@pytest.mark.asyncio
async def test_get_audio_rejects_invalid_sample_rate(mock_env):
    """Test get_audio raises error when microphone sample rate is not 16000 Hz"""
    import types

    wake_word_filter = Mock()
    wake_word_filter.logger = Mock()
    wake_word_filter._validate_mic_properties = types.MethodType(
        WakeWordFilter._validate_mic_properties, wake_word_filter
    )

    # Mock microphone with wrong sample rate
    mic = AsyncMock()
    mic.get_properties.return_value = Mock(sample_rate_hz=48000, num_channels=1)
    wake_word_filter.microphone_client = mic

    with pytest.raises(ValueError, match="16000 Hz"):
        stream = await WakeWordFilter.get_audio(wake_word_filter, "pcm16", 0, 0)
        # Consume the generator to trigger the validation
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_get_properties_returns_mic_properties():
    """Test get_properties returns properties from underlying microphone"""
    wake_word_filter = Mock()
    wake_word_filter.logger = Mock()

    # Mock microphone with specific properties
    mic = AsyncMock()
    mic.get_properties.return_value = Mock(sample_rate_hz=16000, num_channels=1)
    wake_word_filter.microphone_client = mic

    result = await WakeWordFilter.get_properties(wake_word_filter)

    assert result.sample_rate_hz == 16000
    assert result.num_channels == 1


@pytest.mark.asyncio
async def test_get_properties_always_reports_pcm16():
    """Test get_properties always reports PCM16 as supported codec"""
    wake_word_filter = Mock()
    wake_word_filter.logger = Mock()

    mic = AsyncMock()
    mic.get_properties.return_value = Mock(sample_rate_hz=48000, num_channels=2)
    wake_word_filter.microphone_client = mic

    result = await WakeWordFilter.get_properties(wake_word_filter)

    assert result.supported_codecs == ["pcm16"]


@pytest.mark.asyncio
async def test_get_audio_rejects_stereo_audio(mock_env):
    """Test get_audio raises error when microphone is stereo (2 channels)"""
    import types

    wake_word_filter = Mock()
    wake_word_filter.logger = Mock()
    wake_word_filter._validate_mic_properties = types.MethodType(
        WakeWordFilter._validate_mic_properties, wake_word_filter
    )

    # Mock microphone with stereo audio
    mic = AsyncMock()
    mic.get_properties.return_value = Mock(sample_rate_hz=16000, num_channels=2)
    wake_word_filter.microphone_client = mic

    with pytest.raises(ValueError, match="mono.*1 channel"):
        stream = await WakeWordFilter.get_audio(wake_word_filter, "pcm16", 0, 0)
        # Consume the generator to wake_word_filter the validation
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_close_sets_shutdown_flag():
    """Test close() sets is_shutting_down flag"""
    wake_word_filter = Mock()
    wake_word_filter.is_shutting_down = False
    wake_word_filter.executor = Mock()

    await WakeWordFilter.close(wake_word_filter)

    assert wake_word_filter.is_shutting_down is True
    wake_word_filter.executor.shutdown.assert_called_once_with(wait=True)


@pytest.mark.asyncio
async def test_close_shuts_down_executor():
    """Test close() properly shuts down thread pool executor"""
    wake_word_filter = Mock()
    wake_word_filter.is_shutting_down = False
    mock_executor = Mock()
    wake_word_filter.executor = mock_executor

    await WakeWordFilter.close(wake_word_filter)

    mock_executor.shutdown.assert_called_once_with(wait=True)


@pytest.mark.asyncio
async def test_do_command_pause_detection():
    """Test do_command pauses detection"""
    wake_word_filter = Mock()
    wake_word_filter.detection_running = True
    wake_word_filter.logger = Mock()

    result = await WakeWordFilter.do_command(
        wake_word_filter, {"pause_detection": None}
    )

    assert result == {"status": "paused"}
    assert wake_word_filter.detection_running is False


@pytest.mark.asyncio
async def test_do_command_resume_detection():
    """Test do_command resumes detection"""
    wake_word_filter = Mock()
    wake_word_filter.detection_running = False
    wake_word_filter.logger = Mock()

    result = await WakeWordFilter.do_command(
        wake_word_filter, {"resume_detection": None}
    )

    assert result == {"status": "resumed"}
    assert wake_word_filter.detection_running is True


@pytest.mark.asyncio
async def test_do_command_unknown_raises_error():
    """Test do_command raises error for unknown commands"""
    wake_word_filter = Mock()
    wake_word_filter.logger = Mock()

    with pytest.raises(ValueError, match="Unknown command"):
        await WakeWordFilter.do_command(wake_word_filter, {"unknown": None})


# OWW detection tests (via get_audio stream)


def make_audio_chunk(num_bytes=960):
    """Create a mock audio chunk with silence audio data."""
    chunk = Mock()
    chunk.audio = Mock()
    chunk.audio.audio_data = b"\x00" * num_bytes
    return chunk


async def collect_stream(stream):
    """Collect all chunks from an async stream."""
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return chunks


async def async_iter(items):
    """Convert a list to an async iterator."""
    for item in items:
        yield item


def make_oww_filter(threshold=0.5, model_name="okay_gambit"):
    """Create a mock WakeWordFilter configured for OWW."""
    wf = Mock()
    wf.detection_engine = "openwakeword"
    wf.oww_threshold = threshold
    wf.oww_model_name = model_name
    wf.oww_model = Mock()
    wf.oww_model.prediction_buffer = {}
    wf.is_shutting_down = False
    wf.detection_running = True
    wf.vad = Mock()
    wf.silence_duration_ms = 900
    wf.min_speech_duration_ms = 300
    wf.conversation_timeout_seconds = 0.0
    wf._conversation_window_expires_at = 0.0
    wf.logger = Mock()
    wf.microphone_client = AsyncMock()
    # Bind real methods so async-generator calls work correctly
    import types

    wf._validate_mic_properties = types.MethodType(
        WakeWordFilter._validate_mic_properties, wf
    )
    wf._process_vad_frame = types.MethodType(WakeWordFilter._process_vad_frame, wf)
    wf._run_detection = types.MethodType(WakeWordFilter._run_detection, wf)
    wf._finalize_segment = types.MethodType(WakeWordFilter._finalize_segment, wf)
    wf._in_conversation_window = types.MethodType(
        WakeWordFilter._in_conversation_window, wf
    )
    wf._refresh_conversation_window = types.MethodType(
        WakeWordFilter._refresh_conversation_window, wf
    )
    wf._maybe_push_miss = AsyncMock()
    wf.miss_sensor = None
    wf.near_miss_threshold = None
    # Real multi-client machinery: get_audio subscribes to the broadcaster
    # and the shared pipeline task does the detection work.
    from src.models._broadcast import SegmentBroadcaster

    wf._broadcaster = SegmentBroadcaster(wf.logger)
    wf._pipeline_task = None
    wf._ensure_pipeline_running = types.MethodType(
        WakeWordFilter._ensure_pipeline_running, wf
    )
    wf._run_pipeline = types.MethodType(WakeWordFilter._run_pipeline, wf)
    return wf


@pytest.mark.asyncio
async def test_oww_detects_wake_word_above_threshold():
    """Test OWW yields chunks when score is above threshold."""
    wf = make_oww_filter(threshold=0.5)

    wf.vad.is_speech.return_value = True
    wf.oww_model.predict.return_value = {"okay_gambit": 0.9}

    speech_chunks = [make_audio_chunk(960) for _ in range(15)]
    silence_chunks = [make_audio_chunk(960) for _ in range(35)]

    wf.vad.is_speech.side_effect = [True] * 15 + [False] * 35

    wf.microphone_client.get_audio.return_value = async_iter(
        speech_chunks + silence_chunks
    )
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks = await collect_stream(stream)

    assert len(chunks) > 0
    assert chunks[-1].audio.audio_data == b""


@pytest.mark.asyncio
async def test_oww_does_not_yield_when_below_threshold():
    """Test OWW does not yield chunks when score is below threshold."""
    wf = make_oww_filter(threshold=0.5)

    wf.vad.is_speech.side_effect = [True] * 15 + [False] * 35
    wf.oww_model.predict.return_value = {"okay_gambit": 0.1}
    wf.oww_model.prediction_buffer = {"okay_gambit": [0.1]}

    speech_chunks = [make_audio_chunk(960) for _ in range(15)]
    silence_chunks = [make_audio_chunk(960) for _ in range(35)]

    wf.microphone_client.get_audio.return_value = async_iter(
        speech_chunks + silence_chunks
    )
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks = await collect_stream(stream)

    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_oww_no_detection_calls_maybe_push_miss():
    """When OWW doesn't fire on a completed speech segment, _maybe_push_miss
    must be awaited with the buffered speech bytes + the peak prediction-
    buffer score. Guards against the call site being silently dropped."""
    wf = make_oww_filter(threshold=0.5)

    wf.vad.is_speech.side_effect = [True] * 15 + [False] * 35
    wf.oww_model.predict.return_value = {"okay_gambit": 0.3}
    wf.oww_model.prediction_buffer = {"okay_gambit": [0.1, 0.3, 0.2]}

    speech_chunks = [make_audio_chunk(960) for _ in range(15)]
    silence_chunks = [make_audio_chunk(960) for _ in range(35)]
    wf.microphone_client.get_audio.return_value = async_iter(
        speech_chunks + silence_chunks
    )
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    await collect_stream(stream)

    wf._maybe_push_miss.assert_awaited_once()
    args, _ = wf._maybe_push_miss.call_args
    pcm_bytes, max_score = args
    assert isinstance(pcm_bytes, bytes)
    assert len(pcm_bytes) > 0
    assert max_score == 0.3  # peak of prediction_buffer


@pytest.mark.asyncio
async def test_oww_resets_model_after_segment():
    """Test OWW model is reset after each speech segment."""
    wf = make_oww_filter(threshold=0.5)

    wf.vad.is_speech.side_effect = [True] * 15 + [False] * 35
    wf.oww_model.predict.return_value = {"okay_gambit": 0.1}
    wf.oww_model.prediction_buffer = {"okay_gambit": [0.1]}

    speech_chunks = [make_audio_chunk(960) for _ in range(15)]
    silence_chunks = [make_audio_chunk(960) for _ in range(35)]

    wf.microphone_client.get_audio.return_value = async_iter(
        speech_chunks + silence_chunks
    )
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    await collect_stream(stream)

    wf.oww_model.reset.assert_called()


def make_vad_filter(detection_engine="vosk"):
    """Minimal WakeWordFilter mock for _process_vad_frame tests."""
    wf = Mock()
    wf.detection_engine = detection_engine
    wf.logger = Mock()
    return wf


def make_config(max_silence_frames=10, min_speech_frames=3):
    return _SegmentThresholds(
        max_silence_frames=max_silence_frames,
        min_speech_frames=min_speech_frames,
    )


SILENT_FRAME = b"\x00" * FRAME_SIZE_BYTES
SPEECH_FRAME = b"\x01" * FRAME_SIZE_BYTES


class TestProcessVadFrame:
    """Unit tests for WakeWordFilter._process_vad_frame state machine."""

    # --- IDLE state ---

    def test_idle_silence_stays_idle(self):
        wf = make_vad_filter()
        wf.vad.is_speech.return_value = False
        seg = _SpeechSegment()
        chunk = Mock()

        complete, new_state = WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.IDLE, SILENT_FRAME, chunk, make_config()
        )

        assert new_state == _SpeechState.IDLE
        assert complete is False
        assert seg.speech_frames == 0
        assert len(seg.speech_buffer) == 0

    def test_idle_speech_transitions_to_active(self):
        wf = make_vad_filter()
        wf.vad.is_speech.return_value = True
        seg = _SpeechSegment()
        chunk = Mock()

        complete, new_state = WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.IDLE, SPEECH_FRAME, chunk, make_config()
        )

        assert new_state == _SpeechState.ACTIVE
        assert complete is False
        assert seg.speech_frames == 1
        assert chunk in seg.speech_chunk_buffer

    def test_idle_speech_buffers_frame(self):
        wf = make_vad_filter()
        wf.vad.is_speech.return_value = True
        seg = _SpeechSegment()
        chunk = Mock()

        WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.IDLE, SPEECH_FRAME, chunk, make_config()
        )

        assert bytes(seg.speech_buffer) == SPEECH_FRAME

    # --- ACTIVE state ---

    def test_active_speech_increments_speech_frames(self):
        wf = make_vad_filter()
        wf.vad.is_speech.return_value = True
        seg = _SpeechSegment()
        seg.speech_frames = 5
        chunk = Mock()

        complete, new_state = WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.ACTIVE, SPEECH_FRAME, chunk, make_config()
        )

        assert new_state == _SpeechState.ACTIVE
        assert seg.speech_frames == 6
        assert complete is False

    def test_active_silence_transitions_to_trailing(self):
        wf = make_vad_filter()
        wf.vad.is_speech.return_value = False
        seg = _SpeechSegment()
        chunk = Mock()

        complete, new_state = WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.ACTIVE, SILENT_FRAME, chunk, make_config()
        )

        assert new_state == _SpeechState.TRAILING
        assert seg.silence_frames == 1
        assert complete is False

    def test_active_buffers_every_frame(self):
        wf = make_vad_filter()
        wf.vad.is_speech.return_value = True
        seg = _SpeechSegment()
        chunk = Mock()

        WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.ACTIVE, SPEECH_FRAME, chunk, make_config()
        )

        assert SPEECH_FRAME in seg.speech_buffer

    def test_active_same_chunk_not_added_twice(self):
        """Two VAD frames from the same audio_chunk -> chunk appended only once."""
        wf = make_vad_filter()
        wf.vad.is_speech.return_value = True
        seg = _SpeechSegment()
        chunk = Mock()
        seg.speech_chunk_buffer.append(chunk)  # already added

        WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.ACTIVE, SPEECH_FRAME, chunk, make_config()
        )

        assert seg.speech_chunk_buffer.count(chunk) == 1

    def test_active_different_chunk_is_added(self):
        wf = make_vad_filter()
        wf.vad.is_speech.return_value = True
        seg = _SpeechSegment()
        first_chunk = Mock()
        seg.speech_chunk_buffer.append(first_chunk)
        new_chunk = Mock()

        WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.ACTIVE, SPEECH_FRAME, new_chunk, make_config()
        )

        assert new_chunk in seg.speech_chunk_buffer

    # --- TRAILING state ---

    def test_trailing_speech_resumes_active(self):
        wf = make_vad_filter()
        wf.vad.is_speech.return_value = True
        seg = _SpeechSegment()
        seg.speech_frames = 5
        seg.silence_frames = 3
        chunk = Mock()

        complete, new_state = WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.TRAILING, SPEECH_FRAME, chunk, make_config()
        )

        assert new_state == _SpeechState.ACTIVE
        assert seg.silence_frames == 0
        assert seg.speech_frames == 6
        assert complete is False

    def test_trailing_silence_increments_silence_frames(self):
        wf = make_vad_filter()
        wf.vad.is_speech.return_value = False
        seg = _SpeechSegment()
        seg.silence_frames = 2
        chunk = Mock()

        complete, new_state = WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.TRAILING, SILENT_FRAME, chunk, make_config()
        )

        assert new_state == _SpeechState.TRAILING
        assert seg.silence_frames == 3
        assert complete is False

    def test_trailing_silence_reaches_max_returns_complete(self):
        wf = make_vad_filter()
        wf.vad.is_speech.return_value = False
        seg = _SpeechSegment()
        seg.silence_frames = 9  # one below max
        chunk = Mock()
        config = make_config(max_silence_frames=10)

        complete, _ = WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.TRAILING, SILENT_FRAME, chunk, config
        )

        assert complete is True

    def test_segment_complete_when_buffer_exceeds_max(self):
        """Buffer overflow triggers segment_complete regardless of state."""
        from src.models.wake_word_filter import MAX_BUFFER_SIZE_BYTES

        wf = make_vad_filter()
        wf.vad.is_speech.return_value = True
        seg = _SpeechSegment()
        seg.speech_buffer.extend(b"\x00" * MAX_BUFFER_SIZE_BYTES)
        chunk = Mock()

        complete, _ = WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.ACTIVE, SPEECH_FRAME, chunk, make_config()
        )

        assert complete is True

    # --- VAD error handling ---

    def test_vad_error_treated_as_silence(self):
        """VAD exception -> is_speech=False, state unchanged from IDLE."""
        wf = make_vad_filter()
        wf.vad.is_speech.side_effect = Exception("VAD exploded")
        seg = _SpeechSegment()
        chunk = Mock()

        complete, new_state = WakeWordFilter._process_vad_frame(
            wf, seg, _SpeechState.IDLE, SILENT_FRAME, chunk, make_config()
        )

        assert new_state == _SpeechState.IDLE
        assert complete is False


@pytest.mark.asyncio
async def test_finalize_segment_oww_false_positive_resets_model():
    """OWW segment below min_speech_frames -> oww_model.reset() called, nothing yielded."""
    wf = make_oww_filter()
    seg = _SpeechSegment()
    seg.speech_frames = 1  # below default min (300ms // 30ms = 10 frames)
    config = _SegmentThresholds(max_silence_frames=30, min_speech_frames=10)

    chunks = []
    async for chunk in WakeWordFilter._finalize_segment(wf, seg, config):
        chunks.append(chunk)

    assert len(chunks) == 0
    wf.oww_model.reset.assert_called_once()


@pytest.mark.asyncio
async def test_finalize_segment_vosk_false_positive_does_not_reset_oww_model():
    """Vosk segment below min_speech_frames -> oww_model.reset() NOT called."""
    wf = make_vad_filter(detection_engine="vosk")
    wf.oww_model = Mock()
    seg = _SpeechSegment()
    seg.speech_frames = 1
    config = _SegmentThresholds(max_silence_frames=30, min_speech_frames=10)

    async for _ in WakeWordFilter._finalize_segment(wf, seg, config):
        pass

    wf.oww_model.reset.assert_not_called()


@pytest.mark.asyncio
async def test_finalize_segment_resets_speech_segment_on_false_positive():
    """_finalize_segment always resets the segment, even on false positive."""
    wf = make_oww_filter()
    seg = _SpeechSegment()
    seg.speech_frames = 1
    seg.speech_buffer.extend(b"\x00" * 100)
    seg.speech_chunk_buffer.append(Mock())
    config = _SegmentThresholds(max_silence_frames=30, min_speech_frames=10)

    async for _ in WakeWordFilter._finalize_segment(wf, seg, config):
        pass

    assert seg.speech_frames == 0
    assert len(seg.speech_buffer) == 0
    assert len(seg.speech_chunk_buffer) == 0


@pytest.mark.asyncio
async def test_stream_ends_mid_speech_runs_detection():
    """Mic stream ends while ACTIVE with enough frames -> _run_detection called."""
    wf = make_oww_filter(threshold=0.5)
    wf.vad.is_speech.return_value = True
    wf.oww_model.predict.return_value = {"okay_gambit": 0.9}

    # Enough speech frames to exceed min_speech_frames (300ms // 30ms = 10)
    speech_chunks = [make_audio_chunk(960) for _ in range(12)]

    wf.microphone_client.get_audio.return_value = async_iter(speech_chunks)
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks = await collect_stream(stream)

    # Wake word was detected, so chunks should be yielded
    assert len(chunks) > 0
    assert chunks[-1].audio.audio_data == b""


@pytest.mark.asyncio
async def test_stream_ends_mid_speech_below_min_frames_discards():
    """Mic stream ends while ACTIVE but below min_speech_frames -> nothing yielded."""
    wf = make_oww_filter(threshold=0.5)
    wf.vad.is_speech.return_value = True
    wf.oww_model.predict.return_value = {"okay_gambit": 0.9}

    # Only 2 frames — below min_speech_frames (10)
    speech_chunks = [make_audio_chunk(960) for _ in range(2)]

    wf.microphone_client.get_audio.return_value = async_iter(speech_chunks)
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks = await collect_stream(stream)

    assert len(chunks) == 0
    wf.oww_model.reset.assert_called()


@pytest.mark.asyncio
async def test_pause_with_buffered_audio_resets_segment():
    """Pausing detection mid-speech discards buffered audio and resets to IDLE."""
    wf = make_oww_filter(threshold=0.5)
    wf.vad.is_speech.return_value = True
    wf.oww_model.predict.return_value = {"okay_gambit": 0.1}

    # Two speech chunks, then detection is paused, then stream ends
    speech_chunks = [make_audio_chunk(960) for _ in range(2)]
    pause_chunk = make_audio_chunk(960)

    call_count = 0

    async def mic_stream_with_pause():
        nonlocal call_count
        for chunk in speech_chunks:
            yield chunk
        # Pause detection before the next chunk
        wf.detection_running = False
        yield pause_chunk

    wf.microphone_client.get_audio.return_value = mic_stream_with_pause()
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks = await collect_stream(stream)

    # Nothing should be yielded — segment was discarded on pause
    assert len(chunks) == 0


# OWW streaming behavior tests
#
# OWW inference runs once oww_audio_buffer >= 2560 bytes (OWW_CHUNK_SIZE).
# With 960-byte VAD frames (== one chunk in these tests), detection fires
# on the 3rd frame when oww_model.predict returns a score above threshold.


@pytest.mark.asyncio
async def test_oww_streaming_flushes_buffered_chunks_on_detection():
    """First detection yields all pre-detection chunks once, in order."""
    wf = make_oww_filter(threshold=0.5)
    wf.vad.is_speech.return_value = True
    wf.oww_model.predict.return_value = {"okay_gambit": 0.9}

    chunks_in = [make_audio_chunk(960) for _ in range(3)]
    wf.microphone_client.get_audio.return_value = async_iter(chunks_in)
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks_out = await collect_stream(stream)

    # Real chunks (drop any trailing sentinel) yielded once each, in order
    real = [c for c in chunks_out if c.audio.audio_data != b""]
    assert real == chunks_in


@pytest.mark.asyncio
async def test_oww_streaming_yields_post_detection_chunks_immediately():
    """After detection, each new chunk is yielded as it arrives (live)."""
    wf = make_oww_filter(threshold=0.5)
    wf.vad.is_speech.return_value = True
    wf.oww_model.predict.return_value = {"okay_gambit": 0.9}

    # 3 pre-detection + 4 post-detection speech chunks
    chunks_in = [make_audio_chunk(960) for _ in range(7)]
    wf.microphone_client.get_audio.return_value = async_iter(chunks_in)
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks_out = await collect_stream(stream)

    real = [c for c in chunks_out if c.audio.audio_data != b""]
    # All 7 input chunks yielded once each, none lost, none duplicated
    assert real == chunks_in


@pytest.mark.asyncio
async def test_oww_streaming_segment_end_emits_sentinel_and_resets():
    """When the segment ends mid-stream, emit an empty sentinel and reset OWW."""
    wf = make_oww_filter(threshold=0.5)
    wf.oww_model.predict.return_value = {"okay_gambit": 0.9}

    # 3 speech frames (trigger detection), then enough silence to close segment
    # silence_duration_ms=900, frame=30ms → 30 silence frames closes the segment
    speech = [make_audio_chunk(960) for _ in range(3)]
    silence = [make_audio_chunk(960) for _ in range(35)]
    wf.vad.is_speech.side_effect = [True] * 3 + [False] * 35

    wf.microphone_client.get_audio.return_value = async_iter(speech + silence)
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks_out = await collect_stream(stream)

    assert chunks_out[-1].audio.audio_data == b""
    wf.oww_model.reset.assert_called()


@pytest.mark.asyncio
async def test_oww_evaluates_each_speech_segment_independently():
    """Each speech segment is evaluated by its own OWW inference. A detection
    during IDLE (silence) clears, so the next utterance gets a fresh inference
    and only yields when its own audio contains the wake word."""
    wf = make_oww_filter(threshold=0.5)

    # OWW returns a positive score during pre-segment silence, then below-threshold
    # for the actual speech segment. Inference runs every 2560 bytes; frames are
    # 960 bytes, so inference fires roughly every 3 frames.
    wf.oww_model.predict.side_effect = [{"okay_gambit": 0.9}] + [
        {"okay_gambit": 0.1}
    ] * 50
    wf.oww_model.prediction_buffer = {"okay_gambit": [0.1]}

    silence_pre = [make_audio_chunk(960) for _ in range(3)]
    speech = [make_audio_chunk(960) for _ in range(12)]
    silence_post = [make_audio_chunk(960) for _ in range(35)]
    wf.vad.is_speech.side_effect = [False] * 3 + [True] * 12 + [False] * 35

    wf.microphone_client.get_audio.return_value = async_iter(
        silence_pre + speech + silence_post
    )
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks_out = await collect_stream(stream)

    # The real speech segment had no wake word, so output is empty — the IDLE-time
    # detection was scoped to silence and did not carry into the speech segment.
    assert chunks_out == []
    wf.oww_model.reset.assert_called()


@pytest.mark.asyncio
async def test_oww_streaming_yields_full_segment_before_sentinel():
    """All streamed chunks — including the one whose frame trips segment_complete —
    must be yielded before the sentinel."""
    wf = make_oww_filter(threshold=0.5)
    wf.oww_model.predict.return_value = {"okay_gambit": 0.9}

    # 3 speech frames trigger detection (OWW inference fires once buffer >= 2560B,
    # i.e. on the 3rd 960B frame). Then 30 silence frames close the segment —
    # the 30th silence chunk is the one that trips segment_complete mid-chunk.
    speech = [make_audio_chunk(960) for _ in range(3)]
    silence = [make_audio_chunk(960) for _ in range(30)]
    chunks_in = speech + silence
    wf.vad.is_speech.side_effect = [True] * 3 + [False] * 30

    wf.microphone_client.get_audio.return_value = async_iter(chunks_in)
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks_out = await collect_stream(stream)

    real = [c for c in chunks_out if c.audio.audio_data != b""]
    sentinels = [c for c in chunks_out if c.audio.audio_data == b""]

    # Every input chunk yielded once, in order — none dropped, none duplicated
    assert real == chunks_in
    assert len(sentinels) == 1
    assert chunks_out[-1].audio.audio_data == b""


@pytest.mark.asyncio
async def test_oww_streaming_pause_mid_stream_emits_sentinel():
    """Pausing detection while streaming emits sentinel and resets OWW."""
    wf = make_oww_filter(threshold=0.5)
    wf.vad.is_speech.return_value = True
    wf.oww_model.predict.return_value = {"okay_gambit": 0.9}

    pre = [make_audio_chunk(960) for _ in range(3)]  # triggers detection
    post = [make_audio_chunk(960) for _ in range(2)]  # streamed live
    pause_chunk = make_audio_chunk(960)

    async def mic_stream():
        for c in pre + post:
            yield c
        wf.detection_running = False
        yield pause_chunk

    wf.microphone_client.get_audio.return_value = mic_stream()
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks_out = await collect_stream(stream)

    real = [c for c in chunks_out if c.audio.audio_data != b""]
    sentinels = [c for c in chunks_out if c.audio.audio_data == b""]
    assert real == pre + post  # paused chunk dropped
    assert len(sentinels) >= 1
    wf.oww_model.reset.assert_called()


@pytest.mark.asyncio
async def test_oww_streaming_stream_end_mid_detection_emits_sentinel():
    """Mic stream ending mid-stream emits sentinel and resets OWW."""
    wf = make_oww_filter(threshold=0.5)
    wf.vad.is_speech.return_value = True
    wf.oww_model.predict.return_value = {"okay_gambit": 0.9}

    # Detection fires on frame 3, then mic ends with stream still open
    chunks_in = [make_audio_chunk(960) for _ in range(4)]
    wf.microphone_client.get_audio.return_value = async_iter(chunks_in)
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks_out = await collect_stream(stream)

    assert chunks_out[-1].audio.audio_data == b""
    wf.oww_model.reset.assert_called()


@pytest.mark.asyncio
async def test_vosk_does_not_stream():
    """Vosk path never enters streaming mode: only _finalize_segment yields chunks."""
    wf = make_oww_filter()  # reuse helper, then flip engine
    wf.detection_engine = "vosk"
    wf.vad.is_speech.side_effect = [True] * 12 + [False] * 35

    # Mock vosk_process_segment to yield one chunk + sentinel like the real path
    yielded_marker = make_audio_chunk(960)

    async def fake_vosk_process_segment(*_args):
        yield yielded_marker
        yield make_audio_chunk(0)

    speech = [make_audio_chunk(960) for _ in range(12)]
    silence = [make_audio_chunk(960) for _ in range(35)]
    wf.microphone_client.get_audio.return_value = async_iter(speech + silence)
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    with patch(
        "src.models.wake_word_filter.vosk_process_segment",
        side_effect=fake_vosk_process_segment,
    ):
        stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
        chunks_out = await collect_stream(stream)

    # Only what vosk_process_segment yielded — not the raw input chunks
    assert yielded_marker in chunks_out
    # None of the streaming-path input chunks leaked through directly
    assert not any(c in chunks_out for c in speech)
    # OWW reset must not have been called — vosk path doesn't touch OWW
    wf.oww_model.reset.assert_not_called()


# conversation_timeout_seconds tests


def test_validate_config_accepts_conversation_timeout(mock_env):
    """validate_config accepts a positive conversation_timeout_seconds."""
    config = Mock()
    config.attributes = Mock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "conversation_timeout_seconds": 10,
    }
    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_accepts_zero_conversation_timeout(mock_env):
    """validate_config accepts 0 (feature disabled)."""
    config = Mock()
    config.attributes = Mock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "conversation_timeout_seconds": 0,
    }
    deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


def test_validate_config_rejects_negative_conversation_timeout(mock_env):
    config = Mock()
    config.attributes = Mock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "conversation_timeout_seconds": -1,
    }
    with pytest.raises(ValueError, match="conversation_timeout_seconds must be non-negative"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_number_conversation_timeout(mock_env):
    config = Mock()
    config.attributes = Mock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "wake_words": ["robot"],
        "conversation_timeout_seconds": "ten",
    }
    with pytest.raises(ValueError, match="conversation_timeout_seconds must be a number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_string_miss_sensor(mock_env):
    """wakeword_miss_sensor must be a string."""
    config = Mock()
    config.attributes = Mock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "detection_engine": "openwakeword",
        "oww_model_path": "/tmp/model.onnx",
        "wakeword_miss_sensor": 42,
        "near_miss_threshold": 0.3,
    }
    with pytest.raises(ValueError, match="wakeword_miss_sensor must be a string"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_out_of_range_near_miss_threshold(mock_env):
    """near_miss_threshold must be between 0.0 and 1.0."""
    config = Mock()
    config.attributes = Mock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "detection_engine": "openwakeword",
        "oww_model_path": "/tmp/model.onnx",
        "near_miss_threshold": 1.5,
    }
    with pytest.raises(ValueError, match="near_miss_threshold must be between"):
        WakeWordFilter.validate_config(config)


def test_validate_config_rejects_non_number_near_miss_threshold(mock_env):
    """near_miss_threshold must be a number."""
    config = Mock()
    config.attributes = Mock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "detection_engine": "openwakeword",
        "oww_model_path": "/tmp/model.onnx",
        "near_miss_threshold": "half",
    }
    with pytest.raises(ValueError, match="near_miss_threshold must be a number"):
        WakeWordFilter.validate_config(config)


def test_validate_config_miss_sensor_requires_near_miss_threshold(mock_env):
    """wakeword_miss_sensor without near_miss_threshold is a silent dead config."""
    config = Mock()
    config.attributes = Mock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "detection_engine": "openwakeword",
        "oww_model_path": "/tmp/model.onnx",
        "wakeword_miss_sensor": "miss-sensor",
    }
    with pytest.raises(ValueError, match="near_miss_threshold is required"):
        WakeWordFilter.validate_config(config)


def test_validate_config_miss_sensor_requires_openwakeword(mock_env):
    """wakeword_miss_sensor only makes sense with openwakeword (Vosk has no score)."""
    config = Mock()
    config.attributes = Mock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "detection_engine": "vosk",
        "wake_words": ["robot"],
        "wakeword_miss_sensor": "miss-sensor",
        "near_miss_threshold": 0.3,
    }
    with pytest.raises(ValueError, match="requires detection_engine='openwakeword'"):
        WakeWordFilter.validate_config(config)


def test_validate_config_near_miss_threshold_must_be_below_oww_threshold(mock_env):
    """near_miss_threshold >= oww_threshold makes the capture band empty."""
    config = Mock()
    config.attributes = Mock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "detection_engine": "openwakeword",
        "oww_model_path": "/tmp/model.onnx",
        "oww_threshold": 0.6,
        "wakeword_miss_sensor": "miss-sensor",
        "near_miss_threshold": 0.7,
    }
    with pytest.raises(ValueError, match="must be less than oww_threshold"):
        WakeWordFilter.validate_config(config)


def test_validate_config_miss_sensor_declared_as_dep(mock_env):
    """When wakeword_miss_sensor is set, it must appear in the dep list."""
    config = Mock()
    config.attributes = Mock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "detection_engine": "openwakeword",
        "oww_model_path": "/tmp/model.onnx",
        "wakeword_miss_sensor": "miss-sensor",
        "near_miss_threshold": 0.3,
    }
    deps, _ = WakeWordFilter.validate_config(config)
    assert "miss-sensor" in deps


def test_new_uses_default_conversation_timeout(mock_env):
    """new() defaults conversation_timeout_seconds to 0 (feature off)."""
    config = Mock()
    mic = AsyncMock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
    }
    instance = WakeWordFilter.new(config, {"mic1": mic})
    assert instance.conversation_timeout_seconds == 0
    assert instance._conversation_window_expires_at == 0.0


def test_new_uses_custom_conversation_timeout(mock_env):
    config = Mock()
    mic = AsyncMock()
    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic1",
        "wake_words": ["robot"],
        "conversation_timeout_seconds": 15,
    }
    instance = WakeWordFilter.new(config, {"mic1": mic})
    assert instance.conversation_timeout_seconds == 15.0


def test_in_conversation_window_true_when_deadline_in_future():
    """_in_conversation_window returns True when monotonic clock < expires_at."""
    wf = make_oww_filter()
    wf.conversation_timeout_seconds = 10.0
    # Far-future deadline
    wf._conversation_window_expires_at = float("inf")
    assert WakeWordFilter._in_conversation_window(wf) is True


def test_in_conversation_window_false_when_disabled():
    """_in_conversation_window returns False when timeout is 0, regardless of deadline."""
    wf = make_oww_filter()
    wf.conversation_timeout_seconds = 0.0
    wf._conversation_window_expires_at = float("inf")
    assert WakeWordFilter._in_conversation_window(wf) is False


def test_in_conversation_window_false_when_deadline_passed():
    wf = make_oww_filter()
    wf.conversation_timeout_seconds = 10.0
    wf._conversation_window_expires_at = 0.0
    assert WakeWordFilter._in_conversation_window(wf) is False


def test_refresh_conversation_window_sets_deadline():
    """_refresh_conversation_window advances the deadline by the timeout."""
    import time as time_module

    wf = make_oww_filter()
    wf.conversation_timeout_seconds = 10.0
    wf._conversation_window_expires_at = 0.0
    before = time_module.monotonic()
    WakeWordFilter._refresh_conversation_window(wf)
    after = time_module.monotonic()
    assert before + 10.0 <= wf._conversation_window_expires_at <= after + 10.0


def test_refresh_conversation_window_noop_when_disabled():
    """_refresh_conversation_window does nothing when timeout is 0."""
    wf = make_oww_filter()
    wf.conversation_timeout_seconds = 0.0
    wf._conversation_window_expires_at = 0.0
    WakeWordFilter._refresh_conversation_window(wf)
    assert wf._conversation_window_expires_at == 0.0


@pytest.mark.asyncio
async def test_oww_conversation_window_streams_speech_without_wake_word():
    """In a conversation window, speech alone (no wake-word inference hit) engages
    streaming because the IDLE→ACTIVE transition sets oww_detected=True."""
    wf = make_oww_filter(threshold=0.5)
    wf.conversation_timeout_seconds = 10.0
    wf._conversation_window_expires_at = float("inf")
    # OWW would return below threshold — but conversation mode bypasses inference.
    wf.oww_model.predict.return_value = {"okay_gambit": 0.1}
    wf.vad.is_speech.side_effect = [True] * 5 + [False] * 35

    speech = [make_audio_chunk(960) for _ in range(5)]
    silence = [make_audio_chunk(960) for _ in range(35)]
    wf.microphone_client.get_audio.return_value = async_iter(speech + silence)
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks_out = await collect_stream(stream)

    real = [c for c in chunks_out if c.audio.audio_data != b""]
    # Conversation mode engaged streaming on the very first speech frame —
    # every speech chunk was yielded without OWW ever firing.
    assert all(c in real for c in speech)
    assert chunks_out[-1].audio.audio_data == b""


@pytest.mark.asyncio
async def test_oww_no_conversation_window_drops_unprefixed_speech():
    """With no conversation window, speech that doesn't trigger OWW detection
    is not yielded."""
    wf = make_oww_filter(threshold=0.5)
    wf.conversation_timeout_seconds = 0.0
    wf._conversation_window_expires_at = 0.0
    wf.oww_model.predict.return_value = {"okay_gambit": 0.1}
    wf.oww_model.prediction_buffer = {"okay_gambit": [0.1]}
    wf.vad.is_speech.side_effect = [True] * 12 + [False] * 35

    speech = [make_audio_chunk(960) for _ in range(12)]
    silence = [make_audio_chunk(960) for _ in range(35)]
    wf.microphone_client.get_audio.return_value = async_iter(speech + silence)
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    chunks_out = await collect_stream(stream)
    assert chunks_out == []


@pytest.mark.asyncio
async def test_vosk_conversation_window_bypasses_transcript_check():
    """In a conversation window, Vosk yields the segment without running
    transcript matching."""
    wf = make_oww_filter()  # reuse helper
    wf.detection_engine = "vosk"
    wf.conversation_timeout_seconds = 10.0
    wf._conversation_window_expires_at = float("inf")

    speech_chunks = [make_audio_chunk(960) for _ in range(3)]
    speech_buffer = bytearray(b"\x00" * 2880)

    async def fake_vosk_process_segment(*_args):
        # Should NOT be called in conversation mode — yield a marker so we can
        # detect if it was incorrectly invoked.
        yield make_audio_chunk(960)
        raise AssertionError("vosk_process_segment should be bypassed in conversation mode")

    with patch(
        "src.models.wake_word_filter.vosk_process_segment",
        side_effect=fake_vosk_process_segment,
    ):
        chunks_out = []
        async for chunk in WakeWordFilter._run_detection(
            wf, speech_chunks, speech_buffer
        ):
            chunks_out.append(chunk)

    real = [c for c in chunks_out if c.audio.audio_data != b""]
    assert real == speech_chunks
    assert chunks_out[-1].audio.audio_data == b""


@pytest.mark.asyncio
async def test_streaming_sentinel_refreshes_conversation_window():
    """Each yielded OWW segment refreshes the conversation window (sliding timer)."""
    wf = make_oww_filter(threshold=0.5)
    wf.conversation_timeout_seconds = 5.0
    wf._conversation_window_expires_at = 0.0  # window currently closed
    wf.oww_model.predict.return_value = {"okay_gambit": 0.9}
    wf.vad.is_speech.side_effect = [True] * 3 + [False] * 35

    speech = [make_audio_chunk(960) for _ in range(3)]
    silence = [make_audio_chunk(960) for _ in range(35)]
    wf.microphone_client.get_audio.return_value = async_iter(speech + silence)
    wf.microphone_client.get_properties.return_value = Mock(
        sample_rate_hz=16000, num_channels=1
    )

    stream = await WakeWordFilter.get_audio(wf, "pcm16", 0, 0)
    await collect_stream(stream)

    import time as time_module
    # After a successful segment, the window deadline should be in the future.
    assert wf._conversation_window_expires_at > time_module.monotonic()
