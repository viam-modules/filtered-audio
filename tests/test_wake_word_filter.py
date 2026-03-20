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


def test_validate_config_rejects_tflite_on_non_linux(mock_env):
    """Test validate_config raises error when oww_model_path is .tflite on non-Linux"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "detection_engine": "openwakeword",
        "oww_model_path": "/path/to/model.tflite",
    }

    with patch("src.models.wake_word_filter.sys.platform", "darwin"):
        with pytest.raises(ValueError, match="tflite models are only supported on Linux"):
            WakeWordFilter.validate_config(config)


def test_validate_config_rejects_tflite_url_on_non_linux(mock_env):
    """Test validate_config raises error when oww_model_path is a .tflite URL on non-Linux"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "detection_engine": "openwakeword",
        "oww_model_path": "https://example.com/model.tflite",
    }

    with patch("src.models.wake_word_filter.sys.platform", "darwin"):
        with pytest.raises(ValueError, match="tflite models are only supported on Linux"):
            WakeWordFilter.validate_config(config)


def test_validate_config_accepts_tflite_on_linux(mock_env):
    """Test validate_config accepts .tflite model on Linux"""
    config = Mock()
    config.attributes = Mock()

    mock_env["struct_to_dict"].return_value = {
        "source_microphone": "mic",
        "detection_engine": "openwakeword",
        "oww_model_path": "/path/to/model.tflite",
    }

    with patch("src.models.wake_word_filter.sys.platform", "linux"):
        deps, errors = WakeWordFilter.validate_config(config)
    assert deps == ["mic"]
    assert not errors


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
