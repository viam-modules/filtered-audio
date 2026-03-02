import pytest
from unittest.mock import Mock, patch, MagicMock
from src.models.vosk import (
    get_vosk_model,
    setup_vosk,
    DEFAULT_VOSK_MODEL,
    _get_bundled_models_dir,
    _extract_vosk_model,
    _resolve_absolute_path,
    _resolve_model_name,
)


class TestGetBundledModelsDir:
    """Tests for _get_bundled_models_dir()"""

    def test_returns_vosk_models_path_in_development(self):
        """Test returns vosk_models path relative to project root in development"""
        with patch("src.models.vosk.sys") as mock_sys:
            mock_sys.frozen = False

            result = _get_bundled_models_dir()

            assert result.endswith("vosk_models")

    def test_uses_meipass_when_frozen(self):
        """Test uses sys._MEIPASS when running in PyInstaller bundle"""
        with (
            patch("src.models.vosk.sys") as mock_sys,
            patch("src.models.vosk.getattr", return_value=True),
            patch("src.models.vosk.hasattr", return_value=True),
        ):
            mock_sys.frozen = True
            mock_sys._MEIPASS = "/tmp/pyinstaller_bundle"

            result = _get_bundled_models_dir()

            assert "/tmp/pyinstaller_bundle" in result or "vosk_models" in result


class TestGetVoskModel:
    """Tests for get_vosk_model()"""

    def test_calls_resolve_absolute_path_for_absolute_paths(self):
        """Test that absolute paths are delegated to _resolve_absolute_path"""
        logger = Mock()
        model_path = "/home/user/models/vosk-model-en-us-0.22"

        with (
            patch("src.models.vosk.os.path.expanduser", return_value=model_path),
            patch("src.models.vosk.os.path.isabs", return_value=True),
            patch("src.models.vosk._resolve_absolute_path") as mock_resolve,
        ):
            mock_resolve.return_value = model_path
            result = get_vosk_model(model_path, logger)

        mock_resolve.assert_called_once_with(model_path, logger)
        assert result == model_path

    def test_calls_resolve_model_name_for_model_names(self):
        """Test that model names are delegated to _resolve_model_name"""
        logger = Mock()
        model_name = "vosk-model-small-en-us-0.15"
        resolved_path = "/app/vosk_models/vosk-model-small-en-us-0.15"

        with (
            patch("src.models.vosk.os.path.expanduser", return_value=model_name),
            patch("src.models.vosk.os.path.isabs", return_value=False),
            patch("src.models.vosk._resolve_model_name") as mock_resolve,
        ):
            mock_resolve.return_value = resolved_path
            result = get_vosk_model(model_name, logger)

        mock_resolve.assert_called_once_with(model_name, logger)
        assert result == resolved_path


class TestResolveAbsolutePath:
    """Tests for _resolve_absolute_path()"""

    def test_uses_absolute_path_directly(self):
        """Test that absolute directory paths are used directly"""
        logger = Mock()
        model_path = "/home/user/models/vosk-model-en-us-0.22"

        with patch("src.models.vosk.os.path.exists", return_value=True):
            result = _resolve_absolute_path(model_path, logger)

        assert result == model_path
        logger.info.assert_any_call(f"Using vosk model at path: {model_path}")

    def test_raises_error_for_nonexistent_absolute_path(self):
        """Test that RuntimeError is raised when absolute path doesn't exist"""
        logger = Mock()
        model_path = "/nonexistent/path/vosk-model"

        with (
            patch("src.models.vosk.os.path.exists", return_value=False),
            pytest.raises(RuntimeError, match="does not exist"),
        ):
            _resolve_absolute_path(model_path, logger)

    def test_extracts_zip_file_path(self):
        """Test that zip file paths trigger extraction"""
        logger = Mock()
        zip_path = "/home/user/downloads/vosk-model-en-us-0.22.zip"
        extracted_path = "/tmp/module-data/vosk-model-en-us-0.22"

        with (
            patch("src.models.vosk.os.path.exists", return_value=True),
            patch("src.models.vosk._extract_vosk_model") as mock_extract,
        ):
            mock_extract.return_value = extracted_path
            result = _resolve_absolute_path(zip_path, logger)

        mock_extract.assert_called_once_with(zip_path, logger)
        assert result == extracted_path


class TestResolveModelName:
    """Tests for _resolve_model_name()"""

    def test_checks_bundled_models_first(self):
        """Test that bundled models are checked before cached or download"""
        logger = Mock()
        model_name = "vosk-model-small-en-us-0.15"
        bundled_path = "/app/vosk_models/vosk-model-small-en-us-0.15"

        with (
            patch("src.models.vosk._get_bundled_models_dir") as mock_bundled,
            patch("src.models.vosk.os.path.join", return_value=bundled_path),
            patch("src.models.vosk.os.path.exists", return_value=True),
        ):
            mock_bundled.return_value = "/app/vosk_models"
            _resolve_model_name(model_name, logger)

        logger.info.assert_any_call("Found bundled vosk model")

    def test_checks_cached_models_when_bundled_not_found(self):
        """Test that cached models in VIAM_MODULE_DATA are checked"""
        logger = Mock()
        model_name = "vosk-model-small-en-us-0.15"

        def exists_side_effect(path):
            # Bundled doesn't exist, cached does
            return "module-data" in path

        with (
            patch("src.models.vosk._get_bundled_models_dir") as mock_bundled,
            patch("src.models.vosk.os.getenv", return_value="/tmp/module-data"),
            patch("src.models.vosk.os.path.exists", side_effect=exists_side_effect),
            patch("src.models.vosk.os.path.join", side_effect=lambda *a: "/".join(a)),
        ):
            mock_bundled.return_value = "/app/vosk_models"
            _resolve_model_name(model_name, logger)

        assert "Found cached model" in str(logger.info.call_args_list)

    def test_downloads_when_not_found_locally(self):
        """Test that download is triggered when model not found locally"""
        logger = Mock()
        model_name = "vosk-model-small-en-us-0.15"
        downloaded_path = "/tmp/module-data/vosk-model-small-en-us-0.15"

        with (
            patch("src.models.vosk._get_bundled_models_dir") as mock_bundled,
            patch("src.models.vosk.os.getenv", return_value="/tmp/module-data"),
            patch("src.models.vosk.os.path.exists", return_value=False),
            patch("src.models.vosk._download_vosk_model") as mock_download,
        ):
            mock_bundled.return_value = "/app/vosk_models"
            mock_download.return_value = downloaded_path
            result = _resolve_model_name(model_name, logger)

        mock_download.assert_called_once_with(model_name, logger)
        assert result == downloaded_path


class TestExtractVoskModel:
    """Tests for _extract_vosk_model()"""

    def test_extracts_zip_to_module_data(self):
        """Test that zip is extracted to VIAM_MODULE_DATA directory"""
        logger = Mock()
        zip_path = "/downloads/vosk-model-en-us-0.22.zip"

        mock_zipfile = MagicMock()

        with (
            patch("src.models.vosk.os.getenv", return_value="/tmp/module-data"),
            patch(
                "src.models.vosk.os.path.basename",
                return_value="vosk-model-en-us-0.22.zip",
            ),
            patch("src.models.vosk.os.path.exists", return_value=False),
            patch(
                "src.models.vosk.os.path.join",
                return_value="/tmp/module-data/vosk-model-en-us-0.22",
            ),
            patch("src.models.vosk.zipfile.ZipFile", return_value=mock_zipfile),
        ):
            _extract_vosk_model(zip_path, logger)

        mock_zipfile.__enter__.return_value.extractall.assert_called_once_with(
            "/tmp/module-data"
        )
        logger.info.assert_any_call(f"Extracting model from {zip_path}...")

    def test_raises_error_on_extraction_failure(self):
        """Test that RuntimeError is raised when extraction fails"""
        logger = Mock()
        zip_path = "/downloads/vosk-model-en-us-0.22.zip"

        with (
            patch("src.models.vosk.os.getenv", return_value="/tmp/module-data"),
            patch(
                "src.models.vosk.os.path.basename",
                return_value="vosk-model-en-us-0.22.zip",
            ),
            patch("src.models.vosk.os.path.exists", return_value=False),
            patch(
                "src.models.vosk.os.path.join",
                return_value="/tmp/module-data/vosk-model-en-us-0.22",
            ),
            patch(
                "src.models.vosk.zipfile.ZipFile", side_effect=Exception("Zip error")
            ),
            pytest.raises(RuntimeError, match="Failed to extract"),
        ):
            _extract_vosk_model(zip_path, logger)


class TestSetupVosk:
    """Tests for setup_vosk()"""

    def _make_instance(self, wake_words=None):
        instance = Mock()
        instance.logger = Mock()
        instance.wake_words = wake_words if wake_words is not None else ["robot"]
        return instance

    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_loads_model(self, mock_get, mock_vosk_model, mock_recognizer):
        """get_vosk_model called, VoskModel created with returned path."""
        instance = self._make_instance()
        setup_vosk(instance)

        mock_get.assert_called_once_with(DEFAULT_VOSK_MODEL, instance.logger)
        mock_vosk_model.assert_called_once_with("/tmp/vosk-model")
        assert instance.vosk_model == mock_vosk_model.return_value

    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_default_vosk_model_name(
        self, mock_get, mock_vosk_model, mock_recognizer
    ):
        """No vosk_model provided -> uses DEFAULT_VOSK_MODEL."""
        instance = self._make_instance()
        setup_vosk(instance)

        mock_get.assert_called_once_with(DEFAULT_VOSK_MODEL, instance.logger)

    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_custom_vosk_model_name(
        self, mock_get, mock_vosk_model, mock_recognizer
    ):
        """Custom vosk_model passed to get_vosk_model."""
        instance = self._make_instance()
        setup_vosk(instance, vosk_model="vosk-model-en-us-0.22")

        mock_get.assert_called_once_with("vosk-model-en-us-0.22", instance.logger)

    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_creates_grammar_recognizer(
        self, mock_get, mock_vosk_model, mock_recognizer
    ):
        """use_grammar=True + wake_words -> KaldiRecognizer called with grammar JSON."""
        instance = self._make_instance(wake_words=["robot", "computer"])
        setup_vosk(instance, use_grammar=True)

        args = mock_recognizer.call_args[0]
        assert args[0] == mock_vosk_model.return_value
        assert args[1] == 16000
        # Third arg should be grammar JSON containing the wake words
        import json

        grammar = json.loads(args[2])
        assert "robot" in grammar
        assert "computer" in grammar

    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_creates_plain_recognizer(
        self, mock_get, mock_vosk_model, mock_recognizer
    ):
        """use_grammar=False -> KaldiRecognizer called without grammar."""
        instance = self._make_instance()
        setup_vosk(instance, use_grammar=False)

        args = mock_recognizer.call_args[0]
        assert len(args) == 2  # No grammar argument
        assert args[0] == mock_vosk_model.return_value
        assert args[1] == 16000

    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_creates_plain_recognizer_no_wake_words(
        self, mock_get, mock_vosk_model, mock_recognizer
    ):
        """use_grammar=True but empty wake_words -> no grammar."""
        instance = self._make_instance(wake_words=[])
        setup_vosk(instance, use_grammar=True)

        args = mock_recognizer.call_args[0]
        assert len(args) == 2  # No grammar argument

    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_enables_word_level_scores(
        self, mock_get, mock_vosk_model, mock_recognizer
    ):
        """recognizer.SetWords(True) called."""
        instance = self._make_instance()
        setup_vosk(instance)

        mock_recognizer.return_value.SetWords.assert_called_once_with(True)

    @patch("src.models.vosk.FuzzyWakeWordMatcher")
    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_enables_fuzzy_matcher(
        self, mock_get, mock_vosk_model, mock_recognizer, mock_fuzzy
    ):
        """fuzzy_threshold=3 -> FuzzyWakeWordMatcher(threshold=3) stored on instance."""
        instance = self._make_instance()
        setup_vosk(instance, fuzzy_threshold=3)

        mock_fuzzy.assert_called_once_with(threshold=3)
        assert instance.fuzzy_matcher == mock_fuzzy.return_value

    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_no_fuzzy_matcher_by_default(
        self, mock_get, mock_vosk_model, mock_recognizer
    ):
        """No fuzzy_threshold -> instance.fuzzy_matcher is None."""
        instance = self._make_instance()
        setup_vosk(instance)

        assert instance.fuzzy_matcher is None

    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_default_use_grammar(
        self, mock_get, mock_vosk_model, mock_recognizer
    ):
        """Default use_grammar is True."""
        instance = self._make_instance()
        setup_vosk(instance)

        assert instance.use_grammar is True

    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_custom_use_grammar_false(
        self, mock_get, mock_vosk_model, mock_recognizer
    ):
        """use_grammar=False when provided."""
        instance = self._make_instance()
        setup_vosk(instance, use_grammar=False)

        assert instance.use_grammar is False

    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_default_grammar_confidence(
        self, mock_get, mock_vosk_model, mock_recognizer
    ):
        """Default grammar_confidence is 0.7."""
        instance = self._make_instance()
        setup_vosk(instance)

        assert instance.grammar_confidence == 0.7

    @patch("src.models.vosk.KaldiRecognizer")
    @patch("src.models.vosk.VoskModel")
    @patch("src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model")
    def test_setup_vosk_custom_grammar_confidence(
        self, mock_get, mock_vosk_model, mock_recognizer
    ):
        """Custom grammar_confidence when provided."""
        instance = self._make_instance()
        setup_vosk(instance, grammar_confidence=0.85)

        assert instance.grammar_confidence == 0.85
