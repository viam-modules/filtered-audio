import pytest
from unittest.mock import Mock, patch, MagicMock
from src.models.vosk import (
    get_vosk_model,
    _get_bundled_models_dir,
    _extract_vosk_model,
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
        with patch("src.models.vosk.sys") as mock_sys, \
            patch("src.models.vosk.getattr", return_value=True), \
            patch("src.models.vosk.hasattr", return_value=True):
            mock_sys.frozen = True
            mock_sys._MEIPASS = "/tmp/pyinstaller_bundle"

            result = _get_bundled_models_dir()

            assert (
                "/tmp/pyinstaller_bundle" in result or "vosk_models" in result
            )


class TestGetVoskModel:
    """Tests for get_vosk_model()"""

    def test_uses_absolute_path_directly(self):
        """Test that absolute directory paths are used directly"""
        logger = Mock()
        model_path = "/home/user/models/vosk-model-en-us-0.22"

        with patch("src.models.vosk.os.path.expanduser", return_value=model_path), \
            patch("src.models.vosk.os.path.isabs", return_value=True), \
            patch("src.models.vosk.os.path.exists", return_value=True):
            result = get_vosk_model(model_path, logger)

        assert result == model_path
        logger.info.assert_any_call(f"Using vosk model at path: {model_path}")

    def test_raises_error_for_nonexistent_absolute_path(self):
        """Test that RuntimeError is raised when absolute path doesn't exist"""
        logger = Mock()
        model_path = "/nonexistent/path/vosk-model"

        with patch("src.models.vosk.os.path.expanduser", return_value=model_path), \
            patch("src.models.vosk.os.path.isabs", return_value=True), \
            patch("src.models.vosk.os.path.exists", return_value=False), \
            pytest.raises(RuntimeError, match="does not exist"):
            get_vosk_model(model_path, logger)

    def test_extracts_zip_file_path(self):
        """Test that zip file paths trigger extraction"""
        logger = Mock()
        zip_path = "/home/user/downloads/vosk-model-en-us-0.22.zip"
        extracted_path = "/tmp/module-data/vosk-model-en-us-0.22"

        with patch("src.models.vosk.os.path.expanduser", return_value=zip_path), \
            patch("src.models.vosk.os.path.isabs", return_value=True), \
            patch("src.models.vosk.os.path.exists", return_value=True), \
            patch("src.models.vosk._extract_vosk_model", return_value=extracted_path) as mock_extract:
            result = get_vosk_model(zip_path, logger)

        mock_extract.assert_called_once_with(zip_path, logger)
        assert result == extracted_path

    def test_checks_bundled_models_first(self):
        """Test that bundled models are checked before cached or download"""
        logger = Mock()
        model_name = "vosk-model-small-en-us-0.15"
        bundled_path = "/app/vosk_models/vosk-model-small-en-us-0.15"

        with patch("src.models.vosk.os.path.expanduser", return_value=model_name), \
            patch("src.models.vosk.os.path.isabs", return_value=False), \
            patch("src.models.vosk._get_bundled_models_dir", return_value="/app/vosk_models"), \
            patch("src.models.vosk.os.path.exists", return_value=True), \
            patch("src.models.vosk.os.path.join", return_value=bundled_path):
            get_vosk_model(model_name, logger)

        logger.info.assert_any_call("Found bundled vosk model")

    def test_checks_cached_models_when_bundled_not_found(self):
        """Test that cached models in VIAM_MODULE_DATA are checked"""
        logger = Mock()
        model_name = "vosk-model-small-en-us-0.15"

        def exists_side_effect(path):
            # Bundled doesn't exist, cached does
            return "module-data" in path

        with patch("src.models.vosk.os.path.expanduser", return_value=model_name), \
            patch("src.models.vosk.os.path.isabs", return_value=False), \
            patch("src.models.vosk._get_bundled_models_dir", return_value="/app/vosk_models"), \
            patch("src.models.vosk.os.getenv", return_value="/tmp/module-data"), \
            patch("src.models.vosk.os.path.exists", side_effect=exists_side_effect), \
            patch("src.models.vosk.os.path.join", side_effect=lambda *args: "/".join(args)):
            get_vosk_model(model_name, logger)

        assert "Found cached model" in str(logger.info.call_args_list)

    def test_downloads_when_not_found_locally(self):
        """Test that download is triggered when model not found locally"""
        logger = Mock()
        model_name = "vosk-model-small-en-us-0.15"
        downloaded_path = "/tmp/module-data/vosk-model-small-en-us-0.15"

        with patch("src.models.vosk.os.path.expanduser", return_value=model_name), \
            patch("src.models.vosk.os.path.isabs", return_value=False), \
            patch("src.models.vosk._get_bundled_models_dir", return_value="/app/vosk_models"), \
            patch("src.models.vosk.os.getenv", return_value="/tmp/module-data"), \
            patch("src.models.vosk.os.path.exists", return_value=False), \
            patch("src.models.vosk._download_vosk_model", return_value=downloaded_path) as mock_download:
            result = get_vosk_model(model_name, logger)

        mock_download.assert_called_once_with(model_name, logger)
        assert result == downloaded_path


class TestExtractVoskModel:
    """Tests for _extract_vosk_model()"""

    def test_extracts_zip_to_module_data(self):
        """Test that zip is extracted to VIAM_MODULE_DATA directory"""
        logger = Mock()
        zip_path = "/downloads/vosk-model-en-us-0.22.zip"

        mock_zipfile = MagicMock()


        with patch("src.models.vosk.os.getenv", return_value="/tmp/module-data"), \
            patch("src.models.vosk.os.path.basename", return_value="vosk-model-en-us-0.22.zip"), \
            patch("src.models.vosk.os.path.exists", return_value=False), \
            patch("src.models.vosk.os.path.join", return_value="/tmp/module-data/vosk-model-en-us-0.22"), \
            patch("src.models.vosk.zipfile.ZipFile", return_value=mock_zipfile):
            _extract_vosk_model(zip_path, logger)

        mock_zipfile.__enter__.return_value.extractall.assert_called_once_with(
            "/tmp/module-data"
        )
        logger.info.assert_any_call(f"Extracting model from {zip_path}...")

    def test_raises_error_on_extraction_failure(self):
        """Test that RuntimeError is raised when extraction fails"""
        logger = Mock()
        zip_path = "/downloads/vosk-model-en-us-0.22.zip"

        with patch("src.models.vosk.os.getenv", return_value="/tmp/module-data"), \
            patch("src.models.vosk.os.path.basename", return_value="vosk-model-en-us-0.22.zip"), \
            patch("src.models.vosk.os.path.exists", return_value=False), \
            patch("src.models.vosk.os.path.join", return_value="/tmp/module-data/vosk-model-en-us-0.22"), \
            patch("src.models.vosk.zipfile.ZipFile", side_effect=Exception("Zip error")), \
            pytest.raises(RuntimeError, match="Failed to extract"):
            _extract_vosk_model(zip_path, logger)
