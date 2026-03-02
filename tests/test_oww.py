import pytest
from unittest.mock import Mock, patch, MagicMock


class TestSetupOww:
    """Tests for setup_oww()"""

    def _make_instance(self):
        instance = Mock()
        instance.logger = Mock()
        return instance

    def _mock_openwakeword(self):
        """Create mock openwakeword module with FEATURE_MODELS and VAD_MODELS."""
        mock_oww = MagicMock()
        mock_oww.__file__ = "/fake/openwakeword/__init__.py"
        mock_oww.FEATURE_MODELS = {
            "melspectrogram": {
                "download_url": "https://example.com/melspectrogram.tflite"
            }
        }
        mock_oww.VAD_MODELS = {
            "silero_vad": {"download_url": "https://example.com/silero_vad.tflite"}
        }
        mock_oww.utils = Mock()
        return mock_oww

    @patch("src.models.oww.os.path.exists", return_value=True)
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_loads_local_model(self, mock_makedirs, mock_exists):
        """Local .onnx path exists -> OWWModel created with correct args."""
        from src.models.oww import setup_oww

        instance = self._make_instance()
        mock_oww = self._mock_openwakeword()
        mock_oww_model_cls = Mock()

        with patch.dict(
            "sys.modules",
            {
                "openwakeword": mock_oww,
                "openwakeword.model": Mock(Model=mock_oww_model_cls),
                "openwakeword.utils": mock_oww.utils,
            },
        ):
            setup_oww(
                instance, oww_model_path="/tmp/my_wakeword.onnx", oww_threshold=0.5
            )

        assert instance.oww_model == mock_oww_model_cls.return_value
        mock_oww_model_cls.assert_called_once()
        call_kwargs = mock_oww_model_cls.call_args
        assert call_kwargs[1]["wakeword_models"] == ["/tmp/my_wakeword.onnx"]
        assert call_kwargs[1]["inference_framework"] == "onnx"

    @patch("src.models.oww.os.path.exists", return_value=True)
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_derives_model_name(self, mock_makedirs, mock_exists):
        """Path /tmp/my_wakeword.onnx -> oww_model_name == 'my_wakeword'."""
        from src.models.oww import setup_oww

        instance = self._make_instance()
        mock_oww = self._mock_openwakeword()

        with patch.dict(
            "sys.modules",
            {
                "openwakeword": mock_oww,
                "openwakeword.model": Mock(Model=Mock()),
                "openwakeword.utils": mock_oww.utils,
            },
        ):
            setup_oww(
                instance, oww_model_path="/tmp/my_wakeword.onnx", oww_threshold=0.5
            )

        assert instance.oww_model_name == "my_wakeword"

    @patch("src.models.oww.os.path.exists", return_value=True)
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_default_threshold(self, mock_makedirs, mock_exists):
        """oww_threshold=0.5 (the default) -> instance.oww_threshold == 0.5."""
        from src.models.oww import setup_oww

        instance = self._make_instance()
        mock_oww = self._mock_openwakeword()

        with patch.dict(
            "sys.modules",
            {
                "openwakeword": mock_oww,
                "openwakeword.model": Mock(Model=Mock()),
                "openwakeword.utils": mock_oww.utils,
            },
        ):
            setup_oww(instance, oww_model_path="/tmp/model.onnx", oww_threshold=0.5)

        assert instance.oww_threshold == 0.5

    @patch("src.models.oww.os.path.exists", return_value=True)
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_custom_threshold(self, mock_makedirs, mock_exists):
        """oww_threshold=0.8 -> instance.oww_threshold == 0.8."""
        from src.models.oww import setup_oww

        instance = self._make_instance()
        mock_oww = self._mock_openwakeword()

        with patch.dict(
            "sys.modules",
            {
                "openwakeword": mock_oww,
                "openwakeword.model": Mock(Model=Mock()),
                "openwakeword.utils": mock_oww.utils,
            },
        ):
            setup_oww(
                instance,
                oww_model_path="/tmp/model.onnx",
                oww_threshold=0.8,
            )

        assert instance.oww_threshold == 0.8

    @patch("src.models.oww.download_file")
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_downloads_url_model(self, mock_makedirs, mock_download):
        """HTTP URL triggers download_file(), sets oww_model_path to cached path."""
        from src.models.oww import setup_oww

        instance = self._make_instance()
        mock_oww = self._mock_openwakeword()

        cached_checked = []

        def exists_side_effect(path):
            # preprocessing models exist
            if path.endswith(".onnx") and "resources" in path:
                return True
            # First check on cached path: not cached yet; second check: exists after download
            if "my_model.onnx" in path:
                cached_checked.append(path)
                return len(cached_checked) > 1
            return True

        with (
            patch.dict(
                "sys.modules",
                {
                    "openwakeword": mock_oww,
                    "openwakeword.model": Mock(Model=Mock()),
                    "openwakeword.utils": mock_oww.utils,
                },
            ),
            patch("src.models.oww.os.path.exists", side_effect=exists_side_effect),
            patch("src.models.oww.os.getenv", return_value="/tmp/cache"),
            patch("src.models.oww.urllib.request.urlopen"),
        ):
            setup_oww(
                instance,
                oww_model_path="https://example.com/my_model.onnx",
                oww_threshold=0.5,
            )

        mock_download.assert_called_once_with(
            "https://example.com/my_model.onnx",
            "/tmp/cache/my_model.onnx",
            instance.logger,
        )

    @patch("src.models.oww.download_file")
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_uses_cached_url_model(self, mock_makedirs, mock_download):
        """Cached file exists -> download_file NOT called."""
        from src.models.oww import setup_oww

        instance = self._make_instance()
        mock_oww = self._mock_openwakeword()

        with (
            patch.dict(
                "sys.modules",
                {
                    "openwakeword": mock_oww,
                    "openwakeword.model": Mock(Model=Mock()),
                    "openwakeword.utils": mock_oww.utils,
                },
            ),
            patch("src.models.oww.os.path.exists", return_value=True),
            patch("src.models.oww.os.getenv", return_value="/tmp/cache"),
        ):
            setup_oww(
                instance,
                oww_model_path="https://example.com/my_model.onnx",
                oww_threshold=0.5,
            )

        mock_download.assert_not_called()

    @patch("src.models.oww.os.path.exists", return_value=False)
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_raises_for_missing_local_path(self, mock_makedirs, mock_exists):
        """Nonexistent local path -> ValueError."""
        from src.models.oww import setup_oww

        instance = self._make_instance()
        mock_oww = self._mock_openwakeword()

        with (
            patch.dict(
                "sys.modules",
                {
                    "openwakeword": mock_oww,
                    "openwakeword.model": Mock(Model=Mock()),
                    "openwakeword.utils": mock_oww.utils,
                },
            ),
            pytest.raises(ValueError, match="oww_model_path does not exist"),
        ):
            setup_oww(
                instance, oww_model_path="/nonexistent/model.onnx", oww_threshold=0.5
            )

    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_downloads_preprocessing_models(self, mock_makedirs):
        """Missing preprocessing model triggers openwakeword.utils.download_file()."""
        from src.models.oww import setup_oww

        instance = self._make_instance()
        mock_oww = self._mock_openwakeword()

        def exists_side_effect(path):
            # Preprocessing models don't exist, but the user model does
            if "resources/models" in path:
                return False
            return True

        with (
            patch.dict(
                "sys.modules",
                {
                    "openwakeword": mock_oww,
                    "openwakeword.model": Mock(Model=Mock()),
                    "openwakeword.utils": mock_oww.utils,
                },
            ),
            patch("src.models.oww.os.path.exists", side_effect=exists_side_effect),
        ):
            setup_oww(instance, oww_model_path="/tmp/model.onnx", oww_threshold=0.5)

        assert mock_oww.utils.download_file.call_count == 2  # one per model

    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_skips_existing_preprocessing_models(self, mock_makedirs):
        """Preprocessing model exists -> openwakeword.utils.download_file() NOT called."""
        from src.models.oww import setup_oww

        instance = self._make_instance()
        mock_oww = self._mock_openwakeword()

        with (
            patch.dict(
                "sys.modules",
                {
                    "openwakeword": mock_oww,
                    "openwakeword.model": Mock(Model=Mock()),
                    "openwakeword.utils": mock_oww.utils,
                },
            ),
            patch("src.models.oww.os.path.exists", return_value=True),
        ):
            setup_oww(instance, oww_model_path="/tmp/model.onnx", oww_threshold=0.5)

        mock_oww.utils.download_file.assert_not_called()

    @patch("src.models.oww.os.path.exists", return_value=True)
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_speex_linux_only(self, mock_makedirs, mock_exists):
        """enable_speex_noise_suppression matches sys.platform == 'linux'."""
        from src.models.oww import setup_oww

        instance = self._make_instance()
        mock_oww = self._mock_openwakeword()
        mock_oww_model_cls = Mock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "openwakeword": mock_oww,
                    "openwakeword.model": Mock(Model=mock_oww_model_cls),
                    "openwakeword.utils": mock_oww.utils,
                },
            ),
            patch("src.models.oww.sys") as mock_sys,
        ):
            mock_sys.platform = "linux"
            setup_oww(instance, oww_model_path="/tmp/model.onnx", oww_threshold=0.5)

            call_kwargs = mock_oww_model_cls.call_args[1]
            assert call_kwargs["enable_speex_noise_suppression"] is True

        mock_oww_model_cls.reset_mock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "openwakeword": mock_oww,
                    "openwakeword.model": Mock(Model=mock_oww_model_cls),
                    "openwakeword.utils": mock_oww.utils,
                },
            ),
            patch("src.models.oww.sys") as mock_sys,
        ):
            mock_sys.platform = "darwin"
            setup_oww(instance, oww_model_path="/tmp/model.onnx", oww_threshold=0.5)

            call_kwargs = mock_oww_model_cls.call_args[1]
            assert call_kwargs["enable_speex_noise_suppression"] is False
