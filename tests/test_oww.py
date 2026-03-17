import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.models.oww import setup_oww, oww_check_for_wake_word, OWW_CHUNK_SIZE


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

    def _call_setup_oww(
        self, model_path, platform="linux", oww_model_cls=None, threshold=0.5
    ):
        """Run setup_oww with mocked openwakeword and sys.platform. Returns (instance, oww_model_cls)."""
        instance = self._make_instance()
        mock_oww = self._mock_openwakeword()
        if oww_model_cls is None:
            oww_model_cls = Mock()
        with (
            patch.dict(
                "sys.modules",
                {
                    "openwakeword": mock_oww,
                    "openwakeword.model": Mock(Model=oww_model_cls),
                    "openwakeword.utils": mock_oww.utils,
                },
            ),
            patch("src.models.oww.os.path.exists", return_value=True),
            patch("src.models.oww.os.makedirs"),
            patch("src.models.oww.sys") as mock_sys,
        ):
            mock_sys.platform = platform
            setup_oww(instance, oww_model_path=model_path, oww_threshold=threshold)
        return instance, oww_model_cls

    def test_setup_oww_loads_local_model(self):
        """Local .onnx path exists -> OWWModel created with correct args."""
        mock_cls = Mock()
        instance, oww_model_cls = self._call_setup_oww(
            "/tmp/my_wakeword.onnx", oww_model_cls=mock_cls
        )
        assert instance.oww_model == mock_cls.return_value
        mock_cls.assert_called_once()
        assert mock_cls.call_args[1]["wakeword_models"] == ["/tmp/my_wakeword.onnx"]
        assert mock_cls.call_args[1]["inference_framework"] == "onnx"

    def test_setup_oww_derives_model_name(self):
        """Path /tmp/my_wakeword.onnx -> oww_model_name == 'my_wakeword'."""
        instance, _ = self._call_setup_oww("/tmp/my_wakeword.onnx")
        assert instance.oww_model_name == "my_wakeword"

    def test_setup_oww_default_threshold(self):
        """oww_threshold=0.5 -> instance.oww_threshold == 0.5."""
        instance, _ = self._call_setup_oww("/tmp/model.onnx", threshold=0.5)
        assert instance.oww_threshold == 0.5

    def test_setup_oww_custom_threshold(self):
        """oww_threshold=0.8 -> instance.oww_threshold == 0.8."""
        instance, _ = self._call_setup_oww("/tmp/model.onnx", threshold=0.8)
        assert instance.oww_threshold == 0.8

    @patch("src.models.oww.download_file")
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_downloads_url_model(self, mock_makedirs, mock_download):
        """HTTP URL triggers download_file(), sets oww_model_path to cached path."""

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

        assert mock_oww.utils.download_file.call_count == 4  # tflite + onnx per model

    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_skips_existing_preprocessing_models(self, mock_makedirs):
        """Preprocessing model exists -> openwakeword.utils.download_file() NOT called."""

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

    def test_setup_oww_speex_linux_only(self):
        """enable_speex_noise_suppression matches sys.platform == 'linux'."""
        mock_cls = Mock()
        self._call_setup_oww(
            "/tmp/model.onnx", platform="linux", oww_model_cls=mock_cls
        )
        assert mock_cls.call_args[1]["enable_speex_noise_suppression"] is True

        mock_cls.reset_mock()
        self._call_setup_oww(
            "/tmp/model.onnx", platform="darwin", oww_model_cls=mock_cls
        )
        assert mock_cls.call_args[1]["enable_speex_noise_suppression"] is False

    @patch("src.models.oww.os.path.exists", return_value=True)
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_raises_for_invalid_extension(self, mock_makedirs, mock_exists):
        """Non-.onnx/.tflite extension raises ValueError."""
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
            pytest.raises(ValueError, match="file extension must be .onnx or .tflite"),
        ):
            setup_oww(instance, oww_model_path="/tmp/model.pt", oww_threshold=0.5)

    @patch("src.models.oww.os.path.exists", return_value=True)
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_tflite_raises_on_non_linux(self, mock_makedirs, mock_exists):
        """tflite model on non-Linux raises ValueError."""
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
            patch("src.models.oww.sys") as mock_sys,
            pytest.raises(ValueError, match="tflite models are only supported on Linux"),
        ):
            mock_sys.platform = "darwin"
            setup_oww(instance, oww_model_path="/tmp/model.tflite", oww_threshold=0.5)

    @patch("src.models.oww.os.path.exists", return_value=True)
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_tflite_uses_tflite_framework_on_linux(self, mock_makedirs, mock_exists):
        """tflite model on Linux uses inference_framework='tflite'."""
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
            setup_oww(instance, oww_model_path="/tmp/model.tflite", oww_threshold=0.5)

        call_kwargs = mock_oww_model_cls.call_args[1]
        assert call_kwargs["inference_framework"] == "tflite"

    @patch("src.models.oww.os.path.exists", return_value=True)
    @patch("src.models.oww.os.makedirs")
    def test_setup_oww_onnx_uses_onnx_framework(self, mock_makedirs, mock_exists):
        """onnx model uses inference_framework='onnx'."""
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
        assert call_kwargs["inference_framework"] == "onnx"


class TestOwwCheckForWakeWord:
    """Tests for oww_check_for_wake_word()"""

    def _make_instance(self, threshold=0.5, model_name="okay_gambit"):
        instance = Mock()
        instance.logger = Mock()
        instance.oww_threshold = threshold
        instance.oww_model_name = model_name
        return instance

    def _make_buffer(self, audio_bytes=b""):
        buf = bytearray()
        buf.extend(audio_bytes)
        return buf

    def test_returns_false_when_buffer_too_small(self):
        """Buffer smaller than OWW_CHUNK_SIZE -> returns False, no inference."""
        instance = self._make_instance()
        buf = self._make_buffer(b"\x00" * (OWW_CHUNK_SIZE - 1))

        result = oww_check_for_wake_word(instance, buf)

        assert result is False
        instance.oww_model.predict.assert_not_called()

    def test_returns_false_when_buffer_empty(self):
        """Empty buffer -> returns False immediately."""
        instance = self._make_instance()
        buf = self._make_buffer()

        result = oww_check_for_wake_word(instance, buf)

        assert result is False
        instance.oww_model.predict.assert_not_called()

    def test_returns_true_when_score_above_threshold(self):
        """Score >= threshold -> returns True."""
        instance = self._make_instance(threshold=0.5)
        instance.oww_model.predict.return_value = {"okay_gambit": 0.9}
        buf = self._make_buffer(b"\x00" * OWW_CHUNK_SIZE)

        result = oww_check_for_wake_word(instance, buf)

        assert result is True

    def test_returns_false_when_score_below_threshold(self):
        """Score < threshold -> returns False."""
        instance = self._make_instance(threshold=0.5)
        instance.oww_model.predict.return_value = {"okay_gambit": 0.1}
        buf = self._make_buffer(b"\x00" * OWW_CHUNK_SIZE)

        result = oww_check_for_wake_word(instance, buf)

        assert result is False

    def test_returns_false_when_model_name_not_in_prediction(self):
        """Model name missing from prediction dict -> score defaults to 0.0."""
        instance = self._make_instance(threshold=0.5)
        instance.oww_model.predict.return_value = {"other_model": 0.9}
        buf = self._make_buffer(b"\x00" * OWW_CHUNK_SIZE)

        result = oww_check_for_wake_word(instance, buf)

        assert result is False

    def test_score_exactly_at_threshold_triggers_detection(self):
        """Score == threshold -> returns True."""
        instance = self._make_instance(threshold=0.5)
        instance.oww_model.predict.return_value = {"okay_gambit": 0.5}
        buf = self._make_buffer(b"\x00" * OWW_CHUNK_SIZE)

        result = oww_check_for_wake_word(instance, buf)

        assert result is True

    def test_drains_buffer_in_chunks(self):
        """Buffer with 2x OWW_CHUNK_SIZE -> predict called twice."""
        instance = self._make_instance(threshold=0.5)
        instance.oww_model.predict.return_value = {"okay_gambit": 0.1}
        buf = self._make_buffer(b"\x00" * (OWW_CHUNK_SIZE * 2))

        oww_check_for_wake_word(instance, buf)

        assert instance.oww_model.predict.call_count == 2

    def test_consumed_bytes_removed_from_buffer(self):
        """Processed bytes are deleted from the buffer."""
        instance = self._make_instance(threshold=0.5)
        instance.oww_model.predict.return_value = {"okay_gambit": 0.1}
        buf = self._make_buffer(b"\x00" * OWW_CHUNK_SIZE + b"\x01" * 100)

        oww_check_for_wake_word(instance, buf)

        assert len(buf) == 100

    def test_stops_early_on_detection(self):
        """Returns True on first detection without draining remaining buffer."""
        instance = self._make_instance(threshold=0.5)
        instance.oww_model.predict.return_value = {"okay_gambit": 0.9}
        buf = self._make_buffer(b"\x00" * (OWW_CHUNK_SIZE * 3))

        result = oww_check_for_wake_word(instance, buf)

        assert result is True
        assert instance.oww_model.predict.call_count == 1

    def test_passes_int16_array_to_predict(self):
        """Audio bytes are converted to int16 numpy array before predict()."""
        instance = self._make_instance(threshold=0.5)
        instance.oww_model.predict.return_value = {"okay_gambit": 0.1}
        buf = self._make_buffer(b"\x00" * OWW_CHUNK_SIZE)

        oww_check_for_wake_word(instance, buf)

        call_args = instance.oww_model.predict.call_args[0]
        audio_array = call_args[0]
        assert isinstance(audio_array, np.ndarray)
        assert audio_array.dtype == np.int16
        assert len(audio_array) == OWW_CHUNK_SIZE // 2
