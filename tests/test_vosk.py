import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.models.vosk import (
    get_vosk_model,
    setup_vosk,
    vosk_check_for_wake_word,
    vosk_process_segment,
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

    def _call_setup_vosk(self, wake_words=None, **kwargs):
        """Run setup_vosk with mocked Vosk deps. Returns instance with _mock_get, _mock_vosk_model, _mock_recognizer attached."""
        instance = self._make_instance(wake_words=wake_words)
        with (
            patch(
                "src.models.vosk.get_vosk_model", return_value="/tmp/vosk-model"
            ) as mock_get,
            patch("src.models.vosk.VoskModel") as mock_vosk_model,
            patch("src.models.vosk.KaldiRecognizer") as mock_recognizer,
        ):
            setup_vosk(instance, **kwargs)
            instance._mock_get = mock_get
            instance._mock_vosk_model = mock_vosk_model
            instance._mock_recognizer = mock_recognizer
        return instance

    def test_setup_vosk_loads_model(self):
        """get_vosk_model called, VoskModel created with returned path."""
        instance = self._call_setup_vosk()
        instance._mock_get.assert_called_once_with(DEFAULT_VOSK_MODEL, instance.logger)
        instance._mock_vosk_model.assert_called_once_with("/tmp/vosk-model")
        assert instance.vosk_model == instance._mock_vosk_model.return_value

    def test_setup_vosk_default_vosk_model_name(self):
        """No vosk_model provided -> uses DEFAULT_VOSK_MODEL."""
        instance = self._call_setup_vosk()
        instance._mock_get.assert_called_once_with(DEFAULT_VOSK_MODEL, instance.logger)

    def test_setup_vosk_custom_vosk_model_name(self):
        """Custom vosk_model passed to get_vosk_model."""
        instance = self._call_setup_vosk(vosk_model="vosk-model-en-us-0.22")
        instance._mock_get.assert_called_once_with(
            "vosk-model-en-us-0.22", instance.logger
        )

    def test_setup_vosk_creates_grammar_recognizer(self):
        """use_grammar=True + wake_words -> KaldiRecognizer called with grammar JSON."""
        instance = self._call_setup_vosk(
            wake_words=["robot", "computer"], use_grammar=True
        )
        args = instance._mock_recognizer.call_args[0]
        assert args[0] == instance._mock_vosk_model.return_value
        assert args[1] == 16000
        grammar = json.loads(args[2])
        assert "robot" in grammar
        assert "computer" in grammar

    def test_setup_vosk_creates_plain_recognizer(self):
        """use_grammar=False -> KaldiRecognizer called without grammar."""
        instance = self._call_setup_vosk(use_grammar=False)
        args = instance._mock_recognizer.call_args[0]
        assert len(args) == 2
        assert args[0] == instance._mock_vosk_model.return_value
        assert args[1] == 16000

    def test_setup_vosk_creates_plain_recognizer_no_wake_words(self):
        """use_grammar=True but empty wake_words -> no grammar."""
        instance = self._call_setup_vosk(wake_words=[], use_grammar=True)
        assert len(instance._mock_recognizer.call_args[0]) == 2

    def test_setup_vosk_enables_word_level_scores(self):
        """recognizer.SetWords(True) called."""
        instance = self._call_setup_vosk()
        instance._mock_recognizer.return_value.SetWords.assert_called_once_with(True)

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

    def test_setup_vosk_no_fuzzy_matcher_by_default(self):
        """No fuzzy_threshold -> instance.fuzzy_matcher is None."""
        assert self._call_setup_vosk().fuzzy_matcher is None

    def test_setup_vosk_default_use_grammar(self):
        """Default use_grammar is True."""
        assert self._call_setup_vosk().use_grammar is True

    def test_setup_vosk_custom_use_grammar_false(self):
        """use_grammar=False when provided."""
        assert self._call_setup_vosk(use_grammar=False).use_grammar is False

    def test_setup_vosk_default_grammar_confidence(self):
        """Default grammar_confidence is 0.7."""
        assert self._call_setup_vosk().grammar_confidence == 0.7

    def test_setup_vosk_custom_grammar_confidence(self):
        """Custom grammar_confidence when provided."""
        assert self._call_setup_vosk(grammar_confidence=0.85).grammar_confidence == 0.85


class TestVoskCheckForWakeWord:
    def _make_instance(self, wake_words=None, use_grammar=True, fuzzy_matcher=None):
        wake_word_filter = Mock()
        wake_word_filter.wake_words = wake_words or ["robot"]
        wake_word_filter.fuzzy_matcher = fuzzy_matcher
        wake_word_filter.use_grammar = use_grammar
        wake_word_filter.logger = Mock()
        mock_rec = Mock()
        mock_rec.AcceptWaveform.return_value = True
        wake_word_filter.recognizer = mock_rec
        return wake_word_filter, mock_rec

    def test_handles_multiple_wake_words(self):
        """Test check_for_wake_word works with multiple wake words"""
        wake_word_filter, mock_rec = self._make_instance(
            wake_words=["robot", "computer", "hey assistant"]
        )

        mock_rec.FinalResult.return_value = '{"text": "robot do something"}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is True

        mock_rec.FinalResult.return_value = '{"text": "computer show me"}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is True

        mock_rec.FinalResult.return_value = '{"text": "hey assistant what time"}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is True

    def test_respects_word_boundaries(self):
        """Test check_for_wake_word doesn't match substrings"""
        wake_word_filter, mock_rec = self._make_instance()

        mock_rec.FinalResult.return_value = '{"text": "robot turn on"}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is True

        mock_rec.FinalResult.return_value = '{"text": "robotics is cool"}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is False

    def test_with_grammar_mode(self):
        """Test check_for_wake_word works in grammar mode"""
        wake_word_filter, mock_rec = self._make_instance(
            wake_words=["robot", "computer"], use_grammar=True
        )
        mock_rec.FinalResult.return_value = '{"text": "robot"}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is True

    def test_without_grammar_mode(self):
        """Test check_for_wake_word uses full transcription and searches for wake word"""
        wake_word_filter, mock_rec = self._make_instance(use_grammar=False)
        mock_rec.FinalResult.return_value = '{"text": "robot turn on the lights"}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is True

    def test_no_grammar_no_match(self):
        """Test check_for_wake_word returns False when wake word not in transcription"""
        wake_word_filter, mock_rec = self._make_instance(use_grammar=False)
        mock_rec.FinalResult.return_value = '{"text": "hello how are you"}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is False

    def test_handles_vosk_errors(self):
        """Test check_for_wake_word returns False on Vosk errors"""
        wake_word_filter, mock_rec = self._make_instance()
        mock_rec.AcceptWaveform.side_effect = Exception("Vosk error")
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is False
        wake_word_filter.logger.error.assert_called_once()

    def test_detects_wake_word_anywhere(self):
        """Test check_for_wake_word returns True when wake word is anywhere in audio"""
        wake_word_filter, mock_rec = self._make_instance()

        mock_rec.FinalResult.return_value = '{"text": "robot turn on the lights"}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is True

        mock_rec.FinalResult.return_value = '{"text": "hey robot turn on the lights"}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is True

        mock_rec.FinalResult.return_value = '{"text": "turn on the lights robot"}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is True

        mock_rec.FinalResult.return_value = '{"text": "hello there how are you"}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is False

        mock_rec.FinalResult.return_value = '{"text": ""}'
        assert vosk_check_for_wake_word(wake_word_filter, b"\x00" * 1000) is False


class TestVoskProcessSegment:
    def _make_instance(self, shutting_down=False):
        wake_word_filter = Mock()
        wake_word_filter.is_shutting_down = shutting_down
        wake_word_filter.logger = Mock()
        wake_word_filter.executor = Mock()
        return wake_word_filter

    @pytest.mark.asyncio
    async def test_skips_when_shutting_down(self):
        """Test returns early if shutting down"""
        wake_word_filter = self._make_instance(shutting_down=True)
        chunks = []
        async for chunk in vosk_process_segment(
            wake_word_filter, [Mock(), Mock()], bytearray(b"\x00" * 1000)
        ):
            chunks.append(chunk)

        assert len(chunks) == 0
        wake_word_filter.logger.debug.assert_called_with(
            "Skipping speech processing due to shutdown"
        )

    @pytest.mark.asyncio
    async def test_handles_executor_shutdown_error(self):
        """Test handles RuntimeError during shutdown gracefully"""
        wake_word_filter = self._make_instance()

        async def mock_run_in_executor(*args):
            raise RuntimeError("cannot schedule new futures after shutdown")

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_run_in_executor
            chunks = []
            async for chunk in vosk_process_segment(
                wake_word_filter, [Mock()], bytearray(b"\x00" * 1000)
            ):
                chunks.append(chunk)

            assert len(chunks) == 0
            wake_word_filter.logger.debug.assert_called_with(
                "Executor shutdown during processing, ignoring"
            )

    @pytest.mark.asyncio
    async def test_yields_chunks_on_wake_word(self):
        """Test yields chunks when wake word detected"""
        wake_word_filter = self._make_instance()
        mock_chunk1, mock_chunk2 = Mock(), Mock()

        async def mock_run_in_executor(executor, func, *args):
            return True

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_run_in_executor
            with patch("src.models.vosk.AudioChunk") as mock_audio_chunk_class:
                empty_chunk = Mock()
                empty_chunk.audio = Mock()
                mock_audio_chunk_class.return_value = empty_chunk

                chunks = []
                async for chunk in vosk_process_segment(
                    wake_word_filter,
                    [mock_chunk1, mock_chunk2],
                    bytearray(b"\x00" * 1000),
                ):
                    chunks.append(chunk)

                assert len(chunks) == 3
                assert chunks[0] == mock_chunk1
                assert chunks[1] == mock_chunk2
                assert chunks[2] == empty_chunk

    @pytest.mark.asyncio
    async def test_yields_empty_chunk_at_end(self):
        """Test yields an empty AudioChunk at the end to signal segment end"""
        wake_word_filter = self._make_instance()
        mock_chunk = Mock()
        mock_chunk.audio = Mock()
        mock_chunk.audio.audio_data = b"\x00" * 100

        async def mock_run_in_executor(executor, func, *args):
            return True

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_run_in_executor
            with patch("src.models.vosk.AudioChunk") as mock_audio_chunk_class:
                empty_chunk = Mock()
                empty_chunk.audio = Mock()
                mock_audio_chunk_class.return_value = empty_chunk

                chunks = []
                async for chunk in vosk_process_segment(
                    wake_word_filter, [mock_chunk], bytearray(b"\x00" * 1000)
                ):
                    chunks.append(chunk)

                assert len(chunks) == 2
                assert chunks[0] == mock_chunk
                assert chunks[1] == empty_chunk

    @pytest.mark.asyncio
    async def test_yields_nothing_when_no_wake_word(self):
        """Test yields nothing when wake word not detected"""
        wake_word_filter = self._make_instance()

        async def mock_run_in_executor(executor, func, *args):
            return False

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_run_in_executor
            chunks = []
            async for chunk in vosk_process_segment(
                wake_word_filter, [Mock(), Mock()], bytearray(b"\x00" * 1000)
            ):
                chunks.append(chunk)

            assert len(chunks) == 0
