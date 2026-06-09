"""Tests for the wakeword-miss-sensor model."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from src.models.wakeword_miss_sensor import (
    WakewordMissSensor,
    _DEFAULT_MAX_QUEUE_SIZE,
    _UPLOAD_COMPONENT_TYPE,
    _UPLOAD_FILE_EXT,
    _UPLOAD_METHOD_NAME,
)
from viam.errors import NoCaptureToStoreError


def make_sensor(uploader=None, max_queue_size=100, dataset_ids=None, part_id="test-part"):
    """Construct a WakewordMissSensor wired with a stub uploader, bypassing
    `new()` which reads env vars and contacts Viam app."""
    s = WakewordMissSensor("test-sensor")
    s.logger = Mock()
    s.dataset_ids = list(dataset_ids or [])
    s.component_name_override = None
    s.max_queue_size = max_queue_size
    s._pending = []
    s._last_reading = None
    s._lock = asyncio.Lock()
    s._part_id = part_id

    if uploader is None:
        s._viam_client = None
    else:
        fake_client = Mock()
        fake_client.data_client = Mock()
        fake_client.data_client.binary_data_capture_upload = uploader
        s._viam_client = fake_client
    return s


def sample_reading(capture_id="cap-1", score=0.6, wake_word="gambit"):
    return {
        "capture_id": capture_id,
        "wake_word": wake_word,
        "max_oww_score": score,
        "oww_threshold": 0.8,
        "oww_model_path": "/tmp/gambit.onnx",
        "audio_bytes": 1024,
        "duration_ms": 32.0,
    }


@pytest.mark.asyncio
async def test_get_readings_empty_queue_raises():
    s = make_sensor()
    with pytest.raises(NoCaptureToStoreError):
        await s.get_readings(extra={"fromDataManagement": True})


@pytest.mark.asyncio
async def test_push_miss_queues_and_dm_get_readings_returns_it():
    upload = AsyncMock(return_value="binid-1")
    s = make_sensor(uploader=upload)

    cap_id, bin_id = await s._push(
        sample_reading(capture_id="cap-1"), b"RIFFfake"
    )
    assert cap_id == "cap-1"
    assert bin_id == "binid-1"
    upload.assert_awaited_once()

    got = await s.get_readings(extra={"fromDataManagement": True})
    assert got["capture_id"] == "cap-1"
    assert got["binary_data_id"] == "binid-1"
    assert got["wake_word"] == "gambit"
    assert got["max_oww_score"] == 0.6

    # Queue should be empty.
    with pytest.raises(NoCaptureToStoreError):
        await s.get_readings(extra={"fromDataManagement": True})


@pytest.mark.asyncio
async def test_push_miss_fifo_across_multiple_pushes():
    upload = AsyncMock(return_value="binid")
    s = make_sensor(uploader=upload)
    for cid in ("a", "b", "c"):
        await s._push(sample_reading(capture_id=cid), b"WAV")
    for want in ("a", "b", "c"):
        got = await s.get_readings(extra={"fromDataManagement": True})
        assert got["capture_id"] == want
    with pytest.raises(NoCaptureToStoreError):
        await s.get_readings(extra={"fromDataManagement": True})


@pytest.mark.asyncio
async def test_push_miss_upload_error_queues_row_with_empty_binary_id():
    upload = AsyncMock(side_effect=RuntimeError("network down"))
    s = make_sensor(uploader=upload)

    cap_id, bin_id = await s._push(
        sample_reading(capture_id="cap-err"), b"WAV"
    )
    assert cap_id == "cap-err"
    assert bin_id == ""

    got = await s.get_readings(extra={"fromDataManagement": True})
    assert got["capture_id"] == "cap-err"
    assert got["binary_data_id"] == ""


@pytest.mark.asyncio
async def test_push_miss_without_viam_client_still_queues_row():
    s = make_sensor(uploader=None)
    cap_id, bin_id = await s._push(
        sample_reading(capture_id="cap-no-upload"), b"WAV"
    )
    assert cap_id == "cap-no-upload"
    assert bin_id == ""

    got = await s.get_readings(extra={"fromDataManagement": True})
    assert got["capture_id"] == "cap-no-upload"
    assert got["binary_data_id"] == ""


@pytest.mark.asyncio
async def test_push_miss_max_queue_size_drops_oldest():
    s = make_sensor(uploader=AsyncMock(return_value="binid"), max_queue_size=2)
    for cid in ("a", "b", "c", "d"):
        await s._push(sample_reading(capture_id=cid), b"WAV")

    got1 = await s.get_readings(extra={"fromDataManagement": True})
    got2 = await s.get_readings(extra={"fromDataManagement": True})
    assert got1["capture_id"] == "c"
    assert got2["capture_id"] == "d"
    with pytest.raises(NoCaptureToStoreError):
        await s.get_readings(extra={"fromDataManagement": True})


@pytest.mark.asyncio
async def test_push_miss_upload_tags_and_metadata():
    upload = AsyncMock(return_value="binid")
    s = make_sensor(uploader=upload, dataset_ids=["ds-1"])

    await s._push(
        sample_reading(capture_id="cap-tags", wake_word="gambit"), b"WAV"
    )

    upload.assert_awaited_once()
    kwargs = upload.await_args.kwargs
    assert kwargs["part_id"] == "test-part"
    assert kwargs["component_type"] == _UPLOAD_COMPONENT_TYPE
    assert kwargs["method_name"] == _UPLOAD_METHOD_NAME
    assert kwargs["file_extension"] == _UPLOAD_FILE_EXT
    assert kwargs["dataset_ids"] == ["ds-1"]
    assert set(kwargs["tags"]) == {
        "wakeword_miss",
        "capture_cap-tags",
        "wake_gambit",
    }


@pytest.mark.asyncio
async def test_push_miss_skips_wake_tag_when_word_empty():
    upload = AsyncMock(return_value="binid")
    s = make_sensor(uploader=upload)

    await s._push(
        sample_reading(capture_id="cap-no-word", wake_word=""), b"WAV"
    )

    kwargs = upload.await_args.kwargs
    assert set(kwargs["tags"]) == {"wakeword_miss", "capture_cap-no-word"}


@pytest.mark.asyncio
async def test_push_miss_generates_capture_id_when_missing():
    upload = AsyncMock(return_value="binid")
    s = make_sensor(uploader=upload)

    reading = sample_reading()
    reading["capture_id"] = ""
    cap_id, _ = await s._push(reading, b"WAV")
    assert cap_id  # non-empty UUID generated
    # Pushed tag should match the generated id.
    kwargs = upload.await_args.kwargs
    assert f"capture_{cap_id}" in kwargs["tags"]


@pytest.mark.asyncio
async def test_get_readings_non_dm_returns_sticky_last_reading():
    """Non-DM callers (UI live preview) see the most recent reading on every
    poll without consuming the queue. DM-flagged polls keep pop semantics."""
    s = make_sensor(uploader=AsyncMock(return_value="binid"))

    # Empty queue: even non-DM polls raise NoCaptureToStoreError.
    with pytest.raises(NoCaptureToStoreError):
        await s.get_readings(extra=None)

    await s._push(sample_reading(capture_id="ui-1"), b"WAV")

    # Multiple non-DM polls return the same row, never empty.
    for _ in range(3):
        got = await s.get_readings(extra=None)
        assert got["capture_id"] == "ui-1"

    # DM-flagged poll pops the row from the queue.
    got = await s.get_readings(extra={"fromDataManagement": True})
    assert got["capture_id"] == "ui-1"

    # Queue empty, but non-DM polls still see the sticky last reading.
    got = await s.get_readings(extra=None)
    assert got["capture_id"] == "ui-1"


def test_max_queue_size_defaults():
    assert _DEFAULT_MAX_QUEUE_SIZE == 1000
