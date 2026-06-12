"""
Tests for multi-client get_audio support.

SegmentBroadcaster (src/models/_broadcast.py) distributes each detected
segment to every subscribed get_audio caller; WakeWordFilter owns the single
shared detection pipeline that feeds it. The detection pipeline itself is
covered by test_integration.py — these tests cover the broadcast plumbing
and the pipeline lifecycle.
"""

import asyncio
import logging

import pytest
from unittest.mock import AsyncMock, Mock

from src.models._broadcast import SegmentBroadcaster, _Subscriber, _STREAM_END
from src.models.wake_word_filter import WakeWordFilter
from viam.components.audio_in import AudioResponse as AudioChunk


def create_audio_chunk(audio_data: bytes) -> AudioChunk:
    chunk = AudioChunk()
    chunk.audio.audio_data = audio_data
    return chunk


SENTINEL = b""

logger = logging.getLogger("test-multi-client")


async def collect(gen, n):
    """Read n chunks from an async generator with a timeout."""
    out = []
    for _ in range(n):
        out.append(await asyncio.wait_for(gen.__anext__(), timeout=1.0))
    return [c.audio.audio_data for c in out]


class TestSegmentBroadcaster:
    @pytest.mark.asyncio
    async def test_two_subscribers_receive_identical_segments(self):
        b = SegmentBroadcaster(logger)
        s1 = b.stream(b.subscribe())
        s2 = b.stream(b.subscribe())
        assert b.count == 2

        for data in (b"\x01\x01", b"\x02\x02", SENTINEL):
            b.broadcast(create_audio_chunk(data))

        assert await collect(s1, 3) == [b"\x01\x01", b"\x02\x02", b""]
        assert await collect(s2, 3) == [b"\x01\x01", b"\x02\x02", b""]

    @pytest.mark.asyncio
    async def test_mid_segment_subscriber_starts_at_next_segment(self):
        b = SegmentBroadcaster(logger)
        s1 = b.stream(b.subscribe())
        b.broadcast(create_audio_chunk(b"\x01"))  # segment 1 opens

        late = b.subscribe()  # joins mid-segment
        s2 = b.stream(late)
        assert not late.active

        b.broadcast(create_audio_chunk(b"\x02"))
        b.broadcast(create_audio_chunk(SENTINEL))  # segment 1 ends
        assert late.active  # activated at the boundary
        b.broadcast(create_audio_chunk(b"\x03"))  # segment 2 opens

        assert await collect(s1, 4) == [b"\x01", b"\x02", b"", b"\x03"]
        # The late joiner sees nothing from segment 1 — not even its sentinel.
        assert await collect(s2, 1) == [b"\x03"]

    @pytest.mark.asyncio
    async def test_slow_subscriber_drops_oldest_without_blocking(self):
        b = SegmentBroadcaster(logger)
        sub = b.subscribe()
        sub.queue = asyncio.Queue(maxsize=2)

        for data in (b"\x01", b"\x02", b"\x03"):  # third overflows: drops \x01
            b.broadcast(create_audio_chunk(data))

        assert sub.queue.get_nowait().audio.audio_data == b"\x02"
        assert sub.queue.get_nowait().audio.audio_data == b"\x03"

    @pytest.mark.asyncio
    async def test_end_all_terminates_stream_and_unsubscribes(self):
        b = SegmentBroadcaster(logger)
        s1 = b.stream(b.subscribe())
        b.broadcast(create_audio_chunk(b"\x01"))
        b.end_all()

        got = [c.audio.audio_data async for c in s1]
        assert got == [b"\x01"]  # queued data drains before the end marker
        assert b.count == 0  # generator cleanup unsubscribed

    @pytest.mark.asyncio
    async def test_cancelled_consumer_unsubscribes(self):
        # Mirrors a gRPC client disconnecting: the SDK's iteration task gets
        # cancelled while the generator is parked at queue.get().
        b = SegmentBroadcaster(logger)
        s1 = b.stream(b.subscribe())

        consumer = asyncio.create_task(collect(s1, 1))
        await asyncio.sleep(0)  # let it block on the empty queue
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert b.count == 0


def bare_filter() -> WakeWordFilter:
    """A WakeWordFilter with only the state the pipeline lifecycle needs."""
    f = WakeWordFilter.__new__(WakeWordFilter)
    f.logger = logger
    f.is_shutting_down = False
    f._broadcaster = SegmentBroadcaster(logger)
    f._pipeline_task = None
    mic = AsyncMock()
    mic.get_properties.return_value = Mock(sample_rate_hz=16000, num_channels=1)
    f.microphone_client = mic
    return f


class TestPipelineLifecycle:
    @pytest.mark.asyncio
    async def test_single_pipeline_for_many_subscribers(self):
        f = bare_filter()
        starts = 0
        parked = asyncio.Event()

        async def fake_pipeline():
            nonlocal starts
            starts += 1
            await parked.wait()

        f._run_pipeline = fake_pipeline

        await f.get_audio("pcm16", 0, 0)
        await f.get_audio("pcm16", 0, 0)
        await asyncio.sleep(0)  # let the task start

        assert starts == 1
        assert f._broadcaster.count == 2

        parked.set()
        await asyncio.wait_for(f._pipeline_task, timeout=1.0)

        # Next subscriber after the pipeline ended starts a fresh one.
        await f.get_audio("pcm16", 0, 0)
        await asyncio.sleep(0)
        assert starts == 2

    @pytest.mark.asyncio
    async def test_rejects_non_pcm16(self):
        f = bare_filter()
        with pytest.raises(ValueError, match="PCM16"):
            await f.get_audio("mp3", 0, 0)
        assert f._broadcaster.count == 0
        assert f._pipeline_task is None

    @pytest.mark.asyncio
    async def test_pipeline_exits_when_last_subscriber_leaves(self):
        f = bare_filter()
        f.silence_duration_ms = 900
        f.min_speech_duration_ms = 300
        f.detection_running = True

        async def mic_stream():
            # Continuous mic: yields empty keepalive chunks forever.
            while True:
                await asyncio.sleep(0.001)
                yield create_audio_chunk(SENTINEL)

        f.microphone_client.get_audio = AsyncMock(return_value=mic_stream())

        s1 = await f.get_audio("pcm16", 0, 0)
        consumer = asyncio.create_task(collect(s1.__aiter__(), 1))
        await asyncio.sleep(0.01)
        assert f._pipeline_task is not None and not f._pipeline_task.done()

        consumer.cancel()  # last subscriber disconnects
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert f._broadcaster.count == 0
        await asyncio.wait_for(f._pipeline_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_close_cancels_pipeline_and_ends_subscribers(self):
        f = bare_filter()
        f.oww_model = None
        parked = asyncio.Event()

        async def fake_pipeline():
            try:
                await parked.wait()
            finally:
                f._broadcaster.end_all()

        f._run_pipeline = fake_pipeline

        s1 = await f.get_audio("pcm16", 0, 0)
        await asyncio.sleep(0)

        await f.close()
        assert f._pipeline_task.done()

        got = [c async for c in s1]
        assert got == []
        assert f._broadcaster.count == 0
