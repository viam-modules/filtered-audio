"""
Segment broadcaster for the wake-word filter.

The detection pipeline runs once per component; every get_audio caller
subscribes here and receives a copy of each detected segment. Segments are
delimited by an empty "sentinel" chunk, and subscribers that join
mid-segment are held until the next boundary so nobody receives a torn
utterance.

Single-threaded: all methods are called from the asyncio event loop, so no
locking is needed.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, List

from viam.components.audio_in import AudioResponse as AudioChunk

# Per-subscriber queue depth. Segments are capped at ~15s of audio upstream,
# so a healthy consumer never accumulates anywhere near this many chunks.
_SUBSCRIBER_QUEUE_MAXSIZE = 2048

# Queue marker telling a subscriber its stream is over (pipeline ended or
# component closing).
_STREAM_END: Any = object()


class _Subscriber:
    """One get_audio caller's view of the shared pipeline output."""

    __slots__ = ("queue", "active")

    def __init__(self, active: bool):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAXSIZE)
        # Subscribers that join mid-utterance would receive a torn segment
        # (missing its leading chunks), so they stay inactive until the
        # current segment's end-sentinel, then receive whole segments only.
        self.active = active


class SegmentBroadcaster:
    """Distributes each detected segment to every subscribed get_audio caller."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._subscribers: List[_Subscriber] = []
        self._segment_open = False

    @property
    def has_subscribers(self) -> bool:
        return bool(self._subscribers)

    @property
    def count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> _Subscriber:
        sub = _Subscriber(active=not self._segment_open)
        self._subscribers.append(sub)
        self.logger.info("broadcast: subscriber added (total=%d)", self.count)
        return sub

    def unsubscribe(self, sub: _Subscriber) -> None:
        if sub in self._subscribers:
            self._subscribers.remove(sub)
        self.logger.info("broadcast: subscriber removed (total=%d)", self.count)

    def stream(self, sub: _Subscriber) -> AsyncGenerator[AudioChunk, None]:
        """The async generator a subscriber consumes; unsubscribes on close."""

        async def _gen() -> AsyncGenerator[AudioChunk, None]:
            try:
                while True:
                    item = await sub.queue.get()
                    if item is _STREAM_END:
                        return
                    yield item
            finally:
                self.unsubscribe(sub)

        return _gen()

    def broadcast(self, chunk: AudioChunk) -> None:
        """Fan one pipeline output chunk out to every active subscriber.

        An empty chunk is the segment-end sentinel: after delivering it,
        subscribers that joined mid-segment become active so they start
        receiving at the next whole segment.
        """
        is_sentinel = not chunk.audio.audio_data
        for sub in list(self._subscribers):
            if not sub.active:
                continue
            self._put_dropping_oldest(sub.queue, chunk)
        if is_sentinel:
            self._segment_open = False
            for sub in self._subscribers:
                sub.active = True
        else:
            self._segment_open = True

    def end_all(self) -> None:
        """Tell every subscriber its stream is over (queues drain first)."""
        self._segment_open = False
        for sub in list(self._subscribers):
            self._put_dropping_oldest(sub.queue, _STREAM_END)

    def end_subscriber(self, sub: _Subscriber) -> None:
        """End one subscriber's stream (its queued chunks drain first)."""
        if sub in self._subscribers:
            self._put_dropping_oldest(sub.queue, _STREAM_END)

    def _put_dropping_oldest(self, queue: asyncio.Queue, item: Any) -> None:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            # Slow consumer: drop its oldest entry rather than stall the
            # pipeline (and every other subscriber) behind it.
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(item)
            self.logger.warning("broadcast: subscriber queue full; dropped oldest chunk")
