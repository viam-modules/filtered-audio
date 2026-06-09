"""
wakeword-miss-sensor

A Viam sensor component that captures one tabular reading per OWW near-miss
segment from the wake-word-filter, and uploads the corresponding WAV to the
Viam binary store. Audio and metadata link via binary_data_id (joined in MQL).

Push API: wake-word-filter calls
`await sensor.do_command({"command": "push_miss", ...})`. Even within the
same module the call goes through Viam's gRPC SensorClient (the dep system
always returns the client, never the Python instance), so we use DoCommand
as the push entry point. Traffic stays on the module's unix socket — no
real cross-process cost — but the WAV bytes must be base64-encoded in the
payload because proto Struct can't carry raw bytes.

get_readings() returns one queued row per poll, or raises
NoCaptureToStoreError when empty (so the Viam data manager skips the poll
instead of writing a blank row).
"""

import base64
import logging
import os
import uuid
from typing import (
    Any,
    ClassVar,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from typing_extensions import Self

from viam.app.viam_client import ViamClient
from viam.components.sensor import Sensor
from viam.errors import NoCaptureToStoreError
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import from_dm_from_extra, struct_to_dict


AUDIO_SAMPLE_RATE_HZ = 16000

_UPLOAD_COMPONENT_TYPE = "rdk:component:sensor"
_UPLOAD_METHOD_NAME = "Readings"
_UPLOAD_FILE_EXT = ".wav"

_ENV_PART_ID = "VIAM_MACHINE_PART_ID"

_DEFAULT_MAX_QUEUE_SIZE = 1000


class WakewordMissSensor(Sensor, EasyResource):
    """Queue-backed sensor for OWW near-miss captures."""

    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam", "filtered-audio"), "wakeword-miss-sensor"
    )

    # Configured at construct time.
    logger: logging.Logger
    dataset_ids: List[str]
    component_name_override: Optional[str]
    max_queue_size: int

    # Runtime state.
    _pending: List[Mapping[str, Any]]
    _viam_client: Optional[ViamClient]
    _part_id: str

    @classmethod
    def new(
        cls,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> Self:
        instance = cls(config.name)
        attrs = struct_to_dict(config.attributes)
        instance.logger = getLogger(cls.__name__)
        instance.dataset_ids = list(attrs.get("dataset_ids") or [])
        cno = attrs.get("component_name", "")
        instance.component_name_override = str(cno).strip() or None
        mqs = attrs.get("max_queue_size", _DEFAULT_MAX_QUEUE_SIZE)
        instance.max_queue_size = int(mqs) if mqs else _DEFAULT_MAX_QUEUE_SIZE

        instance._pending = []
        instance._last_reading = None
        instance._viam_client = None
        instance._part_id = os.getenv(_ENV_PART_ID, "")

        if not instance._part_id:
            instance.logger.warning(
                "%s not set — binary uploads disabled, tabular queue still works",
                _ENV_PART_ID,
            )
        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        ds = attrs.get("dataset_ids", None)
        if ds is not None and not (
            isinstance(ds, list) and all(isinstance(x, str) for x in ds)
        ):
            raise ValueError("dataset_ids must be a list of strings")
        mqs = attrs.get("max_queue_size", None)
        if mqs is not None:
            if not isinstance(mqs, (int, float)) or mqs % 1 != 0:
                raise ValueError("max_queue_size must be a whole number")
            if mqs <= 0:
                raise ValueError("max_queue_size must be positive")
        return [], []

    async def _ensure_viam_client(self) -> Optional[ViamClient]:
        if self._viam_client is not None:
            return self._viam_client
        if not self._part_id:
            return None
        try:
            self._viam_client = await ViamClient.create_from_env_vars()
            self.logger.info(
                "ready with binary uploads enabled (part_id=%s)", self._part_id
            )
        except Exception as e:
            self.logger.warning(
                "Viam app client unavailable (%s) — binary uploads disabled, "
                "tabular queue still works",
                e,
            )
            self._viam_client = None
        return self._viam_client

    async def get_readings(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, Any]:
        from_dm = from_dm_from_extra(extra)
        if from_dm:
            # Data manager: strict queue-pop semantics so each row gets
            # captured exactly once.
            if not self._pending:
                raise NoCaptureToStoreError()
            return self._pending.pop(0)
        # Non-data-manager caller (Viam app live preview, Test panel,
        # manual SDK call): show the most recent reading without consuming
        # the queue. Avoids flicker in the UI.
        if self._last_reading is not None:
            return self._last_reading
        raise NoCaptureToStoreError()

    async def do_command(
        self,
        command: Mapping[str, Any],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, Any]:
        name = command.get("command", "")
        if name == "push_miss":
            payload = dict(command)
            payload.pop("command", None)
            b64 = payload.pop("audio_wav_b64", "")
            wav_bytes = base64.b64decode(b64) if isinstance(b64, str) and b64 else b""
            cap_id, bin_id = await self._push(payload, wav_bytes)
            return {"capture_id": cap_id, "binary_data_id": bin_id}
        raise ValueError(f"unknown command {name!r} (supported: push_miss)")

    async def _push(
        self,
        reading: Mapping[str, Any],
        wav_bytes: bytes,
    ) -> Tuple[str, str]:
        bin_id = ""
        capture_id = str(reading.get("capture_id") or "")
        if not capture_id:
            capture_id = str(uuid.uuid4())

        client = await self._ensure_viam_client()
        if client is not None and wav_bytes:
            tags = [
                "wakeword_miss",
                f"capture_{capture_id}",
            ]
            wake_word = str(reading.get("wake_word") or "").strip()
            if wake_word:
                tags.append(f"wake_{wake_word}")

            component_name = self.component_name_override or self.name
            try:
                bin_id = await client.data_client.binary_data_capture_upload(
                    binary_data=wav_bytes,
                    part_id=self._part_id,
                    component_type=_UPLOAD_COMPONENT_TYPE,
                    component_name=component_name,
                    method_name=_UPLOAD_METHOD_NAME,
                    file_extension=_UPLOAD_FILE_EXT,
                    tags=tags,
                    dataset_ids=self.dataset_ids or None,
                )
            except Exception as e:
                self.logger.warning(
                    "binary upload failed for capture_id=%s: %s", capture_id, e
                )
                bin_id = ""

        row = {
            "capture_id": capture_id,
            "binary_data_id": bin_id,
            "wake_word": str(reading.get("wake_word") or ""),
            "max_oww_score": float(reading.get("max_oww_score", 0.0)),
            "oww_threshold": float(reading.get("oww_threshold", 0.0)),
            "oww_model_path": str(reading.get("oww_model_path") or ""),
            "audio_bytes": int(reading.get("audio_bytes", 0)),
            "duration_ms": float(reading.get("duration_ms", 0.0)),
        }

        self._pending.append(row)
        self._last_reading = row
        over = len(self._pending) - self.max_queue_size
        if over > 0:
            del self._pending[:over]
            self.logger.warning(
                "queue over max_queue_size=%d, dropped %d oldest reading(s)",
                self.max_queue_size,
                over,
            )
        self.logger.debug(
            "queued reading (capture_id=%s, depth=%d)", capture_id, len(self._pending)
        )
        return capture_id, bin_id

    async def close(self):
        if self._viam_client is not None:
            self._viam_client.close()
            self._viam_client = None
