"""Per-key request priority for local model backends.

Registered in the LiteLLM config ``callbacks`` list as
``litellm.spark.backend_priority.handler``.

A virtual key opts in with an integer ``backend_priority`` in its metadata;
higher means served earlier, and a key without it keeps the backend default.
After routing, the selected deployment receives the value in its own
convention as the request body's ``priority``:

- vLLM (DeepSeek V4.1, ``--scheduling-policy priority``): lower is earlier,
  default 0, so the value is negated.
- SGLang (Qwen3.8, ``--enable-priority-scheduling``): higher is earlier,
  so the value is sent as is.

Other deployments are left unchanged. Only waiting requests are reordered;
nothing is rejected or rate limited. See spark-manage
docs/configuration-decisions/2026-09-25-litellm-fork-and-key-priority.md.
"""

from __future__ import annotations

from litellm.integrations.custom_logger import CustomLogger

METADATA_KEY = "backend_priority"
# Served-model prefix -> sign applied to the "higher is earlier" key value.
BACKEND_SIGN = {"deepseek-v4.1": -1, "qwen3.8": 1}


def key_priority(kwargs: dict) -> int | None:
    for field in ("metadata", "litellm_metadata"):
        metadata = kwargs.get(field)
        if not isinstance(metadata, dict):
            continue
        key_metadata = metadata.get("user_api_key_metadata")
        if isinstance(key_metadata, dict):
            value = key_metadata.get(METADATA_KEY)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


class BackendPriority(CustomLogger):
    async def async_pre_call_deployment_hook(self, kwargs: dict, call_type) -> dict | None:
        priority = key_priority(kwargs)
        if priority is None:
            return None
        model = kwargs.get("model", "").split("/")[-1].lower()
        sign = next((s for prefix, s in BACKEND_SIGN.items() if model.startswith(prefix)), None)
        if sign is None:
            return None
        data = dict(kwargs)
        data["extra_body"] = {**(data.get("extra_body") or {}), "priority": sign * priority}
        return data


# LiteLLM resolves callback entries as module attributes and calls their
# hook methods bound; register the instance, not the class.
handler = BackendPriority()
