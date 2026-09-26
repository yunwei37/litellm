"""Unified reasoning-effort normalization for local model roles.

Registered in the LiteLLM config ``callbacks`` list as
``litellm.spark.effort_normalize.handler``. It rewrites the OpenAI-style top-level ``reasoning_effort``
parameter into what the selected backend understands on every attempt, so
every client uses one switch (none/minimal/low/medium/high/xhigh/max) for
both local aliases.

- Qwen3.8 chat template accepts only xhigh (its default), medium and low, so
  minimal clamps to low and high/max clamp to xhigh.
- DeepSeek V4.1 accepts all seven values natively, but its serving default is
  thinking off and this build drops parsed reasoning unless the request sets
  ``include_reasoning``, so a non-none effort injects
  ``chat_template_kwargs`` thinking plus ``include_reasoning``.

Uses LiteLLM's native deployment hook, including cross-family fallbacks.
See spark-manage docs/configuration-decisions/2026-09-12-gateway-fallback-effort.md
and docs/configuration-decisions/2026-09-13-model-specific-market-cache-pricing.md.
"""

from __future__ import annotations

from litellm.integrations.custom_logger import CustomLogger

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
QWEN_CLAMPS = {"minimal": "low", "high": "xhigh", "max": "xhigh"}


class EffortNormalize(CustomLogger):
    async def async_pre_call_deployment_hook(self, kwargs: dict, call_type) -> dict | None:
        effort = kwargs.get("reasoning_effort")
        if call_type not in ("completion", "acompletion") or effort not in EFFORTS:
            return None
        model = kwargs.get("model", "").split("/")[-1].lower()
        data = dict(kwargs)
        template = dict(data.get("chat_template_kwargs") or {})
        if model.startswith("deepseek-v4.1"):
            template.setdefault("thinking", effort != "none")
            if effort != "none":
                template.setdefault("reasoning_effort", effort)
                data.setdefault("include_reasoning", True)
            data["chat_template_kwargs"] = template
        elif model.startswith("qwen3.8"):
            data["reasoning_effort"] = QWEN_CLAMPS.get(effort, effort)
            if "reasoning_effort" in template:
                native = template["reasoning_effort"]
                template["reasoning_effort"] = QWEN_CLAMPS.get(native, native)
                data["chat_template_kwargs"] = template
        else:
            return None
        return data


# LiteLLM resolves callback entries as module attributes and calls their
# hook methods bound; register the instance, not the class.
handler = EffortNormalize()
