import json

import pytest
from aiohttp import web

import litellm
from litellm import Router
from litellm.spark.backend_priority import BackendPriority
from litellm.spark.effort_normalize import EffortNormalize


def _kwargs(model, key_metadata=None, field="metadata", **extra):
    data = {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}
    if key_metadata is not None:
        data[field] = {"user_api_key_metadata": key_metadata}
    return data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected"),
    [("hosted_vllm/deepseek-v4.1-flash", -5), ("hosted_vllm/qwen3.8-27b-nvfp4", 5)],
)
async def test_backend_priority_uses_backend_convention(model, expected):
    result = await BackendPriority().async_pre_call_deployment_hook(
        _kwargs(model, {"backend_priority": 5}, extra_body={"keep": 1}), "acompletion"
    )
    assert result["extra_body"] == {"keep": 1, "priority": expected}


@pytest.mark.asyncio
async def test_backend_priority_reads_litellm_metadata():
    result = await BackendPriority().async_pre_call_deployment_hook(
        _kwargs("hosted_vllm/qwen3.8-27b-nvfp4", {"backend_priority": 3}, field="litellm_metadata"),
        "aresponses",
    )
    assert result["extra_body"] == {"priority": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "key_metadata"),
    [
        ("hosted_vllm/qwen3.8-27b-nvfp4", None),
        ("hosted_vllm/qwen3.8-27b-nvfp4", {}),
        ("hosted_vllm/qwen3.8-27b-nvfp4", {"backend_priority": "high"}),
        ("hosted_vllm/qwen3.8-27b-nvfp4", {"backend_priority": True}),
        ("zai/glm-5", {"backend_priority": 5}),
    ],
)
async def test_backend_priority_leaves_other_requests_unchanged(model, key_metadata):
    assert await BackendPriority().async_pre_call_deployment_hook(_kwargs(model, key_metadata), "acompletion") is None


@pytest.mark.asyncio
async def test_effort_normalize_deepseek_enables_thinking():
    result = await EffortNormalize().async_pre_call_deployment_hook(
        _kwargs("hosted_vllm/deepseek-v4.1-flash", reasoning_effort="high"), "acompletion"
    )
    assert result["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": "high"}
    assert result["include_reasoning"] is True


@pytest.mark.asyncio
async def test_effort_normalize_qwen_clamps():
    result = await EffortNormalize().async_pre_call_deployment_hook(
        _kwargs("hosted_vllm/qwen3.8-27b-nvfp4", reasoning_effort="max"), "acompletion"
    )
    assert result["reasoning_effort"] == "xhigh"


@pytest.mark.asyncio
async def test_routed_request_body_carries_priority_and_effort(monkeypatch):
    bodies = []

    async def chat(request):
        bodies.append(await request.json())
        return web.json_response(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3.8-27b-nvfp4",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setattr(litellm, "callbacks", [EffortNormalize(), BackendPriority()])
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "local-small",
                    "litellm_params": {
                        "model": "hosted_vllm/qwen3.8-27b-nvfp4",
                        "api_base": f"http://127.0.0.1:{port}/v1",
                        "api_key": "test",
                    },
                }
            ]
        )
        await router.acompletion(
            model="local-small",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="high",
            metadata={"user_api_key_metadata": {"backend_priority": 7}},
        )
    finally:
        await runner.cleanup()
    assert bodies[0]["priority"] == 7
    assert bodies[0]["reasoning_effort"] == "xhigh"
    assert "backend_priority" not in json.dumps(bodies[0])
