"""Unified model registry for BlueCodeAgent.

``build_client(name)`` returns a *factory* ``f(sys_msg) -> client`` used by every
detector for both the base model (``--model``) and the constitution summarizer
(``--constitution_model``)::

    Client = build_client(args.model)                    # a factory
    client = Client(system_prompt)                       # -> a client instance
    constitution_summarizer = build_client(args.constitution_model)(constitution_sys_prompt)

API-backed models create a fresh client per call (cheap). Local HuggingFace models
(``llama``, ``qwen3-*``, ``local_qwen35*``) load ONE shared instance per process and
reuse it by swapping the system message, so the weights load only once — even when the
base model and the constitution model are the same local model.
"""
from models.client import (
    ClaudeClient,
    MetaLlamaClient,
    OpenaiClient,
    Qwen3Client,
    QwenClient,
    TogetherLlamaClient,
    TogetherQwen35Client,
)

# name -> (client class, constructor kwargs). API-backed; a fresh client per call is fine.
_API_MODELS = {
    "gpt4o":                (OpenaiClient,         {"model_name": "gpt-4o"}),
    "gpt5":                 (OpenaiClient,         {"model_name": "gpt-5"}),
    "qwen":                 (QwenClient,           {}),
    "together_llama":       (TogetherLlamaClient,  {}),
    "together_qwen35":      (TogetherQwen35Client, {}),
    "together_llama33_70b": (QwenClient,           {"model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo"}),
    "claude":               (ClaudeClient,         {}),
    "claude_sonnet45":      (ClaudeClient,         {"model_name": "claude-sonnet-4-5-20250929"}),
}

# Local HuggingFace models: build ONE shared instance per process (lazy, load-once).
_LOCAL_BUILDERS = {
    "llama":              lambda: MetaLlamaClient("placeholder"),
    "local_qwen35":       lambda: Qwen3Client("placeholder", model_name="Qwen/Qwen3.5-9B", enable_thinking=False),
    "local_qwen35_think": lambda: Qwen3Client("placeholder", model_name="Qwen/Qwen3.5-9B", enable_thinking=True),
}
for _n, _mid in {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "qwen3-1.7b": "Qwen/Qwen3-1.7B",
                 "qwen3-4b": "Qwen/Qwen3-4B", "qwen3-8b": "Qwen/Qwen3-8B"}.items():
    _LOCAL_BUILDERS[_n] = lambda m=_mid: Qwen3Client("placeholder", model_name=m)

_local_cache = {}


def _shared_factory(key, builder):
    """Factory that lazily builds ONE instance for `key` and reuses it across calls."""
    def factory(sys_msg):
        inst = _local_cache.get(key)
        if inst is None:
            inst = builder()
            _local_cache[key] = inst
        inst.sys_msg = sys_msg
        return inst
    return factory


def build_client(name):
    """Return a factory ``f(sys_msg) -> client`` for the model `name`."""
    if name in _API_MODELS:
        cls, kwargs = _API_MODELS[name]
        return lambda sys_msg: cls(sys_msg, **kwargs)
    if name in _LOCAL_BUILDERS:
        return _shared_factory(name, _LOCAL_BUILDERS[name])
    raise ValueError(f"Unsupported model: {name!r}")


def available_models():
    """Sorted list of every registered model name."""
    return sorted(set(_API_MODELS) | set(_LOCAL_BUILDERS))
