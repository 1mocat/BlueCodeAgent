from openai import AzureOpenAI
# torch / transformers are optional: only the local HuggingFace model clients
# (Qwen3Client, MetaLlamaClient, QwenCoderClient, DSClient, CodeLlamaClient) use them.
# Import lazily so API-only tasks (OpenAI / Anthropic / Together) run without torch.
try:
    import torch
except Exception:
    torch = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = AutoTokenizer = None
from openai import OpenAI
try:
    from together import Together
except Exception:
    Together = None
import os 

class DeepSeekV3Client():
    def __init__(self, sys_msg):
        print("deepseek_init")
        self.client = Together()
        self.sys_msg = sys_msg
    def generate(self, query, **kwargs):
        response = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                    {
                        "role": "system",
                        "content": self.sys_msg,
                    },
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
            max_completion_tokens=4096
            )
        return response.choices[0].message.content

class OpenRouterClient():
    """Generic OpenRouter client — works with any model on openrouter.ai."""
    def __init__(self, sys_msg, model_name="qwen/qwen-2.5-72b-instruct"):
        print(f"OpenRouterClient initialized with model: {model_name}")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        self.sys_msg = sys_msg
        self.model_name = model_name

    def generate(self, query, **kwargs):
        import time as _time
        max_tokens = kwargs.get("max_tokens", 4096)
        temperature = kwargs.get("temperature", 0)
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": self.sys_msg},
                        {"role": "user", "content": query},
                    ],
                )
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                print(f"[OpenRouter] Empty response, retry {attempt+1}/5")
            except Exception as e:
                print(f"[OpenRouter] Error: {e}, retry {attempt+1}/5")
            _time.sleep(5 * (attempt + 1))
        return ""


class QwenClient():
    def __init__(self, sys_msg, model_name="Qwen/Qwen2.5-7B-Instruct-Turbo"):
        print(f"QwenClient initialized with model: {model_name}")
        self.client = Together()
        self.sys_msg = sys_msg
        self.model_name = model_name
    def generate(self, query, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": self.sys_msg},{"role": "user", "content": query}]
            )
        return response.choices[0].message.content

class Qwen3Client():
    """Qwen3 client using HuggingFace transformers (not Together API)"""
    def __init__(self, sys_msg="You are a helpful assistant.", model_name="Qwen/Qwen3-0.6B", enable_thinking=True):
        print(f"Qwen3Client initialized with model: {model_name}, enable_thinking={enable_thinking}")
        self.model_name = model_name
        self.sys_msg = sys_msg
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate(self, query, **kwargs):
        # Prepare messages (only user message, no system message)
        messages = [
            {"role": "user", "content": query}
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Get max_new_tokens from kwargs or use default; also accept max_tokens as alias
        max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 32768))
        _skip_keys = {"max_new_tokens", "max_tokens"}

        # Conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            **{k: v for k, v in kwargs.items() if k not in _skip_keys}
        )

        # Extract only the generated part (excluding input)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        if self.enable_thinking:
            # Parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            # Decode thinking content and final content
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        else:
            # No thinking mode - decode everything directly
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

        return content



class O3Client():
    def __init__(self, sys_msg="You are a helpful assistant."):
        print("openai_init")
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),  
        )
        self.sys_msg = sys_msg

    def generate(self, query, **kwargs):
        query = query
        max_attempts = 3
        attempt = 0
        response_content = ""

        while attempt < max_attempts:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.sys_msg,
                    },
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                # temperature=0,
                model="o3",  # or "gpt-4o"
                max_completion_tokens=4096,
                **kwargs
            )
            response_content = response.choices[0].message.content.strip()
            if response_content:
                break
            attempt += 1

        return response_content



class OpenaiClient():
    def __init__(self, sys_msg="You are a helpful assistant.", model_name="gpt-4o"):
        print(f"openai_init with model: {model_name}")
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"), 
        )
        self.sys_msg = sys_msg
        self.model_name = model_name

    def generate(self, query, **kwargs):
        import time as _time
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": self.sys_msg,
                        },
                        {
                            "role": "user",
                            "content": query,
                        }
                    ],
                    model=self.model_name,
                    max_completion_tokens=4096
                )
                text = response.choices[0].message.content
                if text:
                    return text
                print(f"[OpenaiClient] Empty response, retry {attempt+1}/{max_attempts}")
            except Exception as e:
                print(f"[OpenaiClient] Error: {e}, retry {attempt+1}/{max_attempts}")
            _time.sleep(5 * (attempt + 1))
        print(f"[OpenaiClient] All {max_attempts} attempts failed, returning empty string")
        return ""

import anthropic

class ClaudeClient():
    def __init__(self, sys_msg, model_name="claude-sonnet-4-6"):
        print(f"claude_init with model: {model_name}")
        self.sys_msg = sys_msg
        self.model_name = model_name
        self.client = anthropic.Anthropic(
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        )

    def generate(self, query, **kwargs):
        import time as _time
        temperature = kwargs.get("temperature", 0)
        max_tokens = kwargs.get("max_tokens", 4096)
        model = kwargs.get("model", self.model_name)
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                message = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=self.sys_msg,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": query
                                }
                            ]
                        }
                    ]
                )
                text = message.content[0].text if message.content else ""
                if text:
                    return text
                print(f"[Claude] Empty response, retry {attempt+1}/{max_attempts}")
            except Exception as e:
                print(f"[Claude] Error: {e}, retry {attempt+1}/{max_attempts}")
            _time.sleep(5 * (attempt + 1))
        print(f"[Claude] All {max_attempts} attempts failed, returning empty string")
        return ""

        
class QwenCoderClient():
    def __init__(self, sys_msg):
        print("transformers_init")
        self.model_name = "Qwen/Qwen2.5-Coder-7B"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.sys_msg = sys_msg

    def generate(self, query, **kwargs):
        messages = [
            {"role": "system", "content": self.sys_msg},
            {"role": "user", "content": query}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


class DSClient():
    def __init__(self, sys_msg):
        print("transformers_init")
        self.model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"  # Use the DeepSeek model

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,  # Use the bfloat16 dtype
            device_map="auto"  # Automatically map to available devices
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.sys_msg = sys_msg

    def generate(self, query, **kwargs):
        # Build the message format
        messages = [
            {"role": "system", "content": self.sys_msg},
            {"role": "user", "content": query}
        ]

        # Use the tokenizer to build chat-format text
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Convert the text into model input tensors
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate text
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=4096,
            **kwargs  # Allow extra keyword arguments to pass through
        )

        # Process the generated result, stripping the input portion
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the generated text and return it
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

class CodeLlamaClient():
    def __init__(self, sys_msg="You are a helpful assistant.", model_id="codellama/CodeLlama-7b-Instruct-hf"):
        print("CodeLlama initialization")
        self.model_name = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto"
        )
        self.sys_msg = sys_msg

    def generate(self, query, **kwargs):
        # Build the prompt (CodeLlama uses the [INST] ... [/INST] format)
        prompt = f"<s>[INST] {query.strip()} [/INST]"

        # Encode the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Generate the response
        output_ids = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=1024,
            temperature=0,
            **kwargs
        )

        # Keep only the model-generated part (drop the prompt part)
        generated_ids = output_ids[0][input_length:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(output_text)
        return output_text

class TogetherQwen35Client():
    """Qwen3.5-9B via Together API (base model, uses completions API with chat template)."""
    def __init__(self, sys_msg, model_name="Qwen/Qwen3.5-9B", enable_thinking=False):
        print(f"TogetherQwen35Client initialized with model: {model_name}, enable_thinking={enable_thinking}")
        self.client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ.get("TOGETHER_API_KEY"),
        )
        self.sys_msg = sys_msg
        self.model_name = model_name
        self.enable_thinking = enable_thinking

    def generate(self, query, **kwargs):
        import time as _time
        max_tokens = kwargs.get("max_tokens", 4096)
        temperature = kwargs.get("temperature", 0.7)
        # Use /no_think suffix to disable Qwen3's thinking mode
        think_suffix = "" if self.enable_thinking else "/no_think\n"
        prompt = (
            f"<|im_start|>system\n{self.sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n{think_suffix}"
        )
        for attempt in range(5):
            try:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["<|im_end|>"],
                )
                text = response.choices[0].text if response.choices else ""
                # Strip thinking tags if still present (fallback)
                if "</think>" in text:
                    text = text.split("</think>", 1)[1]
                elif "<think>" in text:
                    # Handle truncated thinking (no closing tag)
                    text = ""
                text = text.strip()
                if text:
                    return text
                print(f"[TogetherQwen35] Empty response, retry {attempt+1}/5")
            except Exception as e:
                print(f"[TogetherQwen35] Error: {e}, retry {attempt+1}/5")
            _time.sleep(5 * (attempt + 1))
        return ""


class TogetherLlamaClient():
    def __init__(self, sys_msg):
        print("TogetherLlamaClient meta-llama/Meta-Llama-3-8B-Instruct-Lite")
        self.client = Together()
        self.sys_msg = sys_msg
    def generate(self, query, **kwargs):
        response = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            messages=[{"role": "system", "content": self.sys_msg},{"role": "user", "content": query}]
            )
        return response.choices[0].message.content
        
class MetaLlamaClient:
    def __init__(self, sys_msg="You are a helpful assistant.", model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        print("MetaLlama initialization")
        self.model_name = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.sys_msg = sys_msg
        self.default_max_tokens = 4096
        self.default_top_p = 0

    def generate(self, query, max_new_tokens=None, top_p=None):
        # Build chat messages from the system message and user query
        messages = [
            {"role": "system", "content": self.sys_msg},
            {"role": "user", "content": query.strip()}
        ]

        # Encode the input
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        input_ids = input_ids.to(self.model.device)
        
        # Stop tokens
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens or self.default_max_tokens,
            eos_token_id=terminators,
            do_sample=False,
            top_p=top_p or self.default_top_p
        )

        ans = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        return ans.strip()

class CodexClient():
    """OpenAI Codex via Responses API."""
    def __init__(self, sys_msg="You are a helpful assistant.", model_name="gpt-5-codex"):
        print(f"CodexClient initialized with model: {model_name}")
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.sys_msg = sys_msg
        self.model_name = model_name

    def generate(self, query, **kwargs):
        import time as _time
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    instructions=self.sys_msg,
                    input=query,
                )
                if response.output_text:
                    return response.output_text
                print(f"[Codex] Empty response, retry {attempt+1}/{max_attempts}")
            except Exception as e:
                print(f"[Codex] Error: {e}, retry {attempt+1}/{max_attempts}")
            _time.sleep(5 * (attempt + 1))
        print(f"[Codex] All {max_attempts} attempts failed, returning empty string")
        return ""


class ClaudeCodeAgentClient():
    """Claude Code Agent via claude_code_sdk, using Anthropic API directly."""
    def __init__(self, sys_msg="You are a helpful assistant.", model_name="claude-sonnet-4-6"):
        print(f"ClaudeCodeAgentClient initialized with model: {model_name}")
        self.sys_msg = sys_msg
        self.model_name = model_name

    def generate(self, query, **kwargs):
        import asyncio
        import time as _time
        from claude_code_sdk import query as claude_query, ClaudeCodeOptions, TextBlock, AssistantMessage

        options = ClaudeCodeOptions(
            system_prompt=self.sys_msg,
            model=self.model_name,
            permission_mode="bypassPermissions",
            max_turns=1,
            allowed_tools=[],
            env={
                "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            },
        )

        async def _run():
            text_parts = []
            async for message in claude_query(prompt=query, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
            return "\n".join(text_parts)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = asyncio.run(_run())
                if result:
                    return result
                print(f"[ClaudeCodeAgent] Empty response, retry {attempt+1}/{max_attempts}")
            except Exception as e:
                print(f"[ClaudeCodeAgent] Error: {e}, retry {attempt+1}/{max_attempts}")
            _time.sleep(5 * (attempt + 1))
        return ""
