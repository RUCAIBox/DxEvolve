from typing import Any, List, Mapping, Dict, Optional, Tuple, Union
from transformers import AutoTokenizer

import torch
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.llms.base import LLM
import re
from agents.agent import STOP_WORDS
import openai


def split_tagged_chat_to_messages(
    text: str,
    tags: Dict[str, str],
) -> List[Dict[str, str]]:
    if text is None:
        return []

    s = str(text).replace("\ufeff", "").replace("\u200b", "")

    # Tag tokens (may be empty)
    sys_s = tags.get("system_tag_start", "") or ""
    sys_e = tags.get("system_tag_end", "") or ""
    usr_s = tags.get("user_tag_start", "") or ""
    usr_e = tags.get("user_tag_end", "") or ""
    ai_s  = tags.get("ai_tag_start", "") or ""
    ai_e  = tags.get("ai_tag_end", "") or ""

    # Build token map
    token_map = {}
    if sys_s: token_map[sys_s] = ("system", "start")
    if sys_e: token_map[sys_e] = ("system", "end")
    if usr_s: token_map[usr_s] = ("user", "start")
    if usr_e: token_map[usr_e] = ("user", "end")
    if ai_s:  token_map[ai_s]  = ("assistant", "start")
    if ai_e:  token_map[ai_e]  = ("assistant", "end")

    # If no tags found, fallback to single user message
    if not token_map:
        content = s.strip()
        return [{"role": "user", "content": content}] if content else []

    tokens_sorted = sorted(token_map.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(t) for t in tokens_sorted))

    messages: List[Dict[str, str]] = []
    cur_role: Optional[str] = None
    buf: List[str] = []
    last_end = 0

    def flush():
        nonlocal buf, cur_role, messages
        content = "".join(buf).strip()
        buf = []
        if not content:
            return
        if cur_role is None:
            if messages:
                messages[-1]["content"] = (messages[-1]["content"].rstrip() + "\n" + content).strip()
            else:
                messages.append({"role": "user", "content": content})
            return
        messages.append({"role": cur_role, "content": content})

    for m in pattern.finditer(s):
        seg = s[last_end:m.start()]
        if seg:
            buf.append(seg)

        tok = m.group(0)
        role, kind = token_map[tok]
        if kind == "start":
            flush()
            cur_role = role
        else:
            flush()
            cur_role = None
        last_end = m.end()

    tail = s[last_end:]
    if tail:
        buf.append(tail)
    flush()

    # Merge adjacent same-role messages
    merged: List[Dict[str, str]] = []
    for msg in messages:
        c = (msg.get("content") or "").strip()
        if not c:
            continue
        r = msg.get("role") or "user"
        if merged and merged[-1]["role"] == r:
            merged[-1]["content"] = (merged[-1]["content"].rstrip() + "\n" + c).strip()
        else:
            merged.append({"role": r, "content": c})
    messages = merged

    # If last is assistant, merge into previous and ensure last is user
    if messages and messages[-1]["role"] == "assistant":
        assistant_text = messages.pop()["content"].strip()
        if messages:
            messages[-1]["content"] = (messages[-1]["content"].rstrip() + "\n" + assistant_text).strip()
        else:
            messages.append({"role": "user", "content": assistant_text})

    if messages and messages[-1]["role"] != "user":
        messages[-1]["role"] = "user"

    return messages


class CustomLLM(LLM):
    model_name: str
    max_context_length: int
    probabilities: torch.Tensor = None
    exllama: bool = False
    use_vllm: bool = False
    use_api: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    truncation_side: str = "left"
    model: Any
    generator: Any
    tokenizer: Any
    seed: int
    self_consistency: bool = False
    api_model_name: str = None
    params: Dict[str, Any] = None

    openai_api_key: str = None
    tags: Dict[str, str] = None
    is_baidu: bool = False
    is_reason: bool = False

    @property
    def _llm_type(self) -> Any:
        return "custom"

    @property
    def _llm_name(self) -> str:
        return self.model_name

    @property
    def _llm_device(self) -> str:
        return self.model.device

    @property
    def _llm_8bit(self) -> bool:
        return self.load_in_8bit

    @property
    def _llm_4bit(self) -> bool:
        return self.load_in_4bit

    @property
    def _llm_truncation_side(self) -> str:
        return self.truncation_side

    def load_model(self, base_models: str) -> None:
        if base_models is None:
            base_models = self.model_name
        self.model = openai.OpenAI(
            api_key="empty",
            base_url="Your Local Host" # For example: "http://127.0.0.1:8002/v1/"
        )
        self.use_api = True
        self.api_model_name = base_models
        if "DeepSeek-V3.2" in base_models:
            tokenizer_model_name = "deepseek-ai/DeepSeek-V3.2"
        elif "qwen3-235b-a22b-instruct-2507" in base_models.lower():
            tokenizer_model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
        elif "qwen3-30b-a3b-instruct-2507" in base_models.lower():
            tokenizer_model_name = "../Qwen3-30B-A3B-Instruct-2507"
        elif "glm-4.7" in base_models.lower():
            tokenizer_model_name = "zai-org/GLM-4.7"
        else:
            tokenizer_model_name = self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        return

    @retry(wait=wait_random_exponential(min=0.1, max=1), stop=stop_after_attempt(10))
    def completion_with_backoff(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)

    def remove_input_tokens(self, output_tokens, ids):
        # Truncate the larger tensor to match the size of the smaller one
        min_size = min(output_tokens.size(1), ids.size(1))
        truncated_output_tokens = output_tokens[:, :min_size]
        truncated_ids = ids[:, :min_size]

        # Element-wise comparison and cumulative product to count length of common prefix
        common_prefix = (
            (truncated_output_tokens == truncated_ids).cumprod(dim=0).sum().item()
        )

        return output_tokens[:, common_prefix:]

    def _get_parameters(self):
        params = {}
        params["temperature"] = 0.1
        params["top_p"] = 0.7
        params["extra_body"] = {"top_k": 50}  
        self.params = params
    
    def get_parameters(self):
        if hasattr(self, "params") and not (self.params is None):
            return self.params
        
        self._get_parameters()
        return self.params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]]=[],
        **kwargs,
    ) -> str:
        self.probabilities = None
        params = self.get_parameters()

        if self.use_api:
            model_name = self.model_name
            messages = split_tagged_chat_to_messages(prompt, self.tags)
            stop_current = list(set(STOP_WORDS + stop))
            if self.is_baidu:
                stop_current = stop_current[:4]
            if self.is_reason:
                stop_current = []
            output = ""
            response = self.model.chat.completions.create(
                model=self.api_model_name,
                messages=messages,
                stop=stop_current,
                reasoning_effort="low",
                **params,
            )
            choice0 = response.choices[0]
            msg0 = choice0.message
            output = msg0.content

            if output is None:
                raise RuntimeError(
                    "API returned message.content=None. "
                    f"choices[0]={choice0!r}"
                )

            output = output.strip()

            if output.startswith("Thought:"):
                output = output[len("Thought:"):].strip()

        # Remove observations strings from output if generated
        for stop_word in STOP_WORDS + stop:
            output = output.replace(stop_word, "")

        return output.strip()

    def _call_batch(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        
        if stop is None:
            stop = []
        merged_stop = list(set(STOP_WORDS + stop))
        if self.is_baidu:
            merged_stop = merged_stop[:4]
        if self.is_reason:
            merged_stop = []

        outs = []
        for p in prompts:
            outs.append(
                self._call(
                    p,
                    stop=stop,
                )
            )
        return outs

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
        }
