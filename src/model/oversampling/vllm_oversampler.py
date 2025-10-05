import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Temporary fix for "AttributeError: 'NoneType' object has no attribute 'ensure_kv_transfer_shutdown'"
import os
os.environ["VLLM_USE_V1"] = "0"


class VllmOversampler:
    __SYSTEM_PROMPT = (
        "Your task is to rephrase movie reviews. "
        "Present {} options. "
        "IMPORTANT - each option should be on a separate line. Don't use line breaks within options (event if they contain lists). Output should contain only the options. "
        "Options should differ from each other but the semantics should be preserved. "
        "Option length should be equivalent to the original text length."
    )
    __USER_PROMPT = (
        "Rephrase the following movie review:\n{}\n\n"
    )
    __MAX_TOKENS = 2000
    __TEMPERATURE = 0.8
    __TOP_P = 0.9

    def __init__(self, model_path):
        print("\nVllmOversampler initializing...")        
        self.sampling_params = SamplingParams(
            temperature=self.__TEMPERATURE,
            top_p=self.__TOP_P,
            max_tokens=self.__MAX_TOKENS
        )        
        self.llm = LLM(
            model=model_path,
            dtype="auto",
            max_model_len=8192,
            # kv_cache_memory_bytes=9050494464,
            # enforce_eager=True,
            gpu_memory_utilization=0.98
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        print(f"VllmOversampler initialized")

    def generate(self, text: str, num_answers: int) -> str:
        messages = [
            {"role": "system", "content": self.__SYSTEM_PROMPT.format(num_answers)},
            {"role": "user", "content": self.__USER_PROMPT.format(text)}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        outputs = self.llm.generate([prompt], self.sampling_params)
        text = self.__parse_message(outputs[0].outputs[0].text.strip())
        return [line.strip() for line in text.splitlines() if line.strip()]

    def __parse_message(self, message: str) -> str:
        if m := re.match(r"^\s*<think>(.*?)</think>\s*", message, flags=re.DOTALL):
            return message[len(m.group(0)):].strip()
        else:
            return message