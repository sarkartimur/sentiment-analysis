import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Temporary fix for "AttributeError: 'NoneType' object has no attribute 'ensure_kv_transfer_shutdown'"
import os
os.environ["VLLM_USE_V1"] = "0"


class VllmOversampler:
    __MODEL = "/mnt/f/IdeaProjects/pretrained/Qwen3-8B-FP8"
    __SYSTEM_PROMPT = (
        "Твоя задача - производить переформулировки шаблонов сообщений whatsapp. "
        "Представь {} варианта(ов). "
        "ВАЖНО - каждый вариант должен быть на отдельной строке, в самих вариантах не используй переносы строк (даже если там есть списки), вывод не должен содержать ничего кроме вариантов, пронумеровывать варианты не надо. "
        "Варианты должны отличаться друг от друга, но семантика должна быть сохранена (семантически они должны быть похожи но не более чем на 90%). "
        "Если в сообщении содержится спам, навязчивый маркетинг, он должен быть и в выводе. "
        "По возможности, меняй структуру предложений. Заменяй не только отдельные слова на синонимы но и целые словосочетания. "
        "В вариантах не используй имена людей и названия компаний, ссылки заменяй на рандомные. "
        "Если в тексте есть эмодзи (например :flexed_biceps:) используй их в ответе, но только те, которые есть в оригинальном тексте и в том же количестве. "
        "Сохраняй тон сообщения (если сообщение формальное то и вывод должен быть формальным). "
        "Длина текста должна быть сравнима с длиной оригинального сообщения, но не превышать 200 слов."
    )
    __USER_PROMPT = (
        "Перефразируй следующий шаблон сообщения whatsapp:\n{}\n\n"
    )
    __MAX_TOKENS = 2000
    __TEMPERATURE = 0.8
    __TOP_P = 0.9

    def __init__(self):
        print("\nVllmOversampler initializing...")        
        self.sampling_params = SamplingParams(
            temperature=self.__TEMPERATURE,
            top_p=self.__TOP_P,
            max_tokens=self.__MAX_TOKENS
        )        
        self.llm = LLM(
            model=self.__MODEL,
            dtype="auto",
            max_model_len=8192,
            # kv_cache_memory_bytes=9050494464,
            # enforce_eager=True,
            gpu_memory_utilization=0.98
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.__MODEL)
        print(f"VllmOversampler initialized")

    def generate(self, template: str, num_answers: int) -> str:
        messages = [
            {"role": "system", "content": self.__SYSTEM_PROMPT.format(num_answers)},
            {"role": "user", "content": self.__USER_PROMPT.format(template)}
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