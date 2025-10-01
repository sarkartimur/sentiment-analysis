import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

class TransformersLlmOversampler:
    __MODEL = "/mnt/f/IdeaProjects/pretrained/Qwen3-4B-Instruct-2507-FP8"
    __PROMPT = "Перефразируй следующий шаблон сообщения whatsapp:\n{}.\n Сохраняй тон сообщения (если сообщение формальное то и вывод должен быть формальным). Не используй имена людей и названия компаний, ссылки заменяй на рандомные. Если в тексте есть эмодзи (например :flexed_biceps:) используй их в ответе, но в текстовом формате (например :flexed_biceps:). Ответ не должен содержать ничего кроме перефразированного текста и превышать 500 слов."

    def __init__(self):
        print("\nTransformersLlmOversampler initializing...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.__MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.__MODEL,
            dtype="auto",
            local_files_only=True,
            device_map="auto"
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=2000,
            temperature=0.8,
            do_sample=True,
            top_p=0.9, 
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            # trust_remote_code=True
        )
        print("TransformersLlmOversampler initialized")

    def generate(self, template):
        prompt_content = self.__PROMPT.format(template)
        messages = [{"role": "user", "content": prompt_content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        output = self.pipe(prompt)[0]
        text = output['generated_text'][len(prompt):].strip()
        return self.__parse_message(text)


    def __parse_message(self, message):
        if m := re.match(r"<think>\n(.+)</think>\n\n", message, flags=re.DOTALL):
            return message[len(m.group(0)):].strip()
        else:
            return message