from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

class TransformersLlmOversampler:
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

    def __init__(self, model_path):
        print("\nTransformersLlmOversampler initializing...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
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

    def generate(self, text: str, num_answers: int) -> str:
        messages = [
            {"role": "system", "content": self.__SYSTEM_PROMPT.format(num_answers)},
            {"role": "user", "content": self.__USER_PROMPT.format(text)}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        output = self.pipe(prompt)[0]
        text = output['generated_text'][len(prompt):].strip()
        return self.__parse_message(text)


    def __parse_message(self, message):
        if m := re.match(r"^\s*<think>(.*?)</think>\s*", message, flags=re.DOTALL):
            return message[len(m.group(0)):].strip()
        else:
            return message