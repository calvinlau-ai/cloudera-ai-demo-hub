import os
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from utils import log, prompt_template
torch.manual_seed(42)


class EosLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.stop_rate = None
        self.newline_id = 13
        self.eos_id = 2

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores[0].argmax().item() != self.newline_id and self.stop_rate is None:
            self.stop_rate = 1.0
        if self.stop_rate is not None:
            if scores[0].argmax().item() == self.newline_id:
                self.stop_rate *= 1.1
            scores[:, self.eos_id] = scores[:, self.eos_id] * self.stop_rate

        return scores


class CudaModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model, self.tokenizer = self.load_model()
        self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(EosLogitsProcessor())

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, model_max_length=8192)
        tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print('Loading model ...')
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            use_cache=True,
            # do_sample=True,
            device_map='auto'
        )
        return model, tokenizer

    def gen_output(self, input, text):
        prompt = prompt_template % (text, input)
        log.info('Tokenizing ...')
        input_ids = self.tokenizer([prompt], return_tensors="pt", truncation=True).to('cuda')
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            input_ids,
            streamer=streamer,
            logits_processor=self.logits_processor,
            max_new_tokens=500,
            do_sample=False,
            # top_p=0.9,
            # temperature=0.1
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for outp in streamer:
            yield outp


model = CudaModel(os.getenv('CHAT_MODEL'))
