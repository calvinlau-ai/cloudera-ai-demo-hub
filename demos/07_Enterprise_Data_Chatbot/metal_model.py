import os
from llama_cpp import Llama
from utils import log, prompt_template


class MetalModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        log.info('Loading model %s ...' % self.model_path)
        llm = Llama(
            model_path=self.model_path,
            # chat_format="llama-2",
            # n_threads=4,
            n_gpu_layers=-1,  # Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
            n_ctx=8000,
            n_batch=126,
            seed=42,
            verbose=False
        )
        # llm.verbose = False
        log.info('Model loaded.')
        return llm

    def gen_output(self, input, text):
        log.debug('Begin: gen_output')
        prompt = prompt_template % (text, input)
        stream = self.model(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,  # 0.5,
            top_p=0.95,
            echo=False,
            stream=True
        )
        # truncated = False
        for output in stream:
            outp = output['choices'][0]['text']
            yield outp
        log.debug('End: gen_output')

    def __del__(self):
        del self.model


model = MetalModel(os.getenv('CHAT_MODEL'))
