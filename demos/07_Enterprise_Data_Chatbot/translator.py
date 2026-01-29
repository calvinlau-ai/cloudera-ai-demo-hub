import os
# from threading import Thread
import fasttext
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import log


class Translator:
    def __init__(self, lang):
        model_path_from_en = os.getenv('FROM_EN_MODEL') % lang
        model_path_to_en = os.getenv('TO_EN_MODEL') % lang
        log.info('Loading ' + model_path_from_en)
        self.tokenizer_from_en = AutoTokenizer.from_pretrained(model_path_from_en)
        self.model_from_en = AutoModelForSeq2SeqLM.from_pretrained(model_path_from_en)
        log.info('Loading ' + model_path_to_en)
        self.tokenizer_to_en = AutoTokenizer.from_pretrained(model_path_to_en)
        self.model_to_en = AutoModelForSeq2SeqLM.from_pretrained(model_path_to_en)

    def from_english(self, text):
        # input_ids = self.tokenizer_from_en([text], return_tensors="pt", truncation=True)
        # streamer = TextIteratorStreamer(self.tokenizer_from_en, skip_prompt=True)
        # generation_kwargs = dict(
        #     input_ids,
        #     streamer=streamer,
        #     max_new_tokens=500,
        #     do_sample=False,
        #     num_beams=1,
        #     # top_p=0.9,
        #     # temperature=0.1
        # )
        # thread = Thread(target=self.model_from_en.generate, kwargs=generation_kwargs)
        # thread.start()
        # for outp in streamer:
        #     yield outp
        input_ids = self.tokenizer_from_en(text, return_tensors="pt", truncation=True).input_ids
        output = self.model_from_en.generate(input_ids)
        return self.tokenizer_from_en.decode(output[0], skip_special_tokens=True)

    def to_english(self, text):
        input_ids = self.tokenizer_to_en(text, return_tensors="pt", truncation=True).input_ids
        output = self.model_to_en.generate(input_ids)
        return self.tokenizer_to_en.decode(output[0], skip_special_tokens=True)


class TranslatorV2:
    model_path_to_en = os.getenv('TO_EN_MODEL_V2')
    log.info('Loading ' + model_path_to_en)
    tokenizer_to_en = MBart50TokenizerFast.from_pretrained(model_path_to_en)
    model_to_en = MBartForConditionalGeneration.from_pretrained(model_path_to_en)

    def __init__(self, lang):
        self.lang = lang
        model_path_from_en = os.getenv('FROM_EN_MODEL') % lang
        log.info('Loading ' + model_path_from_en)
        self.tokenizer_from_en = AutoTokenizer.from_pretrained(model_path_from_en)
        self.model_from_en = AutoModelForSeq2SeqLM.from_pretrained(model_path_from_en)

    def from_english(self, text):
        input_ids = self.tokenizer_from_en(text, return_tensors="pt", truncation=True).input_ids
        output = self.model_from_en.generate(input_ids)
        return self.tokenizer_from_en.decode(output[0], skip_special_tokens=True)

    def to_english(self, text):
        tokenizer = TranslatorV2.tokenizer_to_en
        tokenizer.src_lang = langs[self.lang]
        input_ids = tokenizer(text, return_tensors="pt")
        output = TranslatorV2.model_to_en.generate(
            **input_ids,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
        )
        return tokenizer.batch_decode(output, skip_special_tokens=True)[0]


langs = {
    'en': 'en_US',
    'zh': 'zh_CN',
    'id': 'id_ID',
    # 'jp',
}
translators = dict([(lang, TranslatorV2(lang)) for lang in langs.keys() if lang != 'en'])
ft_model = fasttext.load_model('lid.176.ftz')


def translate_from(text):
    lang_preds = ft_model.predict(text, k=3)[0]
    lang_preds = [lang[len('__label__'):] for lang in lang_preds if lang[len('__label__'):] in langs]
    if len(lang_preds) > 0:
        lang = lang_preds[0]
        if lang == 'en':
            return 'en', text
        else:
            return lang, translators[lang].to_english(text)
    else:
        return None, None

def translate_to(lang, text):
    return translators[lang].from_english(text)
