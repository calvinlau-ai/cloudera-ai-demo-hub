import os
import sys
import logging
import gradio as gr
import numpy as np
from dotenv import load_dotenv
log = logging.getLogger('mydata-chatbot-web')
load_dotenv(dotenv_path='mydata-chatbot.env')

from contriever_v2 import Retriever
from translator import translate_from, translate_to


def model_response(message, history):
    yield 'Preprocessing user query ...'
    log.info('Translating: %s', message)
    lang, message = translate_from(message)
    log.info('Identified language: %s', lang)
    log.info('Translated to English: %s', message)
    if lang is None:
        yield 'Sorry, I am unable to recognize your language.'
        return
    yield 'Retrieving enterprise data (lang=%s) ...' % lang
    texts = retriever.get_texts(message, top_k=3)
    print(texts)
    for id, text, score in texts:
        print(id, score, text.split('\n')[0])
        
    if len(texts) > 0:
      i = np.array([score for _, _, score in texts]).argmax()
      id, text, score = texts[i]

      yield 'Generating answer ...'
      output = ''
      for outp in model.gen_output(message, text):
          output += outp
          yield output
      output += ' (%s %s)' % (id, score)
      yield output
      if lang != 'en':
          yield output + "\n\nProcessing with language=%s ..." % lang
          trans_output = translate_to(lang, output)
          yield output + '\n\n' + trans_output
    else:
      yield 'Sorry, no information can be found in your local knowledge base.'


example_inputs = [
    'What is the revenue of Microsoft in game?',
    'What is the revenue of Microsoft?',
    'What do you think about Microsoft and Nvidia?',
    'What is the revenue of Cloudera?',
    'Do you have the revenue information of Cloudera in 2019?',
    'Who wins prime contract of the U.S. Navy?',
    '微软在游戏上的营收如何？',
    'Cloudera在2019年的收入情况如何？',
    'Berapa pendapatan Cloudera pada tahun 2019?'
]
retriever = Retriever()

match os.getenv("RUN_MODE"):
    case 'LOCAL_TEST':
        from metal_model import model
        for text in example_inputs:
            for outp in model_response(text, None):
                print(outp)
    case 'APPLE_WEB':
        from metal_model import model
        gr.ChatInterface(model_response, examples=example_inputs).launch(
            server_port=8080,
            show_error=True,
            enable_queue=True
        )
    case 'CML':
        from cuda_model import model
        gr.ChatInterface(model_response, examples=example_inputs).launch(
            server_port=int(os.getenv('CDSW_APP_PORT')),
            show_error=True,
            enable_queue=True
        )
    case _:
        print('Unknown RUN_MODE: '+os.getenv('run_mode'), file=sys.stderr)
        sys.exit(-1)

#list(model_response(example_inputs[0], None))
#list(model_response(example_inputs[1], None))
#list(model_response(example_inputs[5], None))
