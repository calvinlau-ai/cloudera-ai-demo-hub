import os
import logging

logging.basicConfig(level=logging.INFO)
if '__file__' in vars():
    log = logging.getLogger(os.path.basename(__file__))
else:
    log = logging.getLogger('XXX')

prompt_template = '%s\n%s Please answer with no more than 150 words.'
