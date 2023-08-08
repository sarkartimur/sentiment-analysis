import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger('matplotlib').setLevel(logging.INFO)