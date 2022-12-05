import logging
import logging.config
import yaml

with open('logging.yaml', 'r') as f:
    loggingconfig = yaml.load(f, Loader=yaml.FullLoader)
    
logging.config.dictConfig(loggingconfig)

logging.debug('test debug')
