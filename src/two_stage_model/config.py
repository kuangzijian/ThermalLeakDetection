from configparser import ConfigParser
import ast

config_object = ConfigParser()
config_object.read("config.ini")

configs = config_object['INFERENCEPARAMETERS']

MODEL_ROOT = configs['model_root']
WEIGHTS = configs['weights']
H = configs.getint('input_height')
W = configs.getint('input_width')
IGNORED_AREAS = configs['ignored_areas']
IGNORED_AREAS = ast.literal_eval('[' + IGNORED_AREAS + ']')
