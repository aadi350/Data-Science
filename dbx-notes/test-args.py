import sys
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('env', type=str)
args = parser.parse_known_args(sys.argv[1:])[0]

print(args, type(args))


YAML_STRING='''
key:
  option1: false
  option2: True
  option3: hello
'''


import yaml

print(yaml.safe_load(YAML_STRING))
