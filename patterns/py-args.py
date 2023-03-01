import argparse
import logging

logging.getLogger().setLevel(logging.DEBUG)

def main():

  parser = argparse.ArgumentParser(description='argparse desc')
  parser.add_argument('arg1', type=str)
  parser.add_argument('arg2', type=str)
  parser.add_argument('arg3', type=str)

  args = parser.parse_args()
  arg1 = args.arg1
  arg2 = args.arg2
  arg3 = args.arg3

  logging.debug((arg1, arg2, arg3))

if __name__ == '__main__':
  main()
