import argparse
from argparse import Namespace
from datetime import datetime
from pathlib import Path


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


class Parser(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse(self) -> Namespace:
        parser_ = argparse.ArgumentParser()
        parser_.add_argument('-bs', '--batch', default=1, type=int, help='batch size')
        parser_.add_argument('-e', '--epoch', default=1000, type=int, help='epoch count')
        parser_.add_argument('-lr', '--learning_rate', default=1e-4, type=int, help='learning rate')
        parser_.add_argument('-name', '--model_name', default="check_model", type=str, help='name of the model')
        parser_.add_argument('-w', '--windows_len', default=1, type=int, help='time window length')
        args = parser_.parse_args()
        return args
