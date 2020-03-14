import sys
import logging
from yaml import load
from collections import namedtuple

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except:
    from yaml import Loader, Dumper


def load_configs(config):
    data = load(open(config), Loader=Loader)
    opts = namedtuple('Config', data.keys())(*data.values())
    return opts


def get_logger(logger_name='default'):
    """
    Get logging and format
    All logs will be saved into logs/log-DATE (default)
    Default size of log file = 15m
    :param logger_name:
    :return:
    """
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_format)
    log.addHandler(ch)

    return log