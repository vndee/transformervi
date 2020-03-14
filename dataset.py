import pandas as pd
from torchtext import data
from utils import get_logger

logger = get_logger(__name__)


def load_data(opts):
    src_train, src_test = None, None
    tgt_train, tgt_test = None, None

    try:
        src_train = open(opts.source_train).read().strip().split('\n')
        src_test = open(opts.source_test).read().strip().split('\n')
        tgt_train = open(opts.target_train).read().strip().split('\n')
        tgt_test = open(opts.target_test).read().strip().split('\n')
    except Exception as ex:
        logger.exception(ex)
        quit(0)

    _src = data.Field(lower=True, init_token='<sos>', eos_token='<eos>')
    _tgt = data.Field(lower=True)



    return src_train, src_test, tgt_train, tgt_test



