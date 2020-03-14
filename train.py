import argparse
from utils import load_configs
from dataset import load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', required=True)
    args = parser.parse_args()
    opts = load_configs(args.config)

    src_train, src_test, tgt_train, tgt_test = load_data(opts)



