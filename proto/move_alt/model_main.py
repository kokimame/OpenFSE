import argparse
import json
from datetime import datetime

from model_evaluate import evaluate
from model_train import train

if __name__:
    with open('data/move_defaults.json') as f:
        defaults = json.load(f)

    parser = argparse.ArgumentParser(description='Training code of TA_MODEL')
    parser.add_argument('-dn',
                        '--dataset_name',
                        type=str,
                        default='',
                        help='Specifying a dataset for training and evaluation. ')
    args = parser.parse_args()
    save_name = '_'.join([args.dataset_name, *str(datetime.now()).split()])
    # lr_arg = '{}'.format(args.learning_rate).replace('.', '-')
    # margin_arg = '{}'.format(args.margin).replace('.', '-')

    if defaults['run_type'] == 'train':
        train(defaults, save_name, args.dataset_name)
    # else:
    #     evaluate(defaults, save_name, args.dataset_name)