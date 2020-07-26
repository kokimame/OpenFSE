import argparse
import json
import os
from datetime import datetime

from model_evaluate import evaluate
from model_train import train

if __name__:
    with open('data/model_defaults.json') as f:
        defaults = json.load(f)
        if defaults['dataset_root'] == '':
            if 'kmametani' in os.environ['HOME']:
                defaults['dataset_root'] = '/home/kmametani/Master_Files'
            else:
                defaults['dataset_root'] = '/media/kokimame/Work_A_1TB/Project/Master_Files'

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
    # Use this later
    # else:
    #     evaluate(defaults, save_name, args.dataset_name)