# TBLR:
# TenserBoard visualization for the Latest Run

import os
import glob
from pathlib import Path
import subprocess

paths = sorted(Path('runs').iterdir(), key=os.path.getmtime)

if len(paths) == 0:
    print('No runs avaiable')
    exit()


latest_run = paths[-1]
print(f'Visualize the latest run: {latest_run} at http://kokintu:4000/ (Press CTRL+C to quit)')

FNULL = open(os.devnull, 'w')
subprocess.call(['tensorboard', '--logdir', f'{latest_run}', '--port', '4000'], stdout=FNULL, stderr=subprocess.STDOUT)
