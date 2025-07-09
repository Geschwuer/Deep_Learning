import os
import torch
import pandas as pd
import numpy as np
import os
import tabulate
import argparse
import re
import stat
from data import ChallengeDataset

def setUp():
        # locate the csv file in file system and read it
        csv_path = ''
        for root, _, files in os.walk('.'):
            for name in files:
                if name == 'data.csv':
                    csv_path = os.path.join(root, name)
        return pd.read_csv(csv_path, sep=';')

if __name__ == "main":
     csv = setUp()
     data = (ChallengeDataset(csv, 'train'))