import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap



benchmarks = {
    'ntu': ('ntu/xview', 'ntu/xsub'),
    'ntu120': ('ntu120/xset', 'ntu120/xsub'),
    'kinetics': ('kinetics',)
}

sets = {
    'train', 'val'
}

parts = {'joint',
         'bone'}

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Motion data generation for NTU60/NTU120/Kinetics')
    parser.add_argument('--dataset', choices=['ntu', 'ntu120', 'kinetics'], required=True)
    args = parser.parse_args()

    for benchmark in benchmarks[args.dataset]:
        for set in sets:
            for part in parts:
                print(benchmark, set, part)
                data = np.load('../data/{}/{}_data_{}.npy'.format(benchmark, set, part))
                N, C, T, V, M = data.shape
                fp_sp = open_memmap(
                    '../data/{}/{}_data_{}_motion.npy'.format(benchmark, set, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 3, T, V, M))
                for t in tqdm(range(T - 1)):
                    fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
                fp_sp[:, :, T - 1, :, :] = 0

