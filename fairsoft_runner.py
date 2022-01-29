from functools import reduce
import subprocess
import sys
import os
import argparse
import itertools
from joblib import Parallel, delayed

import torch

my_dir = os.path.dirname(__file__)


parser = argparse.ArgumentParser()
parser.add_argument('-mask_target_label', type=int, default=0)
parser.add_argument('-cuda', type=int, default=0)


def generate_script_call_args(**kwargs):
    return [
        sys.executable,
        os.path.join(my_dir, 'fairsoft_trial.py')
    ] + list(itertools.chain(*[(f'-{k}', str(v)) for k, v in kwargs.items()]))


def submit_one_instance(dataset='credit', mask_target_label=1, batch_size=32, epochs=200, fair_coeff=0.1, debug=False):
    def run_one_instance(seed):
        args = generate_script_call_args(dataset=dataset,
                                         latent_dim=8,
                                         target_label_idx=0,
                                         mask_target_label=mask_target_label,
                                         seed=seed,
                                         epoch=epochs,
                                         bs=batch_size,
                                         fair_coeff=fair_coeff)
        if debug:
            kwargs = {
                'capture_output': False,
                'stderr': subprocess.STDOUT
            }
        else:
            kwargs = {
                'capture_output': False,
                'stdout': subprocess.DEVNULL,
                'stderr': subprocess.DEVNULL
            }
        subprocess.run(args, check=True, **kwargs)

    return run_one_instance


def run_parallel_setting(dataset, mask_target_label, batch_size, epochs, fair_coeff):
    # different seed, same fair coeff
    seeds = list(range(1, 11))
    Parallel(
        n_jobs=len(seeds),
        verbose=100,
    )(
        delayed(submit_one_instance(dataset, mask_target_label, batch_size, epochs, fair_coeff))(s) for s in seeds
    )


if __name__ == '__main__':

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    for fair_coeff in [0.1, 1., 10., 100., 500., 1000., 5000]:
        run_parallel_setting(
            dataset='credit',
            mask_target_label=args.mask_target_label,
            batch_size=32,
            epochs=200,
            fair_coeff=fair_coeff
        )
        run_parallel_setting(
            dataset='adult',
            mask_target_label=args.mask_target_label,
            batch_size=128,
            epochs=20,
            fair_coeff=fair_coeff
        )

# python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 1 -seed 1 -epoch 200 -bs 32 -fair_coeff 0.1
