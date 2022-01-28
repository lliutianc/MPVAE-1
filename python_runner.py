from functools import reduce
import subprocess
import sys
import os
import itertools
from joblib import Parallel, delayed

my_dir = os.path.dirname(__file__)


def generate_script_call_args(**kwargs):
    return [
        sys.executable,
        os.path.join(my_dir, 'fairsoft_trial.py')
    ] + list(itertools.chain(*[(f'-{k}', str(v)) for k, v in kwargs.items()]))


def run_one_instance(seed=0, fair_coeff=0.1, debug=False):
    args = generate_script_call_args(dataset='credit',
                                     latent_dim=8,
                                     target_label_idx=0,
                                     mask_target_label=0,
                                     seed=seed,
                                     epoch=200,
                                     bs=32,
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


def run_parallel_setting_1():
    # different seed, same fair coeff
    seeds = list(range(10))
    Parallel(
        n_jobs=len(seeds),
        verbose=100,
    )(
        delayed(run_one_instance(s, 0.1)) for s in seeds
    )

if __name__ == '__main__':
    #run_one_instance(debug=False)
    run_parallel_setting_1()
# python fairsoft_trial_postprocess.py -dataset credit -latent_dim 8 - target_label_idx 0 - mask_target_label 0 - seed $SLURM_ARRAY_TASK_ID - epoch 200 - bs 32 - fair_coeff 0.1
