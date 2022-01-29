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


def run_one_instance(seed=0, dataset='credit', batch_size=32, epochs=200, fair_coeff=0.1, debug=False):
    args = generate_script_call_args(dataset=dataset,
                                     latent_dim=8,
                                     target_label_idx=0,
                                     mask_target_label=1,
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


def run_parallel_setting(dataset, batch_size, epochs, fair_coeff):
    # different seed, same fair coeff
    seeds = list(range(1, 11))
    Parallel(
        n_jobs=len(seeds),
        verbose=100,
    )(
        delayed(run_one_instance(s, dataset, batch_size, epochs, fair_coeff)) for s in seeds
    )


if __name__ == '__main__':

    for fair_coeff in [0.1, 1., 10., 100., 500., 1000., 5000]:
        run_parallel_setting('credit', 32, 200, fair_coeff)
        run_parallel_setting('adult', 128, 20, fair_coeff)
        
# python fairsoft_trial_postprocess.py -dataset credit -latent_dim 8 - target_label_idx 0 - mask_target_label 0 - seed $SLURM_ARRAY_TASK_ID - epoch 200 - bs 32 - fair_coeff 0.1
