from fairsoft_evaluate import evaluate_target_labels
import torch

from faircluster_train import parser
from utils import build_path

parser.add_argument('-min_support', type=float, default=None)
parser.add_argument('-min_confidence', type=float, default=None)
parser.add_argument('-dist_gamma', type=float, default=1.0)
parser.add_argument('-target_label_idx', type=int, default=0)


def train_fairsoft_arule(args):
    from fairsoft_arule import train_fair_through_regularize

    param_setting = f"arule_{args.target_label_idx}"
    args.model_dir = f"fair_through_distance/model/{args.dataset}/{param_setting}"
    args.summary_dir = f"fair_through_distance/summary/{args.dataset}/{param_setting}"
    build_path(args.model_dir, args.summary_dir)

    train_fair_through_regularize(args)


def train_fairsoft_baseline(args):
    from fairsoft_baseline import train_fair_through_regularize

    param_setting = f"baseline_{args.target_label_idx}" if args.penalize_unfair else f"unfair"
    args.model_dir = f"fair_through_distance/model/{args.dataset}/{param_setting}"
    args.summary_dir = f"fair_through_distance/summary/{args.dataset}/{param_setting}"
    build_path(args.model_dir, args.summary_dir)

    train_fair_through_regularize(args)


def eval_fairsoft_allmodels(args):
    from fairsoft_evaluate import evaluate_target_labels

    args.model_dir = f'fair_through_distance/model/{args.dataset}'

    evaluate_target_labels(args)


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device(
    f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # train unfair model
    args.penalize_unfair = 0
    train_fairsoft_baseline(args)

    args.penalize_unfair = 1
    for target_label_idx in [0, 10, 20, 50]:
        args.target_label_idx = target_label_idx
        train_fairsoft_arule(args)
        train_fairsoft_baseline(args)
        evaluate_target_labels(args)
        
        break

    
# python fairsoft_trial.py -dataset adult -latent_dim 8 -cuda 5
