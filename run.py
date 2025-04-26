from pathlib import Path
from hpman import HyperParameterManager
from hpman.m import hp
from ruamel.yaml import YAML
import os
from typing import Dict, Any
from collections.abc import Mapping
import sys

PWD = os.path.dirname(os.path.abspath(__file__))


def hp_tree_to_dict(d: Dict[str, Any]) -> Dict[str, Any]:

    def _walk(cur: str, tree: Dict[str, Any], res: Dict[str, Any]):
        if not isinstance(tree, Mapping):
            res[cur] = tree
            return

        for key, val in tree.items():
            new_key = "{}.{}".format(cur, key)
            _walk(new_key, val, res)

    res = {}
    for key, tree in d.items():
        _walk(key, tree, res)

    return res


def init_hyper_params(load_from: Path = None):
    mgr = hp.parse_file(PWD)  # type: HyperParameterManager
    defaults = mgr.get_values()

    yaml = YAML(typ='rt')
    if load_from:
        tree = yaml.load(load_from)
        hparams = hp_tree_to_dict(tree)
        defaults.update(hparams)

    mgr.set_values(defaults)
    return mgr


def do_train(args):
    init_hyper_params(args.runfile)

    try:
        import train
        train.main(args)
    except KeyboardInterrupt:
        sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    p = subparsers.add_parser('train')
    p.add_argument('--device', default='cuda', help='device to use for training / testing')
    p.add_argument('--output_dir', default='train_log', help='path where to save, empty for no saving')
    p.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
    p.add_argument('--eval', action='store_true', help='evaluate model on validation set')
    p.add_argument('--num_workers', default=1, type=int, help='number of data loading workers')
    p.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    p.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    p.add_argument('runfile', type=Path)
    p.set_defaults(func=do_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
