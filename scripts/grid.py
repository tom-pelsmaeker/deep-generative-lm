"""Gridsearch utility."""
import os.path as osp
import sys
from copy import deepcopy
from argparse import Namespace
import pickle
from random import shuffle

import numpy as np
import GPy
import GPyOpt

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from scripts.generative import train, test  # noqa: E402
from scripts.bayesopt import OptimizationFunction  # noqa: E402
from util.display import print_flags  # noqa: E402
from util.predefined import predefined  # noqa: E402
from util.error import UnknownArgumentError, InvalidArgumentError  # noqa: E402


class GridFunction(OptimizationFunction):
    """Wrapper around train and test scripts for gridsearch."""

    def __init__(self, tuning_list, opt, parser, out_file, iter=0):
        super(GridFunction, self).__init__(tuning_list, opt, parser, out_file, iter)

        with open(self.out_file, 'w') as f:
            f.write("Starting Grid Search...\n\n")

    def _store_stats(self, stats, opt_dict, y):
        with open(self.out_file, 'a') as f:
            f.write("Run: {}\n".format(self.iter))
            f.write("Grid Value: {}\n".format(y))
            f.write("Settings:\n")
            for param in self.tuning_list:
                f.write("\t{}: {}\n".format(param, opt_dict[param]))
            f.write("\nTest Statistics:\n{}\n".format(str(stats)))


def get_parameters(opt):
    """Add desired parameter lists and initial search space to this function."""
    parameters = list()
    X_init = list()
    if "mdr_example" in opt.bayes_mode:
        # Run grid search over a series of target rates
        parameters.append("min_rate")
        X_init.append([5., 10., 15., 20., 25., 30., 35., 40., 45., 50.])
    else:
        raise UnknownArgumentError(
            "Uknown bayes mode: {}. Please choose another or specify this one yourself.")

    X_init = np.array(X_init).T
    print(X_init)

    return parameters, X_init


def run_grid(opt, parser):
    parameters, X = get_parameters(opt)

    if not opt.retest:
        custom_file = osp.join(opt.out_folder, "grid", "test_output_{}.txt".format(opt.bayes_mode))
        func = GridFunction(parameters, opt, parser, custom_file, opt.start_iter)

        for i in range(X.shape[0]):
            _ = func(X[None, i, :])
    elif opt.retest == "test":
        custom_file = osp.join(opt.out_folder, "grid", "test_output_{}_retest.txt".format(opt.bayes_mode))
        func = GridFunction(parameters, opt, parser, custom_file, opt.start_iter)
        func.re_test(opt.reruns)
    elif opt.retest == "novelty":
        custom_file = osp.join(opt.out_folder, "grid", "test_output_{}_novelty.txt".format(opt.bayes_mode))
        func = GridFunction(parameters, opt, parser, custom_file, opt.start_iter)
        func.novelty(opt.reruns)
    elif opt.retest == "qualitative":
        custom_file = osp.join(opt.out_folder, "grid", "test_output_{}_qualitative.txt".format(opt.bayes_mode))
        func = GridFunction(parameters, opt, parser, custom_file, opt.start_iter)
        func.qualitative(opt.reruns)


if __name__ == '__main__':
    run_grid(None, None)
