"""Bayesian Optimization on model parameters using GPyOpt."""
import os
import os.path as osp
import sys
import pickle

import GPyOpt
import GPy
import numpy as np

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)


from scripts.generative import train, test, novelty, qualitative  # noqa: E402
from util.display import print_flags  # noqa: E402
from util.predefined import predefined  # noqa: E402
from util.error import UnknownArgumentError  # noqa: E402

__author__ = "Tom Pelsmaeker"
__copyright__ = "Copyright 2020"


class OptimizationFunction():
    """Wrapper around train and test scripts so it can be used as objective function for GPyOpt Bayesian Optimization."""

    def __init__(self, tuning_list, opt, parser, out_file, iter=0):
        super().__init__()
        self.tuning_list = tuning_list
        self.opt = opt
        self.parser = parser
        self.out_file = out_file
        self.iter = iter

        with open(self.out_file, 'w') as f:
            f.write("Starting Bayesian Optimisation...\n\n")

    def __call__(self, x):
        """This is the objective function to be minimized with Bayesian optimization.

        This function changes the training hyperparameters as provided by x and trains a model. This model
        is then evaluated. The selected  optimization criteria is returned by the evaluation function, and will be
        provided as scalar output.

        Args:
            x(np.array): 2-D array with a where every row is a list of options.

        Returns:
            np.array: 2-D array with the cost of the objective per row.
        """
        self.opt.script = "generative"

        y = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            # Overwrite default options with options provided by the bayesian optimization
            self.opt = self._set_options(self.opt, x, i)

            self.opt.mode = 'train'
            print_flags(self.opt)
            train(self.opt)
            self.opt.mode = 'test'
            y[i], stats = test(self.opt)
            self._store_stats(stats, vars(self.opt), y[i])
            if self.opt.two_test:
                self.opt.use_test_set = 1
                res, stats = test(self.opt)
                self._store_stats(stats, vars(self.opt), res)
                self.opt.use_test_set = 0
            self.iter += 1

        return y

    def re_test(self, run_ids):
        """Re-tests previous runs of BayesOpt."""
        self.opt.script = "generative"
        self.opt.mode = 'test'
        for run in run_ids:
            self.iter = run
            self.opt.save_suffix = "{}_{}".format(self.opt.bayes_mode, self.iter)
            y, stats = test(self.opt)
            self._store_stats(stats, vars(self.opt), y)
            if self.opt.two_test:
                self.opt.use_test_set = 1
                res, stats = test(self.opt)
                self._store_stats(stats, vars(self.opt), res)
                self.opt.use_test_set = 0

    def novelty(self, run_ids):
        """Runs the novelty script for previous runs of BayesOpt."""
        self.opt.script = "generative"
        self.opt.mode = 'novelty'
        for run in run_ids:
            self.iter = run
            self.opt.save_suffix = "{}_{}".format(self.opt.bayes_mode, self.iter)
            nov = novelty(self.opt)
            self._store_stats(nov, vars(self.opt), nov)

    def qualitative(self, run_ids):
        """Runs the qualitative script for previous runs of BayesOpt."""
        self.opt.script = "generative"
        self.opt.mode = 'qualitative'
        for run in run_ids:
            self.iter = run
            self.opt.save_suffix = "{}_{}".format(self.opt.bayes_mode, self.iter)
            qualitative(self.opt)

    def _store_stats(self, stats, opt_dict, y):
        """Writes training statistics to a file."""
        with open(self.out_file, 'a') as f:
            f.write("Run: {}\n".format(self.iter))
            f.write("Bayesopt Value: {}\n".format(y))
            f.write("Settings:\n")
            for param in self.tuning_list:
                f.write("\t{}: {}\n".format(param, opt_dict[param]))
            f.write("\tTest_set: {}\n".format(opt_dict["use_test_set"]))
            f.write("\nTest Statistics:\n{}\n".format(str(stats)))

    def _map_arguments(self, param, value):
        """Maps arguments from a float value to the required format.

        This function is required because the GpyOpt tools only search over float values. However, many
        arguments (settings) we may wish to search over have different types, e.g. str,
        so we require a deterministic mapping.
        This function covers all settings searched over (Grid or Bayesian) in the paper, but is non-exhaustive.
        Novel settings may have to be added.
        """
        if param in ["layers", "h_dim", "x_dim", "v_dim", "z_dim", "enc_h_dim", "enc_layers", "stop_ticks",
                     "flow_depth", "flow", "num_weights", "tie_in_out", "lagrangian"]:
            value = int(value)
        if param == "ann_mode":
            if value == 0.:
                value = "linear"
            elif value == 1.:
                value = "sfb"
        if param == "rate_mode":
            if value == 0.:
                value = "hinge"
            elif value == 1.:
                value = "fb"
        if param == "flow":
            if value == 0.:
                value = "diag"
            elif value == 1.:
                value = "iaf"
            elif value == 2.:
                value = "vpiaf"
            elif value == 3.:
                value = "planar"
        if param == "prior":
            if value == 0.:
                value = "weak"
            elif value == 1.:
                value = "mog"
            elif value == 2.:
                value = "vamp"
        if param == "constraint":
            if value == 0.:
                value = "mdr"
            elif value == 1.:
                value = "mmd"
            elif value == 2.:
                value = 'mmd mdr'
        if param == 'lamb':
            value = 10. ** value
        if param == 'kl_step':
            value = 10. ** (-value)
        if param == 'word_step':
            value = 10. ** (-value)
        if param == "hinge_weight":
            value = 10. ** value
        if param == "k":
            value = 3.5 ** value
        if param in ["p", "cut_off", 'kl_step', 'word_p', 'beta', 'lamb' 'min_rate', 'enc_p', 'k', 'hinge_weight',
                     "word_step", "enc_word_p", "max_elbo", "max_mmd", "alpha"]:
            # We don't want to get lost in small increments
            value = round(value, 5)

        return str(value)

    def _set_options(self, opt, x, i):
        """Sets the options in a Namespace object given a numpy array of numerical values that can be mapped to options.

        Args:
            opt(argparse.Namespace): options for the train and test functions.
            x(np.array): 2-D array where every row contains numerical values to be mapped to the opt Object.
            i(int): index of the row of x to extract options from.
        """
        opt.save_suffix = "{}_{}".format(opt.bayes_mode, self.iter)

        for j, param in enumerate(self.tuning_list):
            # The tuning list contains the parameter names, x contains the values as set by bayesopt, ordered
            self.parser.parse_args(args=['--'+param, self._map_arguments(param, x[i, j])], namespace=opt)
            if param == 'constraint':
                opt.constraint = opt.constraint[0].split(' ')

        if opt.tie_in_out:
            self.parser.parse_args(args=["--x_dim", self._map_arguments("x_dim",
                                                                        int(opt.h_dim/opt.layers))], namespace=opt)
        opt = predefined(opt)
        return opt


def get_parameters(opt):
    """Add desired parameter lists and initial search space to this function."""
    Y_init = None
    parameters = list()
    X_init = list()
    if "example" in opt.bayes_mode:
        # We perform Bayesian search over four parameters of the decoder (RNNLM) and specify some initial points
        # to test. Alternatively, one can specify no initial points and choose a 'grid' (eg latin) option
        # in GPyOpt to randomly select some starting points. See the GPyOpt docs for more information.
        parameters = [{'name': 'layers', 'type': 'discrete', 'domain': (1, 2)},
                      {'name': 'h_dim', 'type': 'discrete', 'domain': list(range(128, 513, 32))},
                      {'name': 'p', 'type': 'continuous', 'domain': (0., 0.6)},
                      {'name': 'cut_off', 'type': 'continuous', 'domain': (1, 5)}
                      ]
        X_init.append([1., 2., 1., 2.])  # Number of layers
        X_init.append([256., 128., 512., 256.])  # Number of hidden units
        X_init.append([0.2, 0.1, 0.3, 0.4])  # Dropout
        X_init.append([1., 2., 4., 3.])  # Number of standard deviations above mean sentence length to truncate
    elif "lagvae" in opt.bayes_mode:
        parameters = [{'name': 'max_elbo', 'type': 'continuous', 'domain': (100, 140)},
                      {'name': 'max_mmd', 'type': 'continuous', 'domain': (1, 20)},
                      {'name': 'alpha', 'type': 'continuous', 'domain': (-100, -1)}
                      ]
        X_init.append([100., 130., 110., 120.])  # Maximum -ELBO value
        X_init.append([5., 10., 7., 13.])  # Maximum MMD
        X_init.append([-10., -5., -25., -1.])  # Weight of information preference
    else:
        raise UnknownArgumentError(
            "Uknown bayes mode: {}. Please choose another or specify this one yourself.")
    X_init = np.array(X_init).T

    if opt.bayes_load:
        # Load previously stored results as initialization for Bayesian search
        X_init = pickle.load(open(osp.join(opt.out_folder, "bayesian",
                                           "bayesian_X_{}.pickle".format(opt.bayes_mode)), 'rb'))
        Y_init = pickle.load(open(osp.join(opt.out_folder, "bayesian",
                                           "bayesian_Y_{}.pickle".format(opt.bayes_mode)), 'rb'))
    print(X_init)

    tuning_list = [d['name'] for d in parameters]
    return parameters, tuning_list, X_init, Y_init


def optimize_bayesian(opt, parser):
    """
    Main script for Bayesian optimisation.

    In default mode, this script runs Bayesian optimisation for a given number of steps.
    Each optimisation is one full training run, each run having the same parameters except for the ones defined in
    get_parameters() with the bayes_mode argument. The optimisation will automatically find the best values for those
    parameters.

    The script can also be used to run other scripts (test, novelty, qualitative) on models previously trained.
    """
    parameters, tuning_list, X_init, Y_init = get_parameters(opt)

    if not osp.isdir(osp.join(opt.out_folder, "bayesian")):
        os.makedirs(osp.join(opt.out_folder, "bayesian"))

    if not opt.retest:
        custom_file = osp.join(opt.out_folder, "bayesian", "test_output_{}.txt".format(opt.bayes_mode))
        func = OptimizationFunction(tuning_list, opt, parser, custom_file, opt.start_iter)

        problem = GPyOpt.methods.BayesianOptimization(f=func,
                                                      domain=parameters,
                                                      model_type='GP_MCMC',
                                                      acquisition_type='EI_MCMC',
                                                      kernel=GPy.kern.Matern52(len(parameters), ARD=True),
                                                      verbosity=True,
                                                      verbosity_model=True,
                                                      X=X_init,
                                                      Y=Y_init,
                                                      batch_size=1,
                                                      num_cores=1,
                                                      )

        models_file = osp.join(opt.out_folder, "bayesian", "bayesian_models_{}.tex".format(opt.bayes_mode))
        report_file = osp.join(opt.out_folder, "bayesian", "bayesian_report_{}.tex".format(opt.bayes_mode))
        eval_file = osp.join(opt.out_folder, "bayesian", "bayesian_eval_{}.tex".format(opt.bayes_mode))

        problem.run_optimization(opt.bayes_iter, models_file=models_file,
                                 report_file=report_file, evaluations_file=eval_file)

        problem.plot_acquisition(osp.join(opt.out_folder, "bayesian", "aquisition_plot_{}.png".format(opt.bayes_mode)))
        problem.plot_convergence(osp.join(opt.out_folder, "bayesian",
                                          "convergence_plot_{}.png".format(opt.bayes_mode)))
        X, Y = problem.get_evaluations()
        pickle.dump(X, open(osp.join(opt.out_folder, "bayesian", "bayesian_X_{}.pickle".format(opt.bayes_mode)), 'wb'))
        pickle.dump(Y, open(osp.join(opt.out_folder, "bayesian", "bayesian_X_{}.pickle".format(opt.bayes_mode)), 'wb'))
    elif opt.retest == "test":
        custom_file = osp.join(opt.out_folder, "bayesian", "test_output_{}_retest.txt".format(opt.bayes_mode))
        func = OptimizationFunction(tuning_list, opt, parser, custom_file)
        func.re_test(opt.reruns)
    elif opt.retest == "novelty":
        custom_file = osp.join(opt.out_folder, "bayesian", "test_output_{}_novelty.txt".format(opt.bayes_mode))
        func = OptimizationFunction(tuning_list, opt, parser, custom_file)
        func.novelty(opt.reruns)
    elif opt.retest == "qualitative":
        custom_file = osp.join(opt.out_folder, "bayesian", "test_output_{}_qualitative.txt".format(opt.bayes_mode))
        func = OptimizationFunction(tuning_list, opt, parser, custom_file)
        func.qualitative(opt.reruns)
