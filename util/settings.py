"""A parser for command line settings."""
import sys
import os.path as osp
import argparse


def parse_arguments():
    """
    This function parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Training Parameters")

    # Determine the toplevel path to make our life easier
    toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))

    # General settings
    parser.add_argument('--grad_check', default=0, type=int,
                        help="Whether to check gradients during training.")
    parser.add_argument('--use_test_set', default=0, type=int,
                        help="Whether to use the test set when evaluating. Otherwise we use the validation set.")
    parser.add_argument('--script', default='generative', type=str,
                        help="Which script to run via main.py. Choose [generative, adversarial, bayesopt, grid].")
    parser.add_argument('--local_rank', default=0, type=int,
                        help="Device number of GPU, negative when CPU is used.")
    parser.add_argument('--verbosity', default=0, type=int,
                        help="Verbosity of prints. 0: silent, 1: normal, 2: debug.")
    parser.add_argument('--seed', default=None, type=int,
                        help="Random seed for determinstic weight initialization.")
    parser.add_argument('--resume', default=0, type=int,
                        help="Whether to resume from checkpoint.")
    parser.add_argument('--mode', default='train', type=str,
                        help="Whether to run training or testing script.")
    parser.add_argument('--save_suffix', default='', type=str,
                        help="Suffix to append when saving/loading model.")
    parser.add_argument('--criteria', default=["posterior"], type=str, nargs='+',
                        help='Select one or multiple early stopping criteria from [prior, posterior]')
    parser.add_argument('--load_criteria', default="", type=str,
                        help="Load a model stored under a certain criteria. Choose [prior, '']")
    parser.add_argument('--pre_def', default=0, type=int,
                        help="Whether to use default settings and other predefined tricks.")

    # Data settings
    parser.add_argument('--pad_token', default='<pad>', type=str,
                        help="Pad token in the dataset.")
    parser.add_argument('--eos_token', default='</s>', type=str,
                        help="EOS token in the dataset.")
    parser.add_argument('--sos_token', default='<s>', type=str,
                        help="SOS token in the dataset.")
    parser.add_argument('--unk_token', default='<unk>', type=str,
                        help="UNK token in the dataset.")

    # Generative training settings
    parser.add_argument('--min_imp', default=0., type=float,
                        help="Below this threshold an epoch is counted as convergence tick.")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Number of datapoints per minibatch.")
    parser.add_argument('--model', default='bowman', type=str,
                        help="Which type of model to perform training on. Choose [deterministic, bowman, flowbowman]")
    parser.add_argument('--stop_ticks', default=6, type=int,
                        help="Number of consecutive epochs the validation loss increases before stopping.")

    # General optimization settings
    parser.add_argument('--tie_in_out', default=0, type=int,
                        help="Whether to tie the input and output layer weights of the language model.")
    parser.add_argument('--drop_type', default="shared", type=str,
                        help='Which type of dropout to apply. Choose [varied, shared, recurrent].')
    parser.add_argument('--p', default=0., type=float,
                        help="Dropout probability.")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Learning rate of the optimizer.")
    parser.add_argument('--clip', default=0.0, type=float,
                        help="Max norm of gradients, clips larger gradients. When 0 clipping will not be applied.")
    parser.add_argument('--cut_off', default=-1.0, type=float,
                        help="Number of standard deviations above mean allowed for sequence length.")
    parser.add_argument('--ann_word', default=0, type=int,
                        help="Whether to anneal word dropout.")

    # VAE optimization settings
    parser.add_argument('--word_p', default=0.0, type=float,
                        help="Fraction of words to drop before decoding.")
    parser.add_argument('--kl_step', default=1.0, type=float,
                        help="Step size of KL weight increment per minibatch.")
    parser.add_argument('--beta', default=1.0, type=float,
                        help="Weight of the KL term in the ELBO.")
    parser.add_argument('--lamb', default=0.0, type=float,
                        help="Weight of the MMD loss.")
    parser.add_argument('--mmd', default=0, type=int,
                        help="Whether to use the MMD between prior and posterio samples as term in the loss.")
    parser.add_argument('--ann_mode', default="linear", type=str,
                        help="Which KL annealing scheme to use. Choose [linear, sfb].")
    parser.add_argument('--rate_mode', default="hinge", type=str,
                        help="Which minimum rate guarantee mode to use. Choose [hinge, fb].")
    parser.add_argument('--posterior', default="gaussian", type=str,
                        help="Which posterior/encoder to use. Choose [gaussian, vmf].")
    parser.add_argument('--hinge_weight', default=0., type=float,
                        help="Weight of the hinge loss penalty.")
    parser.add_argument('--k', default=0., type=float,
                        help='Fixed value of kappa for vMF-VAE when larger than zero, is fraction of z_dim.')
    parser.add_argument('--min_rate', default=0.0, type=float,
                        help="Minimum rate of information we wish to encode in the latent space.")
    parser.add_argument('--max_mmd', default=1.0, type=float,
                        help="Maximum mmd between prior and posterior samples.")
    parser.add_argument('--num_rate_check', default=0.5, type=float,
                        help="Number of batches after which to update the min_rate. Choose non-integer to turn off.")
    parser.add_argument('--rate_increment', default=1.0, type=float,
                        help="Increment to the best KL with adaptive rate.")
    parser.add_argument('--warm_up_rate', default=5000, type=int,
                        help="Warm up before updating the rate loss.")
    parser.add_argument('--word_step', default=0, type=float,
                        help="Step size of word dropout annealing.")
    parser.add_argument('--lagrangian', default=0, type=int,
                        help='Whether to use Lagrangian optimisation scheme.')
    parser.add_argument('--constraint', default='mdr', type=str, nargs='+',
                        help='Which constraints to use for Lagrangian optimisation.')

    # Path settings
    parser.add_argument('--train_file', default='train.indices', type=str,
                        help='File with training data.')
    parser.add_argument('--val_file', default='val.indices', type=str,
                        help='File with validation data.')
    parser.add_argument('--test_file', default='test.indices', type=str,
                        help='File with test data.')
    parser.add_argument('--word_dict', default='word_to_idx.pickle', type=str,
                        help='File contaning the word to index dictionary.')
    parser.add_argument('--idx_dict', default='idx_to_word.pickle', type=str,
                        help='File contaning the index to word dictionary.')
    parser.add_argument('--data_folder', default=osp.join(toplevel_path, 'dataset/penn_treebank/'), type=str,
                        help="Folder that contains the data file.")
    parser.add_argument('--out_folder', default=osp.join(toplevel_path, 'out/penn_treebank/'), type=str,
                        help="Folder that contains output files.")
    parser.add_argument('--ptb_type', default="", type=str,
                        help="Preprocessing type of PTB. Choose [ , dyer, full].")
    parser.add_argument('--qual_file', default='', type=str,
                        help="File to read qualitative task from.")

    # Evaluation settings
    parser.add_argument('--ter', default=0, type=int,
                        help="Whether to use TER as eval metric.")
    parser.add_argument('--num_samples', default=10, type=int,
                        help="Number of sentences to sample at once when generating.")
    parser.add_argument('--sample_softmax', default=0, type=int,
                        help='Whether to draw samples from the decoder softmax.')
    parser.add_argument('--sample_len', default=26, type=int,
                        help="Length of sampled sequences.")
    parser.add_argument('--log_likelihood', default=0, type=int,
                        help="Whether to estimate log likelihood when testing")
    parser.add_argument('--ll_samples', default=1024, type=int,
                        help="Number of importance samples to use per test points to estimate the log_likelihood.")
    parser.add_argument('--ll_batch', default=128, type=int,
                        help="Batch size when estimating the log likelihood.")
    parser.add_argument('--delta', default=0.01, type=float,
                        help="Threshold for active units.")
    parser.add_argument('--mi', default=0, type=int,
                        help="Whether to compute mutual information.")
    parser.add_argument('--mi_method', default='zhao', type=str,
                        help="Which MI estimation to use. Choose [zhao, hoffman]")
    parser.add_argument('--mi_kde_method', default='pytorch', type=str,
                        help="Which KDE estimation to use. Choose [scipy, pytorch]")

    # Model architecture settings #####################################################################################
    # General
    parser.add_argument('--rnn_type', default="GRU", type=str,
                        help="Which type of RNN to use in the recurrent models. Choose [GRU, LSTM].")
    parser.add_argument('--seq_len', default=56, type=int,
                        help="Length of data sequences.")
    parser.add_argument('--mean_len', default=25.8, type=float,
                        help="Mean length of data sequences.")
    parser.add_argument('--std_len', default=12.1, type=float,
                        help="Std of data sequence length.")
    parser.add_argument('--sparse', default=0, type=int,
                        help="Whether to use sparse Embeddings.")
    parser.add_argument('--v_dim', default=10002, type=int,
                        help="Size of the vocabulary.")
    parser.add_argument('--x_dim', default=256, type=int,
                        help="Size of the embeddings.")
    parser.add_argument('--h_dim', default=256, type=int,
                        help="Size of the hidden states.")
    parser.add_argument('--s_dim', default=250, type=int,
                        help="Size of the negative set when using CSS.")
    parser.add_argument('--layers', default=1, type=int,
                        help="Number of layers in RNN.")
    parser.add_argument('--css', default=0, type=int,
                        help="Whether to use the CSS softmax approximate.")

    # VAE
    parser.add_argument('--z_dim', default=32, type=int,
                        help="Size of the latent variables.")
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoder layers.")
    parser.add_argument('--enc_h_dim', default=256, type=int,
                        help="Size of the encoder hidden layers.")
    parser.add_argument('--enc_p', default=0., type=float,
                        help="Parameter dropout rate in the encoder.")
    parser.add_argument('--enc_word_p', default=0., type=float,
                        help="Word dropout in the encoder.")

    # Normalizing flow
    parser.add_argument('--flow', default='diag', type=str,
                        help="Type of flow. Choose [diag, iaf, iafrnn, sylvester, cclin, ccvp]")
    parser.add_argument('--flow_depth', default=4, type=int,
                        help="Depth of normalizing flow.")
    parser.add_argument('--h_depth', default=1, type=int,
                        help="Number of hidden layers per flow step.")
    parser.add_argument('--prior', default="weak", type=str,
                        help="Which prior to use. Choose [weak, mog, vamp]")
    parser.add_argument('--num_weights', default=100, type=int,
                        help="Number of gaussians in MoG prior.")
    parser.add_argument('--c_dim', default=512, type=int,
                        help="Size of the context vector and flow layers.")
    ###################################################################################################################

    # Bayesian Optimization settings
    parser.add_argument('--bayes_load', default=0, type=int,
                        help="Whether to load stored results from previous bayesian optimization as initial values to seed the optimizer.")
    parser.add_argument('--bayes_iter', default=10, type=int,
                        help="Number of iterations of bayesian optimization.")
    parser.add_argument('--bayes_mode', default="", type=str, nargs='+',
                        help="Predefined templates for bayesian optimization.")
    parser.add_argument('--bayes_free_bits', default=0., type=float,
                        help="Minimum rate of information wanted when using bayesian optimization.")
    parser.add_argument('--two_test', default=0, type=int,
                        help="Whether to use both validation and test set during bayesopt/grid search.")
    parser.add_argument('--start_iter', default=0, type=int,
                        help="Starting iter number for grid/bayes search.")

    # Retesting and training settings
    parser.add_argument('--retest', default="", type=str,
                        help="Whether to retest in the bayesopt script. Choose [test, novelty, qualitative]")
    parser.add_argument('--reruns', default=0, type=int, nargs='+',
                        help="List of runs to retest.")

    return parser.parse_args(), parser
