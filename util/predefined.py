from warnings import warn, simplefilter
import os.path as osp
import sys

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from util.error import UnknownArgumentError  # noqa:402
from dataset.penn_treebank.extract_data import get_stats  # noqa:402


def predefined(opt):
    """
    Runs some checks over the settings to ensure that they don't clash, and sets predefined architectural setting.
    """
    # Enable warnings
    simplefilter('default')
    simplefilter('ignore', ResourceWarning)

    # When a legitimate cut_off is provided, we calculate the seq_len from dataset statistics and the
    # cut_off std provided.
    if opt.cut_off > 0.0:
        warn("The cut_off setting overrides seq_len. Sequence length is now determined by standard deviations.")
        opt = compute_seq_len(opt)

    if opt.tie_in_out == 1:
        warn("Cannot use sparse Embeddings when tying the input and output layer. Using dense embeddings instead.")
        opt.sparse = 0
        warn("h_dim and x_dim need to match when tying the input and output layer. Setting x_dim to h_dim.")
        opt.x_dim = opt.h_dim

    # When a penn treebank Preprocessing type is specified we select the correct data and out folders
    if opt.ptb_type:
        warn("Paths to data and out folder and v_dim are automatically set given the selected ptb_type: {}. User requested paths are ignored. To change this behavior, don't set any ptb_type.".format(opt.ptb_type))
        opt = set_ptb_folders(opt)

    # Multiple modes are joined into a single string
    if isinstance(opt.bayes_mode, list):
        opt.bayes_mode = "_".join(opt.bayes_mode)

    # Everything below here will only be called when we specify it
    if not opt.pre_def:
        return opt
    else:
        warn("Predefined settings are used. Some user requested settings may be ignored. To change this behavior, set --pre_def=0 (the default).")

    # Personal preferences across all models
    opt.verbosity = 1  # We like to see some info
    opt.grad_check = 1  # Checking the gradients cannot harm
    opt.log_likelihood = 1  # We like to see some logs and some likes

    # We showed empirically that these settings work best for the deterministic model
    opt.lr = 1e-3
    opt.clip = 1.5  # Gradient clipping helps for robust optimization
    opt.tie_in_out = 1
    opt.sparse = 0

    # We also fix some settings in the variational model
    opt.enc_h_dim = 256
    opt.enc_layers = 2
    opt.z_dim = 32

    # The optimal settings found with bayesopt on the deterministic model
    opt.p = 0.4
    opt.layers = 2
    opt.h_dim = 256
    opt.x_dim = 256
    opt.cut_off = 0.03
    opt = compute_seq_len(opt)
    opt.rnn_type = "GRU"
    opt.drop_type = "shared"

    return opt


def set_ptb_folders(opt):
    """This function sets default paths given a ptb_type."""
    if opt.ptb_type == "dyer":
        opt.data_folder = osp.join(toplevel_path, "dataset/penn_treebank_dyer")
        opt.out_folder = osp.join(toplevel_path, "out/penn_treebank_dyer")
        opt.v_dim = 25643
    elif opt.ptb_type == "mik":
        opt.data_folder = osp.join(toplevel_path, "dataset/penn_treebank")
        opt.out_folder = osp.join(toplevel_path, "out/penn_treebank")
        opt.v_dim = 10002
    else:
        raise UnknownArgumentError("Unknown ptb_type {}. Please choose [mik, dyer]".format(opt.ptb_type))
    return opt


def compute_seq_len(opt):
    """Computes the sequence length for a given cut-off."""
    mean, std = get_stats(osp.join(opt.data_folder, opt.train_file), True)[:2]
    opt.seq_len = int(mean + opt.cut_off * std)
    opt.mean_len = mean
    opt.std_len = std
    return opt
