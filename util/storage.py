"""Functions that aid storage and loading of models and data."""
from torch.utils.data import DataLoader
import torch
import pickle
import random
import sys
import os
osp = os.path

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from dataset.text import TextDataUnPadded  # noqa: E402
from model.bowman_decoder import BowmanDecoder, FlowBowmanDecoder  # noqa: E402
from model.deterministic_decoder import DeterministicDecoder  # noqa: E402
from util.error import UnknownArgumentError, InvalidPathError, Error  # noqa: E402

__author__ = "Tom Pelsmaeker"
__copyright__ = "Copyright 2020"


def seed(opt):
    """Applies user determined seed for determistic reproducable training/testing."""
    if opt.seed is not None:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)


def load_word_index_maps(opt):
    """Loads word to index and index to word mapping given user specified (path) settings."""
    return pickle.load(open(osp.join(opt.data_folder, opt.word_dict), 'rb')), \
        pickle.load(open(osp.join(opt.data_folder, opt.idx_dict), 'rb'))


def initialize_model(opt, word_to_idx):
    """Initializes model with the given user settings."""
    if opt.model == "bowman":
        decoder = BowmanDecoder(opt.device, opt.seq_len, opt.kl_step, opt.word_p, opt.enc_word_p, opt.p, opt.enc_p,
                                opt.drop_type, opt.min_rate, word_to_idx[opt.unk_token], opt.css, opt.sparse, opt.N,
                                opt.rnn_type, opt.tie_in_out, opt.beta, opt.lamb, opt.mmd, opt.ann_mode,
                                opt.rate_mode, opt.posterior, opt.hinge_weight, opt.k, opt.ann_word, opt.word_step,
                                opt.v_dim, opt.x_dim, opt.h_dim, opt.z_dim, opt.s_dim, opt.layers, opt.enc_h_dim,
                                opt.enc_layers, opt.lagrangian, opt.constraint, opt.max_mmd, opt.max_elbo,
                                opt.alpha).to(opt.device)
    elif opt.model == "flowbowman":
        decoder = FlowBowmanDecoder(opt.device, opt.seq_len, opt.kl_step, opt.word_p, opt.enc_word_p, opt.p, opt.enc_p,
                                    opt.drop_type, opt.min_rate, word_to_idx[opt.unk_token], opt.css, opt.sparse, opt.N,
                                    opt.rnn_type, opt.tie_in_out, opt.beta, opt.lamb, opt.mmd, opt.ann_mode,
                                    opt.rate_mode, opt.posterior, opt.hinge_weight, opt.k, opt.ann_word, opt.word_step,
                                    opt.flow, opt.flow_depth, opt.h_depth, opt.prior, opt.num_weights, opt.mean_len,
                                    opt.std_len, opt.v_dim, opt.x_dim, opt.h_dim, opt.z_dim, opt.s_dim, opt.layers,
                                    opt.enc_h_dim, opt.enc_layers, opt.c_dim, opt.lagrangian, opt.constraint,
                                    opt.max_mmd, opt.max_elbo, opt.alpha).to(opt.device)
    elif opt.model == "deterministic":
        decoder = DeterministicDecoder(opt.device, opt.seq_len, opt.word_p, opt.p, opt.drop_type,
                                       word_to_idx[opt.unk_token], opt.css, opt.sparse, opt.N, opt.rnn_type,
                                       opt.tie_in_out, opt.v_dim, opt.x_dim, opt.h_dim, opt.s_dim,
                                       opt.layers).to(opt.device)
    else:
        raise UnknownArgumentError(
            "--model not recognized, please choose: [deterministic, bowman, flowbowman].")

    return decoder


def initialize_dataloader(opt, word_to_idx, collate_fn):
    """Initializes the dataloader with the given user settings and collate function."""
    if opt.mode in ['train', 'qualitative']:
        data_train = DataLoader(TextDataUnPadded(get_true_data_path(opt, "train")[1], opt.seq_len,
                                                 word_to_idx[opt.pad_token]), collate_fn=collate_fn,
                                batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        data_eval = DataLoader(TextDataUnPadded(get_true_data_path(opt, "valid")[1], 0, word_to_idx[opt.pad_token]),
                               collate_fn=collate_fn, batch_size=opt.batch_size, shuffle=True, num_workers=4,
                               pin_memory=True)
        return data_train, data_eval
    elif opt.mode == 'test':
        # We use the PTB test set only sparsly
        if opt.use_test_set:
            return DataLoader(TextDataUnPadded(get_true_data_path(opt, "test")[1], 0, word_to_idx[opt.pad_token]),
                              collate_fn=collate_fn, batch_size=opt.batch_size, shuffle=False, num_workers=4,
                              pin_memory=True)
        else:
            return DataLoader(TextDataUnPadded(get_true_data_path(opt, "valid")[1], 0, word_to_idx[opt.pad_token]),
                              collate_fn=collate_fn, batch_size=opt.batch_size, shuffle=False, num_workers=4,
                              pin_memory=True)

    else:
        raise UnknownArgumentError(
            "Cannot load data for --mode={}. Please choose [train, test, qualitative]".format(opt.mode))


def save_checkpoint(opt, model, optimizers, epoch, suffix=""):
    """Save a checkpoint of the given model, including the state of the optimizer and the epoch."""
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': [optimizer.state_dict() for optimizer in optimizers] if isinstance(optimizers, list)
        else optimizers.state_dict(),
        'kl_scale': model._scale if hasattr(model, 'scale') else 1.0,
        'opt': opt
    }
    try:
        torch.save(checkpoint, get_model_path(opt, suffix))
    except FileNotFoundError as e:
        raise InvalidPathError(e) from e


def load_options(opt):
    """Load the options from a state dict."""
    try:
        checkpoint = torch.load(get_model_path(opt, opt.load_criteria), map_location=lambda storage, loc: storage)
    except FileNotFoundError as e:
        raise InvalidPathError("No model to load of type {}.".format(opt.model)) from e
    else:
        try:
            # Load options
            resumed_opt = checkpoint['opt']

            # Overwrite options that matter
            # TODO: should be a nicer way to-do this
            # Note that save_suffix, load_criteria, model and out_folder/ptb_type should be set correctly
            opt.pad_token = resumed_opt.pad_token
            opt.unk_token = resumed_opt.unk_token
            opt.eos_token = resumed_opt.eos_token
            opt.sos_token = resumed_opt.sos_token

            opt.batch_size = resumed_opt.batch_size
            opt.tie_in_out = resumed_opt.tie_in_out
            opt.drop_type = resumed_opt.drop_type
            opt.p = resumed_opt.p
            opt.lr = resumed_opt.lr
            opt.clip = resumed_opt.clip
            opt.cut_off = resumed_opt.cut_off

            if opt.model != "deterministic":
                opt.ann_word = resumed_opt.ann_word
                opt.word_p = resumed_opt.word_p
                opt.kl_step = resumed_opt.kl_step
                opt.alpha = resumed_opt.alpha
                opt.beta = resumed_opt.beta
                opt.lamb = resumed_opt.lamb
                opt.mmd = resumed_opt.mmd
                opt.ann_mode = resumed_opt.ann_mode
                opt.rate_mode = resumed_opt.rate_mode
                opt.posterior = resumed_opt.posterior
                opt.hinge_weight = resumed_opt.hinge_weight
                opt.k = resumed_opt.k
                opt.min_rate = resumed_opt.min_rate
                opt.max_mmd = resumed_opt.max_mmd
                opt.max_elbo = resumed_opt.max_elbo
                opt.word_step = resumed_opt.word_step

                opt.z_dim = resumed_opt.z_dim
                opt.enc_p = resumed_opt.enc_p
                opt.enc_h_dim = resumed_opt.enc_h_dim
                opt.enc_layers = resumed_opt.enc_layers
                opt.enc_word_p = resumed_opt.enc_word_p

                opt.lagrangian = resumed_opt.lagrangian
                opt.constraint = resumed_opt.constraint

                if opt.model == "flowbowman":
                    opt.flow = resumed_opt.flow
                    opt.flow_depth = resumed_opt.flow_depth
                    opt.h_depth = resumed_opt.h_depth
                    opt.prior = resumed_opt.prior
                    opt.num_weights = resumed_opt.num_weights
                    opt.c_dim = resumed_opt.c_dim

            opt.train_file = resumed_opt.train_file
            opt.val_file = resumed_opt.val_file
            opt.test_file = resumed_opt.test_file
            opt.word_dict = resumed_opt.word_dict
            opt.idx_dict = resumed_opt.idx_dict

            opt.rnn_type = resumed_opt.rnn_type
            opt.seq_len = resumed_opt.seq_len
            opt.sparse = resumed_opt.sparse
            opt.v_dim = resumed_opt.v_dim
            opt.x_dim = resumed_opt.x_dim
            opt.h_dim = resumed_opt.h_dim
            opt.s_dim = resumed_opt.s_dim
            opt.layers = resumed_opt.layers
            opt.css = resumed_opt.css

        except KeyError:
            raise Error("No options to load.")

    return opt


def load_checkpoint(opt, model, optimizers):
    """Load all information from a checkpoint, will throw an error if no checkpoint is available."""
    try:
        checkpoint = torch.load(get_model_path(opt, opt.load_criteria), map_location=lambda storage, loc: storage)
    except FileNotFoundError as e:
        raise InvalidPathError("No model to load of type {}.".format(opt.model)) from e
    else:
        model.load_state_dict(checkpoint['model'], strict=False)

        if hasattr(model, 'scale'):
            # When the model does not have a scale attribute, we do not load it
            try:
                model.scale = checkpoint['kl_scale'].item()
            except KeyError:
                # For backward compatibility we catch a key error and return scale as one instead
                model.scale = 1.0

        # For backward compatibility we can also load a single stored optimizer
        if isinstance(checkpoint['optimizer'], list):
            [optimizer.load_state_dict(check) for optimizer, check in zip(optimizers, checkpoint['optimizer'])]
        elif isinstance(optimizers, list):
            optimizers[0].load_state_dict(checkpoint['optimizer'])
        else:
            optimizers.load_state_dict(checkpoint['optimizer'])

        epoch = checkpoint['epoch']
    return model, optimizers, epoch


def load_model(opt, model):
    """Only load the model from a checkpoint, ignoring other information."""
    try:
        checkpoint = torch.load(get_model_path(opt, opt.load_criteria), map_location=lambda storage, loc: storage)
    except FileNotFoundError as e:
        raise InvalidPathError("No model to load of type {}.".format(opt.model)) from e
    else:
        model.load_state_dict(checkpoint['model'], strict=False)

    return model


def get_model_path(opt, suffix=""):
    if opt.script == "generative":
        return osp.join(opt.out_folder, "{}_checkpoint_{}{}.pt".format(opt.model, opt.save_suffix, suffix))
    else:
        raise UnknownArgumentError("--script not recognized, please choose: [generative].")


def get_gen_data_path(opt, mode):
    """Return the paths containing generated data text and indices given user settings and a mode."""
    if mode == 'train':
        indices = osp.join(opt.data_folder, "generated", "{}_{}_{}_{}".format(
            opt.model, opt.seq_len, opt.save_suffix, opt.train_file))
    elif mode == 'valid':
        indices = osp.join(opt.data_folder, "generated", "{}_{}_{}_{}".format(
            opt.model, opt.seq_len, opt.save_suffix, opt.val_file))
    elif mode == 'test':
        indices = osp.join(opt.data_folder, "generated", "{}_{}_{}_{}".format(
            opt.model, opt.seq_len, opt.save_suffix, opt.test_file))
    else:
        raise UnknownArgumentError("mode: {} not recognized. Please choose [train, valid, test]".format(mode))

    text = indices.split(".")[0] + ".txt"
    return text, indices


def get_true_data_path(opt, mode):
    """Return the paths containing true data text and indices given user settings and a mode."""
    if mode == 'train':
        indices = osp.join(opt.data_folder, opt.train_file)
    elif mode == 'valid':
        indices = osp.join(opt.data_folder, opt.val_file)
    elif mode == 'test':
        indices = osp.join(opt.data_folder, opt.test_file)
    else:
        raise UnknownArgumentError("mode: {} not recognized. Please choose [train, valid, test]".format(mode))

    text = indices.split(".")[0] + ".txt"
    return text, indices


def get_novelty_path(opt, suffix=""):
    if not osp.isdir(osp.join(opt.out_folder, "novelty")):
        os.makedirs(osp.join(opt.out_folder, "novelty"))
    return osp.join(opt.out_folder, "novelty", "{}_novelty_{}{}".format(opt.model, opt.save_suffix, suffix))


def save_samples(opt, samples, sample_indices, mode):
    """Save samples and their indices to files."""
    if not osp.isdir(osp.join(opt.data_folder, "generated")):
        os.makedirs(osp.join(opt.data_folder, "generated"))

    sample_file, sample_indices_file = get_gen_data_path(opt, mode)

    with open(sample_file, 'w') as f:
        f.write(samples)
    with open(sample_indices_file, 'w') as f:
        f.write(sample_indices)


def save_novelties(opt, novelty_list):
    pickle.dump(novelty_list, open(get_novelty_path(opt, opt.load_criteria), 'wb'))


def load_novelties(opt, novelty_list):
    return pickle.load(open(get_novelty_path(opt, opt.load_criteria), 'rb'))
