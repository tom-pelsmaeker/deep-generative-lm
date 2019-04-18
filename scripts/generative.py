#!/usr/bin/env python3
"""These scripts can be used to train and test various generative models of language."""

import os.path as osp
import sys
import logging
import time
from warnings import warn
from collections import defaultdict
import pickle

import numpy as np
import torch
from torch.optim import Adam, SparseAdam, RMSprop
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from dataset.text import sort_collate, sort_pad_collate  # noqa: E402
from util.statistics import StatsHandler  # noqa: E402
from util.storage import seed, initialize_model, initialize_dataloader, save_checkpoint, load_checkpoint, load_model, save_samples, load_word_index_maps, load_options, save_novelties  # noqa: E402
from util.evaluation import compute_accuracy, compute_perplexity, get_samples, compute_bleu, compute_active_units, compute_ter, compute_novelty, AdaptiveRate, compute_mutual_information  # noqa: E402
from util.error import UnknownArgumentError, InvalidPathError, NoModelError, Error  # noqa: E402
from util.display import vprint, print_flags  # noqa: E402


def train(opt):
    """Script that trains a generative model of language given various user settings."""
    # Try to load options when we resume
    if opt.resume:
        try:
            opt = load_options(opt)
        except InvalidPathError as e:
            warn("{}\n Starting from scratch...".format(e))
            opt.resume = 0
            epoch = 0
        except Error as e:
            warn("{}\n Make sure all preset arguments coincide with the model you are loading.".format(e))
    else:
        epoch = 0

    # Set device so script works on both GPU and CPU
    seed(opt)
    opt.device = torch.device("cuda:{}".format(opt.local_rank) if opt.local_rank >= 0 else "cpu")
    vprint("Using device: {}".format(opt.device), opt.verbosity, 1)

    word_to_idx, idx_to_word = load_word_index_maps(opt)

    # Here we construct all parts of the training ensemble; the model, dataloaders and optimizer
    data_train, data_eval = initialize_dataloader(opt, word_to_idx, sort_pad_collate)
    opt.N = (len(data_train) - 1) * opt.batch_size
    decoder = initialize_model(opt, word_to_idx)
    optimizers = []
    if opt.sparse:
        sparse_parameters = [p[1] for p in filter(lambda p: p[0] == "emb.weight", decoder.named_parameters())]
        parameters = [p[1] for p in filter(lambda p: p[1].requires_grad and p[0] !=
                                           "emb.weight", decoder.named_parameters())]
        optimizers.append(Adam(parameters, opt.lr))
        optimizers.append(SparseAdam(sparse_parameters, opt.lr))
    elif opt.lagrangian:
        lag_parameters = [p[1] for p in filter(lambda p: p[0] == "lag_weight", decoder.named_parameters())]
        parameters = [p[1] for p in filter(lambda p: p[1].requires_grad and p[0] !=
                                           "lag_weight.weight", decoder.named_parameters())]
        optimizers.append(Adam(parameters, opt.lr))
        optimizers.append(RMSprop(lag_parameters, opt.lr))
    else:
        parameters = filter(lambda p: p.requires_grad, decoder.parameters())
        optimizers.append(Adam(parameters, opt.lr))

    # Load from checkpoint
    if opt.resume:
        decoder, optimizers, epoch = load_checkpoint(opt, decoder, optimizers)

    # The SummaryWriter will log certain values for automatic visualization
    writer = SummaryWriter(osp.join(opt.out_folder, opt.model, opt.save_suffix, 'train'))

    # AdaptiveRate will automatically adapt the rate in a model if turned on
    adapt_rate = AdaptiveRate(opt)

    # The StatsHandler object will store important stastics during training and provides printing and logging utilities
    stats = StatsHandler(opt)

    # We will early stop the network based on user specified criteria
    early_stopping = False
    stop_ticks = 0
    prev_crit = [np.inf] * len(opt.criteria)

    while not early_stopping:
        # We reset the stats object to collect fresh stats for every epoch
        stats.reset()
        epoch += 1
        stats.epoch = epoch

        start = time.time()
        for data in data_train:
            # We zero the gradients BEFORE the forward pass, instead of before the backward, to hopefully save some memory
            [optimizer.zero_grad() for optimizer in optimizers]

            # We skip the remainder batch
            if data[0].shape[0] != opt.batch_size:
                continue

            # Prepare
            decoder.train()
            decoder.use_prior = False
            data = [d.to(opt.device) for d in data]

            # Forward
            losses, pred = decoder(data)
            adapt_rate(losses, decoder)
            loss = sum([v for k, v in losses.items() if k not in ["Lag_Weight"]])

            # Log the various losses the models can return, and accuracy
            stats.train_loss.append(losses["NLL"].item())
            stats.train_kl.append(losses["KL"].item())
            stats.train_elbo.append(losses["NLL"].item() + losses["KL"].item())
            stats.train_min_rate.append(losses["Hinge"].item())
            stats.train_l2_loss.append(losses["L2"].item())
            stats.train_mmd.append(losses["MMD"].item())
            stats.train_acc.append(compute_accuracy(pred, data).item())
            stats.constraint.append(losses["Constraint"].item())
            stats.lamb.append(losses["Lag_Weight"].item())
            del data

            loss.backward()

            # Check for bad gradients
            nan = False
            if opt.grad_check:
                for n, p in decoder.named_parameters():
                    if torch.isnan(p.grad).any():
                        nan = True
                        print("{} Contains nan gradients!".format(n))
            if nan:
                break

            if opt.clip > 0.:
                clip_grad_norm_(decoder.parameters(), opt.clip)  # Might prevent exploding gradients

            if opt.lagrangian:
                # This is equivalent to flipping the sign on the loss and computing its backward
                # So it prevents computation of the backward twice, once for max and once for min
                for group in optimizers[1].param_groups:
                    for p in group['params']:
                        p.grad = -1 * p.grad

            [optimizer.step() for optimizer in optimizers]
        end = time.time()
        print("Train time: {}s".format(end - start))

        start = time.time()
        # We wrap the entire evaluation in no_grad to save memory
        with torch.no_grad():
            zs = []
            log_q_z_xs = []
            log_p_zs = []
            mus = []
            vars = []
            for data in data_eval:
                # Catch small batches
                if data[0].shape[0] != opt.batch_size:
                    continue

                # Prepare
                decoder.eval()
                data = [d.to(opt.device) for d in data]

                # Sample a number of log-likehoods to obtain a low-variance estimate of the model perplexity
                # We do this with a single sample when training for speed. On test we will use more samples
                decoder.use_prior = True
                losses, pred = decoder(data)
                stats.val_loss.append(losses["NLL"].item())
                stats.val_l2_loss.append(losses["L2"].item())
                stats.val_acc.append(compute_accuracy(pred, data).item())
                stats.val_log_loss[0].append(losses["NLL"].item())

                if len(data) > 1:
                    stats.avg_len.append(torch.mean(data[1].float()).item() - 1)
                else:
                    stats.avg_len.append(data[0].shape[1] - 1)

                # Also sample the perplexity for the reconstruction case (using the posterior)
                decoder.use_prior = False
                if opt.mi:
                    losses, pred, var, mu, z, _, log_q_z_x, log_p_z = decoder(data, extensive=True)
                    zs.append(z), log_q_z_xs.append(log_q_z_x), log_p_zs.append(
                        log_p_z), mus.append(mu), vars.append(var)
                else:
                    losses, pred = decoder(data)
                stats.val_rec_loss.append(losses["NLL"].item())
                stats.val_rec_kl.append(losses["KL"].item())
                stats.val_rec_elbo.append(losses["NLL"].item() + losses["KL"].item())
                stats.val_rec_min_rate.append(losses["Hinge"].item())
                stats.val_rec_l2_loss.append(losses["L2"].item())
                stats.val_rec_mmd.append(losses["MMD"].item())
                stats.val_rec_acc.append(compute_accuracy(pred, data).item())
                stats.val_rec_log_loss[0].append(losses["NLL"].item() + losses["KL"].item())

            if opt.mi:
                # Stack the collected samples and parameters
                z = torch.cat(zs, 0)
                log_q_z_x = torch.cat(log_q_z_xs, 0)
                log_p_z = torch.cat(log_p_zs, 0)
                mu = torch.cat(mus, 0)
                var = torch.cat(vars, 0)
                avg_kl = torch.tensor(stats.val_rec_kl, dtype=torch.float, device=opt.device).mean()
                avg_h = log_q_z_x.mean()
                log_q_z = decoder.q_z_estimate(z, mu, var)
                stats.val_mi, stats.val_mkl = compute_mutual_information(
                    z, log_p_z, avg_h, avg_kl, opt.mi_method, opt.mi_kde_method, log_q_z)
        end = time.time()
        print("Eval time: {}s".format(end - start))

        # Compute the perplexity and its variance for this batch, sampled N times
        perplexity, variance = compute_perplexity(stats.val_log_loss, stats.avg_len)
        rec_ppl, rec_var = compute_perplexity(stats.val_rec_log_loss, stats.avg_len)

        stats.val_ppl.append(perplexity)
        stats.val_rec_ppl.append(rec_ppl)
        stats.val_ppl_std.append(variance)
        stats.val_rec_ppl_std.append(rec_var)
        stats.kl_scale = decoder._scale

        # Print and log the statistics after every epoch
        # Note that the StatsHandler object automatically prepares the stats, so no more stats can be added
        vprint(stats, opt.verbosity, 0)
        stats.log_stats(writer)

        # We early stop when the model has not improved certain criteria for a given number of epochs.
        stop = [0] * len(opt.criteria)
        i = 0
        # This is the default criteria; we will stop when the ELBO/LL no longer improves
        if 'posterior' in opt.criteria:
            if stats.val_rec_elbo > (prev_crit[i] - opt.min_imp) and epoch > 4:
                stop[i] = 1
            else:
                stop_ticks = 0
            if stats.val_rec_elbo < prev_crit[i]:
                try:
                    save_checkpoint(opt, decoder, optimizers, epoch)
                except InvalidPathError as e:
                    vprint(e, opt.verbosity, 0)
                    vprint("Cannot save model, continuing without saving...", opt.verbosity, 0)
                prev_crit[i] = stats.val_rec_elbo
            i += 1

        # We early stop the model when an estimate of the log-likelihood based on samples from the prior no longer increases
        # This generally only makes sense when we have a learned prior
        if 'prior' in opt.criteria:
            if stats.val_loss > (prev_crit[i] - opt.min_imp) and epoch > 4:
                stop[i] = 1
            else:
                stop_ticks = 0
            if stats.val_loss < prev_crit[i]:
                try:
                    # For each non standard criteria we add a suffix to the model name
                    save_checkpoint(opt, decoder, optimizers, epoch, 'prior')
                except InvalidPathError as e:
                    vprint(e, opt.verbosity, 0)
                    vprint("Cannot save model, continuing without saving...", opt.verbosity, 0)
                prev_crit[i] = stats.val_loss

        # So far we can choose to either/or save models based on prior loss and posterior loss
        if 'prior' not in opt.criteria and 'posterior' not in opt.criteria:
            raise UnknownArgumentError(
                "No valid early stopping criteria found, please choose either/both [posterior, prior]")

        # We only increase the stop ticks if all criteria are not satisfied
        stop_ticks += int(np.all(np.array(stop)))

        # When we reach a user specified amount of stop ticks, we stop training
        if stop_ticks >= opt.stop_ticks:
            writer.close()
            vprint("Early stopping after {} epochs".format(epoch), opt.verbosity, 0)
            early_stopping = True


def test(opt):
    """Script that tests a generative model of language given various user settings."""
    # Load options, if they are stored
    try:
        opt = load_options(opt)
    except InvalidPathError as e:
        raise NoModelError("Aborting testing without a valid model to load.") from e
    except Error as e:
        warn("{}\n Make sure all preset arguments coincide with the model you are loading.".format(e))

    # We test with a batch size of 1 to get exact results
    batch_size = opt.batch_size
    opt.batch_size = 1

    # Set device so script works on both GPU and CPU
    opt.device = torch.device("cuda:{}".format(opt.local_rank) if opt.local_rank >= 0 else "cpu")
    vprint("Using device: {}".format(opt.device), opt.verbosity, 1)

    word_to_idx, idx_to_word = load_word_index_maps(opt)

    data_test = initialize_dataloader(opt, word_to_idx, sort_pad_collate)
    opt.N = (len(data_test) - 1) * opt.batch_size
    decoder = initialize_model(opt, word_to_idx)

    # Model loading is mandatory when testing, otherwise the tests will not be executed
    decoder = load_model(opt, decoder)

    # The summary writer will log certain values for automatic visualization
    writer = SummaryWriter(osp.join(opt.out_folder, opt.model, opt.save_suffix, 'test'))

    # The StatsHandler object will store important stastics during testing and provides printing and logging utilities
    stats = StatsHandler(opt)

    with torch.no_grad():
        zs = []
        z_priors = []
        vars = []
        mus = []
        preds = []
        datas = []
        log_q_z_xs = []
        log_p_zs = []
        for data in data_test:
            # Catch small batches
            if data[0].shape[0] != opt.batch_size:
                opt.batch_size = data[0].shape[0]

            # Save data in a list
            datas.extend(data[0][:, 1:].tolist())

            # Prepare
            decoder.eval()
            data = [d.to(opt.device) for d in data]

            # Sample a number of log-likehoods to obtain a low-variance estimate of the model perplexity
            decoder.use_prior = True
            # Forward pass the evaluation dataset to obtain statistics
            losses, pred = decoder(data)
            stats.val_loss.append(losses["NLL"].item())
            stats.val_l2_loss.append(losses["L2"].item())
            stats.val_acc.append(compute_accuracy(pred, data).item())
            stats.val_log_loss[0].append(losses["NLL"].item())

            # We need the average length of the sequences to get a fair estimate of the perplexity per word
            if len(data) > 1:
                stats.avg_len.extend([torch.mean(data[1].float()).item() - 1])
            else:
                stats.avg_len.extend([data[0].shape[1] - 1])

            # Also sample the perplexity for the reconstruction case
            decoder.use_prior = False
            losses, pred, var, mu, z, z_prior, log_q_z_x, log_p_z = decoder(data, extensive=True)
            preds.extend(pred.tolist())
            zs.append(z), z_priors.append(z_prior), vars.append(var), mus.append(
                mu), log_q_z_xs.append(log_q_z_x), log_p_zs.append(log_p_z)
            stats.val_rec_loss.append(losses["NLL"].item())
            stats.val_rec_kl.append(losses["KL"].item())
            stats.val_rec_elbo.append(losses["NLL"].item() + losses["KL"].item())
            stats.val_rec_min_rate.append(losses["Hinge"].item())
            stats.val_rec_l2_loss.append(losses["L2"].item())
            stats.val_rec_mmd.append(losses["MMD"].item())
            stats.val_rec_acc.append(compute_accuracy(pred, data).item())
            stats.val_rec_log_loss[0].append(losses["NLL"].item() + losses["KL"].item())
            stats.lamb.append(losses["Lag_Weight"].item())
            stats.constraint.append(losses["Constraint"].item())

        # Stack the collected samples and parameters
        mu = torch.cat(mus, 0)
        var = torch.cat(vars, 0)
        z_prior = torch.cat(z_priors, 0)
        z = torch.cat(zs, 0)
        log_q_z_x = torch.cat(log_q_z_xs, 0)
        log_p_z = torch.cat(log_p_zs, 0)
        avg_kl = torch.tensor(stats.val_rec_kl, dtype=torch.float, device=opt.device).mean()
        avg_h = log_q_z_x.mean()

        # We compute the MMD over the full validation set
        if opt.mmd:
            stats.val_rec_mmd = [decoder._mmd(z, z_prior).item()]

        if opt.mi:
            log_q_z = decoder.q_z_estimate(z, mu, var)
            stats.val_mi, stats.val_mkl = compute_mutual_information(
                z, log_p_z, avg_h, avg_kl, opt.mi_method, opt.mi_kde_method, log_q_z)

        # Active units are the number of dimensions in the latent space that do something
        stats.val_au = compute_active_units(mu, opt.delta)

        # BLEU is a measure of the corpus level reconstruction ability of the model
        stats.val_bleu = compute_bleu(preds, datas, word_to_idx[opt.pad_token])

        if opt.ter:
            print("Computing TER....")
            # TER is another measure of the corpus level reconstruction ability of the model
            stats.val_ter = compute_ter(preds, datas, word_to_idx[opt.pad_token])

            print("Computing Novelty....")
            # Novelty is minimum TER of a generated sentence compared with the training corpus
            stats.val_novelty, _ = compute_novelty(get_samples(opt, decoder, idx_to_word,
                                                               word_to_idx)[1], osp.join(opt.data_folder, opt.train_file), opt, idx_to_word)

        if opt.log_likelihood and opt.model != 'deterministic':
            repeat = max(int(opt.ll_samples / opt.ll_batch), 1)
            opt.ll_batch = opt.ll_samples if opt.ll_samples < opt.ll_batch else opt.ll_batch
            normalizer = torch.log(torch.tensor(int(opt.ll_samples / opt.ll_batch) *
                                                opt.ll_batch, device=opt.device, dtype=torch.float))
            stats.val_nll = []
            for i, data in enumerate(data_test):
                if i % 100 == 0:
                    vprint("\r At {:.3f} percent of log likelihood estimation\r".format(
                        float(i) / len(data_test) * 100), opt.verbosity, 1, end="")
                nelbo = []
                for r in range(repeat):
                    data = [d.expand(opt.ll_batch, *d.size()[1:]).to(opt.device) for d in data]
                    losses, _ = decoder(data, True)
                    nelbo.append(losses["NLL"] + losses["KL"])
                nelbo = torch.cat(nelbo, 0)
                stats.val_nll.append(-(torch.logsumexp(-nelbo - normalizer, 0)).item())
            stats.val_nll = np.array(stats.val_nll).mean()
            stats.val_est_ppl = compute_perplexity([stats.val_nll], stats.avg_len)[0]

    perplexity, variance = compute_perplexity(stats.val_log_loss, stats.avg_len)
    rec_ppl, rec_var = compute_perplexity(stats.val_rec_log_loss, stats.avg_len)

    stats.val_ppl.append(perplexity)
    stats.val_rec_ppl.append(rec_ppl)
    stats.val_ppl_std.append(variance)
    stats.val_rec_ppl_std.append(rec_var)

    vprint(stats, opt.verbosity, 0)
    stats.log_stats(writer)
    writer.close()

    opt.batch_size = batch_size

    if opt.log_likelihood:
        return stats.val_nll + compute_free_bits_from_stats(stats, opt), stats
    else:
        return stats.val_rec_elbo + compute_free_bits_from_stats(stats, opt), stats


def novelty(opt):
    """Script that computes the novelty of generated sentences."""
    # Load options, if they are stored
    try:
        opt = load_options(opt)
    except InvalidPathError as e:
        raise NoModelError("Aborting testing without a valid model to load.") from e
    except Error as e:
        warn("{}\n Make sure all preset arguments coincide with the model you are loading.".format(e))

    # Set device so script works on both GPU and CPU
    opt.device = torch.device("cuda:{}".format(opt.local_rank) if opt.local_rank >= 0 else "cpu")
    vprint("Using device: {}".format(opt.device), opt.verbosity, 1)

    word_to_idx, idx_to_word = load_word_index_maps(opt)
    opt.N = 0
    decoder = initialize_model(opt, word_to_idx)

    # Model loading is mandatory when testing, otherwise the tests will not be executed
    decoder = load_model(opt, decoder)

    with torch.no_grad():
        # Novelty is inverse TER of a generated sentence compared with the training corpus
        novelty, full_scores = compute_novelty(get_samples(opt, decoder, idx_to_word,
                                                           word_to_idx), osp.join(opt.data_folder, opt.train_file), opt, idx_to_word)
    vprint("Novelty: {}".format(novelty), opt.verbosity, 0)
    save_novelties(full_scores)
    return novelty


def qualitative(opt):
    """Print some samples using various techniques."""
    # Load options, if they are stored
    try:
        opt = load_options(opt)
    except InvalidPathError as e:
        raise NoModelError("Aborting testing without a valid model to load.") from e
    except Error as e:
        warn("{}\n Make sure all preset arguments coincide with the model you are loading.".format(e))

    # Set device so script works on both GPU and CPU
    opt.device = torch.device("cuda:{}".format(opt.local_rank) if opt.local_rank >= 0 else "cpu")
    vprint("Using device: {}".format(opt.device), opt.verbosity, 1)

    # Here we construct all parts of the training ensemble; the model, dataloaders and optimizer
    word_to_idx, idx_to_word = load_word_index_maps(opt)
    word_to_idx = defaultdict(lambda: word_to_idx[opt.unk_token], word_to_idx)
    data_train, _ = initialize_dataloader(opt, word_to_idx, sort_pad_collate)

    opt.N = 0
    decoder = initialize_model(opt, word_to_idx)

    # Model loading is mandatory when testing, otherwise the tests will not be executed
    decoder = load_model(opt, decoder)

    with torch.no_grad():
        if opt.qual_file:
            with open(opt.qual_file, 'r') as f:
                sentences = f.read().split('\n')

            # Convert sentences to index tensors
            sentences = [[word_to_idx[word] for word in s.strip().split(' ')] for s in sentences]
            x_len = [len(s) for s in sentences]
            max_len = max(x_len)
            pad_idx = word_to_idx[opt.pad_token]
            for s in sentences:
                s.extend([pad_idx] * (max_len - len(s)))
            sentences = torch.LongTensor(sentences).to(opt.device)
            x_len = torch.LongTensor(x_len)
        else:
            for data in data_train:
                # Select short sentences only
                sentences = data[0][(data[1] < 16) & (data[1] > 4)]
                x_len = data[1][(data[1] < 16) & (data[1] > 4)]
                sentences = sentences.to(opt.device)
                break

        if opt.model != 'deterministic':
            # Print some homotopies
            for i in range(sentences.shape[0] - 1):
                if x_len[i].item() < 2 or x_len[i+1].item() < 2:
                    continue
                homotopy_idx = decoder.homotopies(
                    sentences[i][:x_len[i].item()].unsqueeze(0), sentences[i+1][:x_len[i+1].item()].unsqueeze(0), 9, torch.empty([10, 1], device=opt.device, dtype=torch.long).fill_(word_to_idx[opt.sos_token]), word_to_idx[opt.eos_token], word_to_idx[opt.pad_token], sample_softmax=opt.sample_softmax)
                homotopy = "\n".join([" ".join([idx_to_word[s]
                                                for s in hom if s != word_to_idx[opt.pad_token]]) for hom in homotopy_idx])
                homotopy_idx = "\n".join([" ".join([str(s)
                                                    for s in hom if s != word_to_idx[opt.pad_token]]) for hom in homotopy_idx])
                print("Original:\n {} - {}\n\n".format(" ".join([idx_to_word[s] for s in sentences[i][:x_len[i].item()].tolist()]), " ".join(
                    [idx_to_word[s] for s in sentences[i+1][:x_len[i+1].item()].tolist()])))
                print("Homotopies:\n {}\n\n".format(homotopy))
                if opt.ter:
                    print("Novelties:\n")
                    _ = compute_novelty([homotopy, homotopy_idx], osp.join(
                        opt.data_folder, opt.train_file), opt, idx_to_word)

            # Print some posterior samples
            for i in range(sentences.shape[0]):
                if x_len[i].item() < 2:
                    continue
                pos_samples_idx = decoder.sample_posterior(
                    sentences[i][:x_len[i].item()].unsqueeze(0), 3, word_to_idx[opt.eos_token], word_to_idx[opt.pad_token], sample_softmax=opt.sample_softmax)
                pos_samples = "\n".join(
                    [" ".join([idx_to_word[s] for s in sample if s != word_to_idx[opt.pad_token]]) for sample in pos_samples_idx])
                pos_samples_idx = "\n".join(
                    [" ".join([str(s) for s in sample if s != word_to_idx[opt.pad_token]]) for sample in pos_samples_idx])
                print("Original:\n {} \n\n".format(" ".join([idx_to_word[s]
                                                             for s in sentences[i][:x_len[i].item()].tolist()])))
                print("Samples:\n {} \n\n".format(pos_samples))
                if opt.ter:
                    print("Novelties:\n")
                    _ = compute_novelty([pos_samples, pos_samples_idx],
                                        osp.join(opt.data_folder, opt.train_file), opt, idx_to_word)

            # Print some free posterior samples
            print("Free posterior samples.\n\n")
            for i in range(sentences.shape[0]):
                if x_len[i].item() < 2:
                    continue
                pos_samples_idx = decoder.sample_posterior(
                    sentences[i][:x_len[i].item()].unsqueeze(0), 3, word_to_idx[opt.eos_token], word_to_idx[opt.pad_token], torch.empty([3, 1], device=opt.device, dtype=torch.long).fill_(word_to_idx[opt.sos_token]), "free", sample_softmax=opt.sample_softmax)
                pos_samples = "\n".join(
                    [" ".join([idx_to_word[s] for s in sample if s != word_to_idx[opt.pad_token]]) for sample in pos_samples_idx])
                pos_samples_idx = "\n".join(
                    [" ".join([str(s) for s in sample if s != word_to_idx[opt.pad_token]]) for sample in pos_samples_idx])
                print("Original:\n {} \n\n".format(" ".join([idx_to_word[s]
                                                             for s in sentences[i][:x_len[i].item()].tolist()])))
                print("Samples:\n {} \n\n".format(pos_samples))
                if opt.ter:
                    print("Novelties:\n")
                    _ = compute_novelty([pos_samples, pos_samples_idx],
                                        osp.join(opt.data_folder, opt.train_file), opt, idx_to_word)

        # Print some free prior samples
        print("Free samples:\n")
        if opt.ter:
            _ = compute_novelty(get_samples(opt, decoder, idx_to_word, word_to_idx),
                                osp.join(opt.data_folder, opt.train_file), opt, idx_to_word)
        else:
            print(get_samples(opt, decoder, idx_to_word, word_to_idx)[0])


def compute_free_bits_from_stats(stats, opt):
    return max(opt.bayes_free_bits - stats.val_rec_kl, 0.)


def generate_data(opt):
    """Script that generates data from a (trained) generative model of language and writes it to file."""
    # Load options, if they are stored
    try:
        opt = load_options(opt)
    except InvalidPathError as e:
        raise NoModelError("Aborting generating without a valid model to load.") from e
    except Error as e:
        warn("{}\n Make sure all preset arguments coincide with the model you are loading.".format(e))

    opt.N = 0

    # Set device so script works on both GPU and CPU
    opt.device = torch.device("cuda:{}".format(opt.local_rank) if opt.local_rank >= 0 else "cpu")
    vprint("Using device: {}".format(opt.device), opt.verbosity, 1)

    word_to_idx, idx_to_word = load_word_index_maps(opt)

    decoder = initialize_model(opt, word_to_idx)

    # Model loading is mandatory when generating
    decoder = load_model(opt, decoder)

    decoder.eval()
    for mode in ['train', 'valid', 'test']:
        if mode == 'train':
            tot_samples = file_len(osp.join(opt.data_folder, opt.train_file))
        elif mode == 'valid':
            tot_samples = file_len(osp.join(opt.data_folder, opt.val_file))
        elif mode == 'test':
            tot_samples = file_len(osp.join(opt.data_folder, opt.test_file))

        samples = ""
        sample_indices = ""

        # We sample a user-specified number of samples n times to match the amount of data in the respective files
        for _ in range(int(tot_samples / opt.num_samples)):
            s, si = get_samples(opt, decoder, idx_to_word, word_to_idx)
            samples += s + "\n"
            sample_indices += si + "\n"

        # Here we sample the remainder of the division to exactly match the number of sequences in the data files
        opt.num_samples = int(tot_samples % opt.num_samples)
        s, si = get_samples(opt, decoder, idx_to_word, word_to_idx)
        samples += s
        sample_indices += si

        # Eventualy we save this generated data with a filename that refers to its origin (model and settings)
        save_samples(opt, samples, sample_indices, mode)


def file_len(fname):
    with open(fname, 'r') as f:
        for i, l in enumerate(f):
            pass
    return i + 1


if __name__ == "__main__":
    opt = parse_arguments()
    opt = predefined(opt)
    print_flags(opt)

    # Set script info so this can be used without having to think about this setting
    opt.script = "generative"

    if not osp.isdir(opt.out_folder):
        os.makedirs(opt.out_folder)

    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'test':
        test(opt)
    elif opt.mode == 'generate':
        generate_data(opt)
    else:
        raise UnknownArgumentError("--mode not recognized, please choose: [train, test, generate].")
