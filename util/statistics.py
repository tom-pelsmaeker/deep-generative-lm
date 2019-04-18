"""Utilities for handling important statistics during training and testing."""

import numpy as np


class StatsHandler(object):
    """StatsHandler handles the storage, logging and printing of training and testing statistics."""

    def __init__(self, opt):
        num_samples = 1
        self.train_loss = list()
        self.kl_scale = 0.
        self.train_kl = list()
        self.train_elbo = list()
        self.train_min_rate = list()
        self.train_l2_loss = list()
        self.train_acc = list()
        self.train_mmd = list()
        self.val_loss = list()
        self.val_l2_loss = list()
        self.val_acc = list()
        self.val_log_loss = [[] for i in range(num_samples)]
        self.val_ppl = list()
        self.val_ppl_std = list()
        self.val_rec_log_loss = [[] for i in range(num_samples)]
        self.val_rec_acc = list()
        self.val_rec_kl = list()
        self.val_rec_elbo = list()
        self.val_rec_min_rate = list()
        self.val_rec_l2_loss = list()
        self.val_rec_ppl = list()
        self.val_rec_ppl_std = list()
        self.val_rec_loss = list()
        self.val_rec_mmd = list()
        self.avg_len = list()
        self.lamb = list()
        self.constraint = list()
        self.val_nll = 0.
        self.val_mi = 0.
        self.val_mkl = 0.
        self.val_est_ppl = 0.
        self.val_au = 0.
        self.val_bleu = 0.
        self.val_ter = 0.
        self.val_novelty = 0.
        self.epoch = 0
        self.opt = opt
        self.prepared = False

    def __str__(self):
        if not self.prepared:
            try:
                self.prepare_stats()
            except ValueError:
                return ("No stats collected, nothing to print.")

        if self.opt.mode == 'train':
            return ("__________________________________\n" +
                    "Epoch: {:.3f}\n".format(self.epoch) +
                    "Moving Average ELBO: {:.3f} (KL: {:.3f}, Scale: {:.3f})\n".format(self.train_elbo, self.train_kl, self.kl_scale) +
                    "Hinge Loss: {:.3f}, MMD: {:.3f}, L2 Loss: {:.3f}\n".format(self.train_min_rate, self.train_mmd, self.train_l2_loss) +
                    "Accuracy: {:.3f}\n".format(self.train_acc) +
                    "_______________\n" +
                    "Val Prior Loss: {:.3f}\n".format(self.val_loss) +
                    "Val Prior PPL: {:.3f} ({:.3f})\n".format(self.val_ppl, self.val_ppl_std) +
                    "Val Prior Accuracy: {:.3f}\n".format(self.val_acc) +
                    "_______________ \n" +
                    "Val ELBO: {:.3f} (KL: {:.3f})\n".format(self.val_rec_elbo, self.val_rec_kl) +
                    "Val ELBO PPL: {:.3f} ({:.3f})\n".format(self.val_rec_ppl, self.val_rec_ppl_std) +
                    "Val Hinge Loss: {:.3f}, MMD: {:.3f}, L2 Loss: {:.3f}\n".format(self.val_rec_min_rate, self.val_rec_mmd, self.val_rec_l2_loss) +
                    "Val Accuracy: {:.3f}\n".format(self.val_rec_acc) +
                    "Lambda: {:.3f}, Constraint: {:.3f}\n".format(self.lamb, self.constraint) +
                    "Val MI: {:.3f}, Val Marginal KL: {:.3f}\n".format(self.val_mi, self.val_mkl) +
                    "__________________________________\n"
                    )
        else:
            return ("__________________________________\n" +
                    "Test Prior Loss: {:.3f}\n".format(self.val_loss) +
                    "Test Prior PPL: {:.3f} ({:.3f})\n".format(self.val_ppl, self.val_ppl_std) +
                    "Test Prior Accuracy: {:.3f}\n".format(self.val_acc) +
                    "_______________ \n" +
                    "Test ELBO: {:.3f} ({:.3f})\n".format(self.val_rec_elbo, self.val_rec_kl) +
                    "Test Estimated NLL: {:.3f}\n".format(self.val_nll) +
                    "Test Estimated PPL: {:.3f}\n".format(self.val_est_ppl) +
                    "Test ELBO PPL: {:.3f} ({:.3f})\n".format(self.val_rec_ppl, self.val_rec_ppl_std) +
                    "Test Hinge Loss: {:.3f}, MMD: {:.3f}, L2 Loss: {:.3f}\n".format(self.val_rec_min_rate, self.val_rec_mmd, self.val_rec_l2_loss) +
                    "Test Accuracy: {:.3f}\n".format(self.val_rec_acc) +
                    "Test Active Units: {}\n".format(self.val_au) +
                    "Test Corpus BLEU: {:.3f}\n".format(self.val_bleu) +
                    "Test Corpus TER: {:.3f}\n".format(self.val_ter) +
                    "Test Novelty: {:.3f}\n".format(self.val_novelty) +
                    "Lambda: {:.3f}, Constraint: {:.3f}\n".format(self.lamb, self.constraint) +
                    "Test MI: {:.3f}, Test Marginal KL: {:.3f}\n".format(self.val_mi, self.val_mkl) +
                    "__________________________________\n"
                    )

    def log_stats(self, writer):
        """Logs the collected stats to a TensorBoard logging file given a SummaryWriter."""
        if not self.prepared:
            try:
                self.prepare_stats()
            except ValueError:
                print("No stats collected, nothing to log.")
                return

        if self.opt.mode == 'train':
            writer.add_scalar("Train Loss", self.train_loss, self.epoch)
            writer.add_scalar("Train KL", self.train_kl, self.epoch)
            writer.add_scalar("Train ELBO", self.train_elbo, self.epoch)
            writer.add_scalar("Train Min Rate", self.train_min_rate, self.epoch)
            writer.add_scalar("Train L2 Loss", self.train_l2_loss, self.epoch)
            writer.add_scalar("Train MMD", self.train_mmd, self.epoch)
            writer.add_scalar("KL Step", self.kl_scale, self.epoch)
            writer.add_scalar("Train Accuracy", self.train_acc, self.epoch)
            writer.add_scalar("Val Prior Loss", self.val_loss, self.epoch)
            writer.add_scalar("Val Prior L2 Loss", self.val_l2_loss, self.epoch)
            writer.add_scalar("Val Prior Accuracy", self.val_acc, self.epoch)
            writer.add_scalar("Val Prior PPL", self.val_ppl, self.epoch)
            writer.add_scalar("Val Marginal KL", self.val_mkl, self.epoch)
            writer.add_scalar("Val MI", self.val_mi, self.epoch)
            writer.add_scalar("Val Loss", self.val_rec_loss, self.epoch)
            writer.add_scalar("Val KL", self.val_rec_kl, self.epoch)
            writer.add_scalar("Val ELBO", self.val_rec_elbo, self.epoch)
            writer.add_scalar("Val Min Rate", self.val_rec_min_rate, self.epoch)
            writer.add_scalar("Val L2 Loss", self.val_rec_l2_loss, self.epoch)
            writer.add_scalar("Val Accuracy", self.val_rec_acc, self.epoch)
            writer.add_scalar("Val PPL", self.val_rec_ppl, self.epoch)
            writer.add_scalar("Val MMD", self.val_rec_mmd, self.epoch)
        else:
            writer.add_scalar("Test Prior Loss", self.val_loss, self.epoch)
            writer.add_scalar("Test Prior L2 Loss", self.val_l2_loss, self.epoch)
            writer.add_scalar("Test Prior Accuracy", self.val_acc, self.epoch)
            writer.add_scalar("Test Prior PPL", self.val_ppl, self.epoch)
            writer.add_scalar("Test Marginal KL", self.val_mkl, self.epoch)
            writer.add_scalar("Test MI", self.val_mi, self.epoch)
            writer.add_scalar("Test Loss", self.val_rec_loss, self.epoch)
            writer.add_scalar("Test KL", self.val_rec_kl, self.epoch)
            writer.add_scalar("Test ELBO", self.val_rec_elbo, self.epoch)
            writer.add_scalar("Test Min Rate", self.val_rec_min_rate, self.epoch)
            writer.add_scalar("Test L2 Loss", self.val_rec_l2_loss, self.epoch)
            writer.add_scalar("Test Accuracy", self.val_rec_acc, self.epoch)
            writer.add_scalar("Test PPL", self.val_rec_ppl, self.epoch)

        writer.add_scalar("Lagrangian Lambda", self.lamb, self.epoch)
        writer.add_scalar("Lagrangian Constraint", self.constraint, self.epoch)

    def prepare_stats(self):
        """Prepares the collected stats for printing and logging.

        Note: this function should only be called after collecting all stats, further collection of stats is impossible
        after calling this function. Call reset() to start collecting new stats. This function is automatically called
        before printing or logging.
        """
        if self.opt.mode == 'train':
            self.train_loss = self.moving_average(self.train_loss, len(self.train_loss))
            self.train_kl = self.moving_average(self.train_kl, len(self.train_kl))
            self.train_elbo = self.moving_average(self.train_elbo, len(self.train_elbo))
            self.train_min_rate = self.moving_average(self.train_min_rate, len(self.train_min_rate))
            self.train_l2_loss = self.moving_average(self.train_l2_loss, len(self.train_l2_loss))
            self.train_acc = np.array(self.train_acc).mean()
            self.train_mmd = np.array(self.train_mmd).mean()

        self.lamb = self.moving_average(self.lamb, len(self.lamb))
        self.constraint = self.moving_average(self.constraint, len(self.constraint))

        self.val_loss = np.array(self.val_loss).mean()
        self.val_l2_loss = np.array(self.val_l2_loss).mean()
        self.val_ppl = np.array(self.val_ppl).mean()
        self.val_ppl_std = np.array(self.val_ppl_std).mean()
        self.val_acc = np.array(self.val_acc).mean()

        self.val_rec_loss = np.array(self.val_rec_loss).mean()
        self.val_rec_kl = np.array(self.val_rec_kl).mean()
        self.val_rec_elbo = np.array(self.val_rec_elbo).mean()
        self.val_rec_min_rate = np.array(self.val_rec_min_rate).mean()
        self.val_rec_l2_loss = np.array(self.val_rec_l2_loss).mean()
        self.val_rec_ppl = np.array(self.val_rec_ppl).mean()
        self.val_rec_ppl_std = np.array(self.val_rec_ppl_std).mean()
        self.val_rec_acc = np.array(self.val_rec_acc).mean()
        self.val_rec_mmd = np.array(self.val_rec_mmd).mean()

        self.prepared = True

    @staticmethod
    def moving_average(losses, max_step):
        """Computes the linear decaying moving average of given losses over a range of steps."""
        if max_step < 1:
            raise ValueError("max_steps needs to be above 1.")
        if not losses:
            raise ValueError("losses cannot be empty.")
        denom = np.array(range(1, max_step + 1))
        return sum(np.array(losses[-max_step:]) * denom) / denom.sum()

    def reset(self):
        """Resets all stats to their empty state."""
        self.__init__(self.opt)
