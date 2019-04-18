"""Functions that manage what output is displayed."""

import os.path as osp
import sys

# We include the path of the toplevel package in the system path so we can always use absolute imports within the package.
toplevel_path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if toplevel_path not in sys.path:
    sys.path.insert(1, toplevel_path)

from util.error import InvalidArgumentError


def vprint(message, verbosity, verbosity_required, end=None):
    """Print message given the verbosity level."""
    if verbosity >= verbosity_required:
        if end is not None:
            print(message, end=end)
        else:
            print(message)


def print_flags(opt):
    """Prints all entries in an argument parser object."""
    if opt.verbosity > 0:
        print("Settings:")
        for key, value in vars(opt).items():
            print(key + ' : ' + str(value))
        print("\n")
