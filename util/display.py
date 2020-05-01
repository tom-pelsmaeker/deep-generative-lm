"""Functions that manage what output is displayed."""

__author__ = "Tom Pelsmaeker"
__copyright__ = "Copyright 2020"


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
