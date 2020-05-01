"""This module contains custom Exceptions for error management."""

__author__ = "Tom Pelsmaeker"
__copyright__ = "Copyright 2020"


class Error(Exception):
    """Base error class, from which all other errors derive."""
    pass


class UnknownArgumentError(Error):
    """This error will be shown when the user specified argument is unknown."""
    pass


class NoModelError(Error):
    """This error will be shown when there is no model to use."""
    pass


class InvalidPathError(FileNotFoundError):
    """This error will be shown when a path cannot be found or otherwise accessed."""
    pass


class InvalidLengthError(Error):
    """This error will be shown when a given iterator does not have the correct length."""
    pass


class InvalidArgumentError(Error):
    """This error will be shown when a given argument has an invalid value."""
    pass
