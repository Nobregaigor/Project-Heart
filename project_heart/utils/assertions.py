try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable



def assert_iterable(arg, accept_none=True):
    if accept_none and arg is None:
        return True
    else:
        assert isinstance(arg, Iterable), "Expected iterable. Received: '{}'".format(type(arg))