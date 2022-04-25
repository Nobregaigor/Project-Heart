from enum import Enum, IntEnum, EnumMeta

def check_enum(name):
    if isinstance(name, (Enum, IntEnum)):
        name = name.value
    return name

def check_enum_name(name):
    if isinstance(name, (Enum, IntEnum)):
        name = name.name
    return name

def add_to_enum(enum_holder, *args):
    if not issubclass(enum_holder.__class__, EnumMeta):
        raise TypeError("Expected EnumMeta as first argument, got {}".format(enum_holder.__class__))
    enumdict = {name: value.value for (name, value) in enum_holder.__members__.items()}
    n = "_".join([str(check_enum_name(v)).upper() for v in args ])
    v = "_".join([str(check_enum(v)) for v in args ])
    enumdict[n] = v
    return Enum(enum_holder.__name__,enumdict), (n, v)