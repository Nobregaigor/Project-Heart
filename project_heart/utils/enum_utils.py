from enum import Enum, IntEnum, EnumMeta

def check_enum_value(name):
    if isinstance(name, (Enum, IntEnum)):
        name = name.value
    return name

def check_enum_name(name):
    if isinstance(name, (Enum, IntEnum)):
        name = name.name
    return name

def check_for_enum_name(value, enum_holder):
    if value in enum_holder.__members__.keys():
        return enum_holder[value]
    else:
        return value

def check_enum(arg):
    return check_enum_name(arg), check_enum_value(arg)

def add_to_enum(enum_holder, *args):
    if not issubclass(enum_holder.__class__, EnumMeta):
        raise TypeError("Expected EnumMeta as first argument, got {}".format(enum_holder.__class__))
    enumdict = {name: value.value for (name, value) in enum_holder.__members__.items()}
    n = "_".join([str(check_enum_name(v)).upper() for v in args ])
    v = "_".join([str(check_enum_value(v)) for v in args ])
    enumdict[n] = v
    return Enum(enum_holder.__name__,enumdict), (n, v)

def enum_to_dict(enum_holder):
    return {name: value.value for (name, value) in enum_holder.__members__.items()}


def assert_member(enum_holder, member):
    assert member in enum_holder.__members__, (
        "Invalid Enum member. Member must be a valid enum for '{}'. \n"
        "Received: '{}'. Options are (member, value):\n"
        "{}".format(member, enum_holder.__name__, enum_to_dict(enum_holder)))

def assert_value(enum_holder, value):
    assert value in enum_holder.__members__.values(), (
        "Invalid Enum value. Value must be a valid enum for '{}'. \n"
        "Received: '{}'. Options are (member, value):\n"
        "{}".format(value, enum_holder.__name__, enum_to_dict(enum_holder)))