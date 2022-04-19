import numpy as np
from operator import xor
from numbers import Number  # for type comparison
from project_heart.enums import DATA_FORMAT, STATES
from enum import Enum


class States():
    def __init__(self, enums={}):
        self.timesteps = []
        self.data = {} # this is the default data container. will be used to export saved data.
        self.data_format = {}
        self.n = lambda: len(self.timesteps)
        self.STATES = enums["STATES"] if "STATES" in enums else STATES
        self.STATE_FORMATS = enums["STATE_FORMATS"] if "STATE_FORMATS" in enums else DATA_FORMAT

    def keys(self) -> set:
        """ Returns all avaiable keys in state """
        return set(self.data.keys())

    def add(self, key: str, data: np.ndarray, data_format: int = DATA_FORMAT.UNKNOWN):

        # try to make key as a standard DATA_FIELD enum.
        try:
            key = self.STATES(key).value
        except ValueError:
            key = self.check_enum(key)
            # key = key
        # add data and data format
        self.data[key] = data
        self.data_format[key] = data_format

    def get(self, key: str, mask=None, i=None, t=None):

        # check if key is valid enum
        key = self.check_enum(key)
        # check if key is string
        if not (isinstance(key, str) or isinstance(key, self.STATES)):
            raise TypeError("key %s must be a string." % key)
        # if timesteps is requested, return timesteps
        if key == "timesteps":
            return np.array(self.timesteps)
        # if key is not timestep, check if key exists
        if not key in self.keys():
            raise KeyError("key '%s' does not exist in states" % key)

        # if a specific index is requested:
        specified_step = False
        if i is not None:
            if i > len(self.timesteps):
                raise ValueError(
                    "i must be less of equal to the length of timesteps. \
                        It refers to the state index.")
            if not isinstance(i, Number):
                raise TypeError(
                    "i must be an integer. It refers to the state index.")
            data = self.data[key][int(i)]
            specified_step = True
        # if a specific timestep is requested:
        elif t is not None:
            if not isinstance(t, Number):
                raise TypeError(
                    "t must be an float. It refers to a state timestep.")
            if not t >= 0:
                raise ValueError("timestep must be positive float.")
            data = self.data[key][self.get_timestep_index(float(t))]
            specified_step = True
        # if no special request:
        else:
            data = self.data[key]
            specified_step = False

        # if mask is requested
        if mask is not None:
            if specified_step:
                return data[mask]
            else:
                return data[:, mask]
        # if mask is not requested
        else:
            return data

    def set_timesteps(self, timesteps: list):
        self.timesteps = list(timesteps)

    def get_timestep_index(self, t: float) -> int:
        """Matches a given timestep to index of state in the timestep array.
            If given timestep is not found, it matches the closest existing timestep.

        Args:
            t ([float]): [timestep]

        Returns:
            [int]: [index of matching timestep]
        """
        if t not in self.timesteps:
            return np.argmin(np.abs(np.asarray(self.timesteps, dtype=np.float32) - t))
        return self.timesteps.index(t)

    def check_key(self, key: str) -> bool:
        key = self.check_enum(key)
        return key in self.data.keys()

    def check_enum(self, name):
        if isinstance(name, Enum):
            name = name.value
        return name

