from project_heart.modules.container.components import States
import numpy as np
from project_heart.enums import DATA_FORMAT, STATES
from .speckle import Speckle
from collections import deque

class SpeckleStates(States):
    def __init__(self, *args, **kwargs):
        super(SpeckleStates, self).__init__(*args, **kwargs)
        self.data_spks_rel = dict()

    def get_spk_state_key(self, spk: object, key: str, join="all"):
        self.check_spk(spk)
        try:
            key = STATES(key).value
        except ValueError:
            key = self.check_enum(key)
        
        join = self.check_enum(join)
        if join == "group":
            return "{}_{}".format(key, spk.group)
        elif join == "name":
            return "{}_{}".format(key, spk.name)
        elif join == "group_name":
            return "{}_{}_{}".format(key, spk.group, spk.name)
        elif join == "all":
            return "{}_{}".format(key, spk.str)
        else:
            raise ValueError("Unknown join method. Options are: group, name, group_name.")
    
    def set_data_spk_rel(self, spks, key):
        key = self.check_enum(key)
        if issubclass(spks.__class__, Speckle):
            spks = deque([spks])
        elif isinstance(spks, (list, tuple, np.ndarray)):
            spks = deque(spks)
        self.data_spks_rel[key] = spks

    def get_spks_from_data(self, key):
        key = self.check_enum(key)
        if key in self.data_spks_rel:
            return self.data_spks_rel[key]
        else:
            raise KeyError("Key '{}' does not exist within data_spks_rel. "\
                "Are you sure data was computed using spks? "\
                "If so, check if data was saved using 'add_spk_data' or "\
                "if 'set_data_spk_rel' was used after data computation."
                .format(key))

    def add_spk_data(self, spk: object, key: str, data: np.ndarray, join="all"):
        self.check_spk(spk)
        state_key = self.get_spk_state_key(spk, key, join=join)
        self.data[state_key] = data
        self.data_format[state_key] = DATA_FORMAT.SPK
        self.set_data_spk_rel(spk, state_key)
    
    def get_spk_data(self, spk: object, key: str, join="all", **kwargs):
        self.check_spk(spk)
        state_key = self.get_spk_state_key(spk, key, join=join)
        return self.get(key=state_key, **kwargs)

    def check_spk(self, spk: object):
        if not issubclass(spk.__class__, Speckle):
            raise ValueError("spk must be derived from Speckle class."
                            "Received: {}".format(spk.__class__))

    def check_spk_key(self, spk: object, key: str, join="all"):
        state_key = self.get_spk_state_key(spk, key, join=join)
        return self.check_key(state_key)
