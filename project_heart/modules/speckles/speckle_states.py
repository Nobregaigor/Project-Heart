from project_heart.modules.container.components import States
import numpy as np
from project_heart.enums import DATA_FORMAT, STATES
from .speckle import Speckle


class SpeckleStates(States):
    def __init__(self, *args, **kwargs):
        super(SpeckleStates, self).__init__(*args, **kwargs)

    def set_spk_state_key(self, spk: object, key: str):
        self.check_spk(spk)
        try:
            key = STATES(key).value
        except ValueError:
            key = self.check_enum(key)
        return "{}_{}".format(key, spk.str)

    def add_spk_data(self, spk: object, key: str, data: np.ndarray):
        self.check_spk(spk)
        state_key = self.set_spk_state_key(spk, key)
        self.data[state_key] = data
        self.data_format[state_key] = DATA_FORMAT.SPK

    def get_spk_data(self, spk: object, key: str, **kwargs):
        self.check_spk(spk)
        state_key = self.set_spk_state_key(spk, key)
        return self.get(key=state_key, **kwargs)

    def check_spk(self, spk: object):
        if not issubclass(spk.__class__, Speckle):
            raise ValueError("spk must be derived from Speckle class.")

    def check_spk_key(self, spk: object, key: str):
        state_key = self.set_spk_state_key(spk, key)
        return self.check_key(state_key)
