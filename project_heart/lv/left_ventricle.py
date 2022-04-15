from .modules import LV_FiberEstimator, LV_Speckles


class LV(LV_FiberEstimator, LV_Speckles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
