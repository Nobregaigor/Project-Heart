from .modules import LV_FiberEstimator


class LV(LV_FiberEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
