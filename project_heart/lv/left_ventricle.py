from .modules import LV_Geometry


class LV(LV_Geometry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
