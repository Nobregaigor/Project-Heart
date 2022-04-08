from .modules import LV_ContainerHandler


class LV(LV_ContainerHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
