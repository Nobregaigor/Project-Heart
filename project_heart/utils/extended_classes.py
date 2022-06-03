class ExtendedDict(dict):
    def __init__(self, *args, **kwargs):
        super(ExtendedDict, self).__init__(*args, **kwargs)

    def all(self, key):
        return ExtendedDict({_k: val for _k, val in self.items() if key in _k})

    def prefix(self, key, prefix):
        return ExtendedDict({_k: val for _k, val in self.items() if (key in _k) and _k.startswith(prefix)})

    def suffix(self, key, suffix):
        return ExtendedDict({_k: val for _k, val in self.items() if (key in _k) and _k.endswith(suffix)})