import numpy as np
import warnings
import pickle

DEFAULT_SUBSET_KEY = "default"
DEFAULT_NAME_KEY = "default"
DEFAULT_GROUP_KEY = "default"
DEFAULT_COLLECTION_KEY = "default"
DEFAULT_T_KEY = 0.0

DEFAULT_REF_KEY = 0.0

from collections import deque
from enum import Enum
from project_heart.enums import SPK_SETS

class Speckle():
    def __init__(self,
                 subset:str=DEFAULT_SUBSET_KEY,
                 name:str=DEFAULT_NAME_KEY,
                 group:str=DEFAULT_GROUP_KEY,
                 collection:str=DEFAULT_COLLECTION_KEY,
                 t:float=DEFAULT_T_KEY,
                 k:float=DEFAULT_REF_KEY,
                 center:np.ndarray=None,
                 radius:float=0.0,
                 mask:np.ndarray=None,
                 elmask:np.ndarray=None,
                 ids:np.ndarray=None,
                 normal:np.ndarray=None,
                 k_ids:np.ndarray=None,
                 ):
        
        # check for valid keys
        assert isinstance(subset, (str, int, float)), AssertionError(
            "Subset must by a string or integer or a float.")
        assert isinstance(name, (str, int, float)), AssertionError(
            "Name must by a string or integer or a float.")
        assert isinstance(group, (str, int, float)), AssertionError(
            "Group must by a string or integer or a float.")
        assert isinstance(collection, (str, int, float)), AssertionError(
            "Collection must by a string or integer or a float.")

        # set default values
        center = center if center is not None else np.zeros(3)
        mask = mask if mask is not None else np.asarray([])
        elmask = elmask if elmask is not None else np.asarray([])
        ids = ids if ids is not None else np.asarray([])
        normal = normal if normal is not None else np.asarray([])       
        k_ids = k_ids if k_ids is not None else np.asarray([])       
        
        
        # reference keys
        self.subset = subset
        self.name = name
        self.group = group
        self.collection = collection
        
        # reference metrics
        self.k = k           # relative position (percentage) from apex to base at long. line
        self.t = t           # timestep used to create speckle (right now we just use first ts)
        self.center = center # original center based on k position at longitudinal line 
        self.radius = radius # reference radius at moment of speckle creation
        self.normal = normal # normal to speckle plane
        
        # geoemetry relationships
        self.mask = mask     
        self.elmask = elmask
        self.ids = ids
        self.k_ids = k_ids  # spk cluster ids (array of ids arrays -> (k,n))

        # conversion to string (just to facilitate parts of the code).
        self.str = "{}_{}_{}_{}".format(
            self.subset, self.name, self.group, self.collection)


    def __repr__(self):
        return "<Speckle: .subset: {}, .name: {}, .group: {}, .collection: {}, .t: {}>".format(
            self.subset, self.name, self.group, self.collection, self.t)

    def intersection(self, other):
        """ Returns the interction of self.mask and other Specke mask """
        return np.logical_and(self.mask, other.mask)

    def update_group(self, ref):
        self.group = ref
    
    def key(self):
        return hash((self.t, self.subset, self.name, self.group, self.collection))

class SpeckeDeque(deque):
    def __init__(self, *args, **kwargs):
        super(SpeckeDeque, self).__init__(*args, **kwargs)
    
    def by(self, option):

        if isinstance(option, Enum):
            option = option.value

        grouped = {}
        if option == SPK_SETS.GROUP.value:
            for spk in list(self):
                if spk.group not in grouped:
                    grouped[spk.group] = SpeckeDeque([spk])
                else:
                    grouped[spk.group].append(spk)
        elif option == SPK_SETS.NAME.value:
            for spk in list(self):
                if spk.name not in grouped:
                    grouped[spk.name] = SpeckeDeque([spk])
                else:
                    grouped[spk.name].append(spk)
        elif option == SPK_SETS.GROUP_NAME.value:
            for spk in list(self):
                if (spk.group, spk.name) not in grouped:
                    grouped[(spk.group, spk.name)] = SpeckeDeque([spk])
                else:
                    grouped[(spk.group, spk.name)].append(spk)
        else:
            raise ValueError("Options are: name, group, group_name. Check SPK_SETS for details.")

        return grouped


class SpecklesDict():
    def __init__(self):
        self._speckles = dict()

        self._collections = set((DEFAULT_COLLECTION_KEY,))
        self._groups = set((DEFAULT_GROUP_KEY,))
        self._names = set((DEFAULT_NAME_KEY,))
        self._subsets = set((DEFAULT_SUBSET_KEY,))

    def __repr__(self):
        to_print = ""
        to_print += "<SpecklesDict>: \n"
        to_print += "Collections: \n{}\n".format(self._collections)
        to_print += "Groups: \n{}\n".format(self._groups)
        to_print += "Names: \n{}\n".format(self._names)
        to_print += "Subsets: \n{}\n".format(self._subsets)
        return to_print

    # ------------------------------
    # standard methods

    def get_key(self,
                subset=DEFAULT_SUBSET_KEY,
                name=DEFAULT_NAME_KEY,
                group=DEFAULT_GROUP_KEY,
                collection=DEFAULT_COLLECTION_KEY,
                t=DEFAULT_T_KEY,
                **kwargs):
        return hash((t, subset, name, group, collection))

    def append(self, **kwargs):
        """Appends new Speckles object to list
        """
        key = self.get_key(**kwargs)
        self._speckles[key] = Speckle(**kwargs)
        if "collection" in kwargs:
            self._collections.add(kwargs["collection"])
        if "group" in kwargs:
            self._groups.add(kwargs["group"])
        if "name" in kwargs:
            self._names.add(kwargs["name"])
        if "subset" in kwargs:
            self._subsets.add(kwargs["subset"])

    def get(self, spk_subset=None, spk_name=None, spk_group=None, spk_collection=None, i=None, t=None, **kwargs):

        assert spk_subset is not None or \
            spk_name is not None or \
            spk_group is not None or \
            spk_collection is not None, \
            warnings.warn("No selection criteria was specified")

        selection = deque()
        # start with broad selection and narrow down
        if spk_collection is not None:
            selection = [spk for spk in self._speckles.values(
            ) if spk.collection == spk_collection]
        if spk_group is not None:
            search_at = self._speckles.values()
            if len(selection) > 0:
                search_at = selection
            selection = [spk for spk in search_at if spk.group == spk_group]
        if spk_name is not None:
            search_at = self._speckles.values()
            if len(selection) > 0:
                search_at = selection
            selection = [spk for spk in search_at if spk.name == spk_name]
        if spk_subset is not None:
            search_at = self._speckles.values()
            if len(selection) > 0:
                search_at = selection
            selection = [spk for spk in search_at if spk.subset == spk_subset]
        if t is not None:
            search_at = self._speckles.values()
            if len(selection) > 0:
                search_at = selection
            selection = [spk for spk in search_at if spk.t == t]

        if i is not None and len(selection) > 0:
            return selection[i]
        return SpeckeDeque(selection)

    def remove(self, **kwargs):
        selected = self.get(**kwargs)
        for spk in selected:
            self._speckles.remove(spk)

    # ------------------------------
    # these modify information from selected speckles

    def form_group(self, new_group: str, spk_name=None, spk_collection=None):
        """Sets all selected speckles (by name and/or by collection) to specified group

        Args:
            new_group (str): [name of new group]
            spk_name (str, optional): [speckles reference name]. Defaults to None.
            spk_collection (str, optional): [speckles reference collection]. Defaults to None.
        """
        selected = self.get(spk_name=spk_name, spk_collection=spk_collection)
        for spk in selected:
            spk.spk_group = new_group

    def form_collection(self, new_collection, spk_name=None, spk_group=None):
        """Sets all selected speckles (by name and/or by group) to specified collection

        Args:
            new_collection (str): [name of new collection]
            spk_name (str, optional): [speckles reference name]. Defaults to None.
            spk_group (str, optional): [speckles reference group]. Defaults to None.
        """
        selected = self.get(spk_name=spk_name, spk_group=spk_group)
        for spk in selected:
            spk.spk_collection = new_collection

    # ------------------------------
    # these returns unified information from multiple speckles

    def get_ids(self, **kwargs):
        selected = self.get(**kwargs)
        ids = set()
        for spk in selected:
            ids.add(spk.ids)
        return ids

    def get_mask(self, **kwargs):
        selected = self.get(**kwargs)
        if len(selected) > 0:
            mask = selected[0].mask
            if len(selected) > 1:
                for spk in selected[1:]:
                    mask = np.logical_or(mask, spk.mask)
            return mask
        else:
            warnings.warn(
                "No mask found as no spk was selected. kwargs: {}".format(kwargs))
            return []

    def merge(self, new_subset=None, new_name=None, new_group=None, new_collection=None, **kwargs):
        selected = self.get(**kwargs)

        k = np.mean([v.k for v in selected])
        center = np.mean([v.center for v in selected])
        normal = np.mean([v.normal for v in selected], axis=0)
        mask = self.get_mask(**kwargs)
        ids = self.get_ids(**kwargs)

        self.append(
            subset=new_subset, name=new_name, group=new_group, collection=new_collection,
            k=k, center=center, normal=normal, mask=mask, ids=ids)

        return self.get(spk_subset=new_subset, spk_name=new_name, spk_group=new_group, spk_collection=new_collection)

    # ------------------------------
    # I/O operations

    def save(self, filepath):
        with open(filepath, 'wb') as handle:
            pickle.dump(self._speckles, handle)

    def load(self, filepath):
        with open(filepath, 'rb') as handle:
            data = pickle.load(handle)
            if isinstance(data, list):
                _asdict = dict()
                for item in data:
                    key = self.get_key(
                        subset=item.subset, name=item.name, group=item.group, collection=item.collection)
                    _asdict[key] = item
                self._speckles = _asdict
            elif isinstance(data, dict):
                self._speckles = data
            else:
                raise ValueError("Invalid data type loaded for speckles.")
        self.n_groups = len(self._speckles)
