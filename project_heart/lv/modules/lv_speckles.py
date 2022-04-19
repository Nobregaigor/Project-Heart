import logging
import os
from pathlib import Path
import numpy as np
from collections import deque

from project_heart.modules.speckles import SpecklesDict, SpeckleStates, Speckle
from project_heart.utils.vector_utils import *
from project_heart.utils.spatial_utils import *
from project_heart.utils.spatial_points import *
from project_heart.utils.cloud_ops import *
from project_heart.enums import *
from .lv_region_identifier import LV_RegionIdentifier


from sklearn.cluster import KMeans
from functools import reduce


logging.basicConfig()

default_lvSpeckles_enums = {
    "SPK_SETS": LV_SPK_SETS
}

class LV_Speckles(LV_RegionIdentifier):
    """_summary_
    """

    def __init__(self, enums={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speckles = SpecklesDict()
        self.states = SpeckleStates()

        # ---- Enums
        self.SPK_SETS = LV_SPK_SETS
        if len(enums) > 0:
            self.config_enums(enums, check_keys=default_lvSpeckles_enums.keys())

    def create_speckles(self,
                        name=None,
                        group=None,
                        collection=None,
                        d=3.0,
                        from_nodeset=None,
                        perpendicular_to=None,
                        normal_to=None,
                        k=1.0,
                        kmin=0.1,
                        kmax=0.95,
                        n_subsets=0,
                        subsets_criteria="z",
                        subsets_names=[],
                        t=0.0,
                        include_elmask=False,
                        _use_long_line=False,
                        log_level=logging.WARNING,
                        **kwargs
                        ):
        """Creates Speckles

        Args:
            name (_type_, optional): _description_. Defaults to None.
            group (_type_, optional): _description_. Defaults to None.
            collection (_type_, optional): _description_. Defaults to None.
            d (float, optional): _description_. Defaults to 3.0.
            from_nodeset (_type_, optional): _description_. Defaults to None.
            perpendicular_to (_type_, optional): _description_. Defaults to None.
            normal_to (_type_, optional): _description_. Defaults to None.
            k (float, optional): _description_. Defaults to 1.0.
            kmin (float, optional): _description_. Defaults to 0.1.
            kmax (float, optional): _description_. Defaults to 0.95.
            n_subsets (int, optional): _description_. Defaults to 0.
            subsets_criteria (str, optional): _description_. Defaults to "z".
            subsets_names (list, optional): _description_. Defaults to [].
            t (float, optional): _description_. Defaults to 0.0.
            include_elmask (bool, optional): _description_. Defaults to False.
            _use_long_line (bool, optional): _description_. Defaults to False.
            log_level (_type_, optional): _description_. Defaults to logging.WARNING.

        Raises:
            RuntimeError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        # set logger
        logger = logging.getLogger('create_speckles')
        logger.setLevel(log_level)

        logger.info("Adding spk subset.")
        logger.info("name: {}".format(name))
        logger.info("group: {}".format(group))
        logger.info("collection: {}".format(collection))
        logger.info("Subname(s): {}".format(subsets_names))
        logger.info("t (timestep): {}".format(t))  # timestep

        # determine nodes to use
        if from_nodeset is None:
            logger.debug("Using nodes from nodeset %s" % from_nodeset)
            # keep track of nodes data
            nodes = self.nodes()
            # keep track of nodes ids
            ids = np.arange(1, len(nodes) + 1, dtype=np.int32)
        else:
            logger.debug("Using all avaiable nodes.")
            ids = self.get_nodeset(from_nodeset)
            nodes = self.nodes(mask=ids)

        # Determine which normal to use
        if perpendicular_to is not None:
            assert len(perpendicular_to) == 3, ValueError(
                'Perpendicular vector length must be 3 -> [x,y,z]')
            normal = np.cross(self.get_normal(), perpendicular_to)
        # if user provided 'normal_to' overwrite normal value to match user's requested value
        elif normal_to is not None:
            assert len(normal_to) == 3, ValueError(
                'Normal vector length must be 3 -> [x,y,z]')
            normal = np.array(normal_to, dtype=np.float32)
        else:
            normal = self.get_normal()
        logger.debug("Normal: {}".format(normal))

        # get longitudinal line
        long_line = self.get_long_line()
        logger.debug("long_line: {}".format(long_line))

        # determine point along longitudinal line to be used as reference
        p = get_p_along_line(k, long_line)
        spk_center = p
        logger.debug("ref p (spk center): {}".format(p))
        # get points close to plane at height k (from p) and threshold d
        ioi = get_pts_close_to_plane(
            nodes, d, normal, p)
        pts = nodes[ioi]
        logger.debug("pts close to plane: {}".format(len(pts)))

        # check for k boundaries --> We assume the long-line
        if angle_between(self.get_normal(), self._Z) < np.radians(10):
            adjusted_due_normal_aligment = False
            if kmin > 0:
                adjusted_due_normal_aligment = True
                logger.debug(
                    "Adjusting for geometry aligned with normal. kmin: {}".format(kmin))
                p = get_p_along_line(kmin, long_line)
                ioi = np.setdiff1d(ioi, np.where(nodes[:, 2] < p[2]))
                pts = nodes[ioi]
            if kmax > 0:
                adjusted_due_normal_aligment = True
                logger.debug(
                    "Adjusting for geometry aligned with normal. kmax: {}".format(kmax))
                p = get_p_along_line(kmax, long_line)
                ioi = np.setdiff1d(ioi, np.where(nodes[:, 2] > p[2]))
                pts = nodes[ioi]
            if adjusted_due_normal_aligment:
                logger.debug(
                    "New spk center and pts found after adjustment.")
                logger.debug("ref p (spk center): {}".format(p))
                logger.debug("pts close to plane: {}".format(len(pts)))

        if len(pts) == 0:
            raise RuntimeError("Found number of points is zero.")

        valid_ids = ids[ioi]  # get valid ids
        mask = ioi  # self.Nodes.create_mask_from_ids(valid_ids)  # global mask

        if n_subsets <= 1:
            logger.debug("Adding single subset.")
            self.speckles.append(
                name=name,
                group=group,
                collection=collection,
                t=t,
                k=k,
                mask=mask,
                elmask=None,
                ids=valid_ids,
                normal=normal,
                center=spk_center
            )
        else:
            # cehck if subset names was provided
            logger.debug("Adding multiple subsets: {}.".format(n_subsets))
            if n_subsets > 0 and len(subsets_names) > 0:
                assert len(subsets_names) == n_subsets, AssertionError(
                    "If list of prefixes was provided, its length must be \
                        equal to the number of subsets.")
            elif n_subsets > 0 and len(subsets_names) == 0:
                subsets_names = list(range(n_subsets))
            logger.debug("subsets_names: {}.".format(subsets_names))

            buckets = deque([deque() for _ in range(n_subsets)])

            logger.debug("subsets_criteria: {}.".format(subsets_criteria))
            if subsets_criteria == "z":
                zs = pts[:, 2]
                min_z, max_z = np.min(zs), np.max(zs)
                bins = np.digitize(zs, np.linspace(min_z, max_z+1, n_subsets))
            elif subsets_criteria == "z2":
                # for this criteria, we will be spliting in z and if nodes are left/right
                # side of the plane with respect to the ZAXIS. Therefore we must have equal
                # number of buckets for both sides so that we can easily split by z distance
                assert n_subsets % 2 == 0, AssertionError(
                    "n_subsets must be even when z2 is selected")

                # set values to be digitized. The final values will have the similar meaning
                # as a 2D view on xz as: y = sign(x)*y. This method allows for quickly
                # approximate left and right sides of the plane without the need to sort
                # note: y values must be positive.
                zs = pts[:, 2]
                angle_y = angle_between(np.cross(self._Z, normal)[
                                        :2], self._Y[:2], check_orientation=False)

                if abs(angle_y) <= np.radians(45) or abs(angle_y) >= np.radians(135):
                    ys = pts[:, 1]
                    arr = np.sign(ys)*(zs-np.min(zs))
                else:
                    xs = pts[:, 0]
                    arr = np.sign(xs)*(zs-np.min(zs))
                # get ranges of of modified z-axis and compute bins
                min_z, max_z = np.min(arr), np.max(arr)
                bins = np.digitize(arr, np.linspace(
                    min_z, max_z+1, n_subsets+1))

                # The previous method works for most scenarios. However, there are some
                # limitation when deciding which bin the bottom nodes belong to.
                # Let's check if any modification needs to be made. The heuristic will
                # be based on the distance between a given node and the median value of
                # its bin and the opposite bin. If the distance between the opposite bin
                # is less than of the one to its bin's median, it probably should be
                # in the opposite bin.

                # simply get the left and right ids (bucket idexes) of bottom bins
                # They sould be the two middle bins.
                right_id = n_subsets//2
                left_id = n_subsets//2+1

                # get indexes of right and left ids (so that we can refer to them later)
                right_idexes = np.argwhere(bins == right_id).reshape((-1,))
                left_idexes = np.argwhere(bins == left_id).reshape((-1,))

                # get right and left pts
                right_pts = pts[right_idexes][:, :2]
                left_pts = pts[left_idexes][:, :2]

                # compute median of each bin
                right_median = np.median(right_pts, axis=0)
                left_median = np.median(left_pts, axis=0)

                # compute the distance from each node to it's current bin's median
                d_r_to_m = np.linalg.norm((right_pts-right_median), axis=1)
                d_l_to_m = np.linalg.norm((left_pts-left_median), axis=1)

                # compute the distance from each node to the opposite bin's median
                d_r_to_l = np.linalg.norm((right_pts-left_median), axis=1)
                d_l_to_r = np.linalg.norm((left_pts-right_median), axis=1)

                # For each distance, if the distance between the opposite bin and the
                # current bin is shorter than the one from the current bin, the node
                # belongs to the opposite bin.
                for i, (drm, drl) in enumerate(zip(d_r_to_m, d_r_to_l)):
                    if drl < drm:
                        bins[right_idexes[i]] = left_id
                for i, (dlm, dlr) in enumerate(zip(d_l_to_m, d_l_to_r)):
                    if dlr < dlm:
                        bins[left_idexes[i]] = right_id
            elif subsets_criteria == "angles":
                # unit_vecs = unit_vector_arr((pts-spk_center).T)
                vecs = pts - np.mean(pts, axis=0)
                # angles = np.array([angle_between_2D(v[:2], XAXIS[:2]) for v in vecs])
                angles = angle_between(vecs, self._X, check_orientation=True)
                min_a, max_a = np.min(angles), np.max(angles)
                logger.debug("min_a, max_a: {}, {}.".format(min_a, max_a))
                bins = np.digitize(
                    angles, np.linspace(0, 2*np.pi, n_subsets+1))
                logger.debug("bins: {}.".format(np.unique(bins)))
            else:
                raise ValueError(
                    "Unknown subset criteria. Valid options are: 'z', 'z2', or 'angles'")

            # add ids to each bucket
            for i, pool_idx in enumerate(bins):
                buckets[pool_idx-1].append(valid_ids[i])

            non_empty_buckets = deque()
            for i, pool in enumerate(buckets):
                if len(pool) > 0:
                    non_empty_buckets.append(buckets[i])

            logger.debug("len(non_empty_buckets): {}.".format(
                len(non_empty_buckets)))

            for subname, sub_ids in zip(subsets_names, non_empty_buckets):
                logger.debug("Subname: {}".format(subname))

                self.speckles.append(
                    subset=subname,
                    name=name,
                    group=group,
                    collection=collection,
                    t=t,
                    k=k,
                    mask=sub_ids,
                    elmask=None,
                    ids=sub_ids,
                    normal=normal,
                    center=spk_center
                )

        return self.speckles.get(
            spk_name=name,
            spk_group=group,
            spk_collection=collection,
            t=t
        )

    def get_speckles(self, **kwargs):
        """ Returns a specified datatype of speckles """
        return self.speckles.get(**kwargs)

    def set_region_from_speckles(self, region_key, region_vals=None, **kwargs):
        spks = self.get_speckles(**kwargs)
        if len(spks) == 0:
            raise ValueError("No valid speckles found")
        if region_vals is not None:
            if len(region_vals) != len(spks):
                raise ValueError(
                    "The number of region values must match number of spks.")

        region = np.zeros(self.mesh.n_points, dtype=np.int64)

        for i, spk in enumerate(spks):
            nodeids = spk.ids
            if region_vals is None:
                region[nodeids] = i + 1  # zero is reserved to 'non-region'
            else:
                region[nodeids] = region_vals[i]

        return self.set_region_from_mesh_ids(region_key, region)

    def check_spk(self, spk):
        return isinstance(spk, Speckle)