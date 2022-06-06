import logging
import os
from pathlib import Path
import numpy as np
from collections import deque

from project_heart.modules.speckles import SpecklesDict, SpeckleStates, Speckle
from project_heart.modules.speckles.speckle import SpeckeDeque
from project_heart.utils.vector_utils import *
from project_heart.utils.spatial_utils import *
from project_heart.utils.spatial_points import *
from project_heart.utils.cloud_ops import *
from project_heart.enums import *
from .lv_region_identifier import LV_RegionIdentifier

from project_heart.utils.assertions import assert_iterable

from sklearn.cluster import KMeans
from functools import reduce


# logging.basicConfig()
logger = logging.getLogger('LV.Speckles')


default_lvSpeckles_enums = {
    "SPK_SETS": LV_SPK_SETS,
    "SPK_STATES": LV_SPK_STATES
}

class LV_Speckles(LV_RegionIdentifier):
    """_summary_
    """

    def __init__(self, enums={}, *args, **kwargs):
        super(LV_RegionIdentifier, self).__init__(*args, **kwargs)
        self.speckles = SpecklesDict()
        self.states = SpeckleStates()

        # ---- Enums
        self.SPK_SETS = LV_SPK_SETS
        self.SPK_STATES = LV_SPK_STATES
        
        if len(enums) > 0:
            self.config_enums(enums, check_keys=default_lvSpeckles_enums.keys())
        
        
    def create_speckles(self,
                        name=None,
                        group=None,
                        collection=None,
                        from_nodeset=None,
                        exclude_nodeset=None,
                        use_all_nodes=False, # will ignore 'd', k, kmin, kmax -- NEED TO IMPLEMENT
                        normal_to=None,
                        perpendicular_to=None,
                        d=3.0,
                        k=1.0,
                        kmin=None,
                        kmax=None,
                        extrapolate_k=False,
                        n_subsets=0, 
                        subsets_criteria=None,
                        subsets_names=None,
                        t=0.0,
                        n_clusters=0,
                        cluster_criteria=None,
                        log_level=logging.WARNING,
                        ignore_unmatch_number_of_clusters=True,
                        **kwargs
                        ):
        """
            Creates Speckles
        """

        # set logger
        logger = logging.getLogger('create_speckles')
        logger.setLevel(log_level)

        logger.info("Speckle: name: {}, group: {}, collection: {}"
                    .format(name, group, collection))

        # apply checks
        if n_subsets > 0:
            assert subsets_criteria is not None, ("If speckle subsets is requested, "
                "must provide subset criteria")
            if subsets_names is None:
                subsets_names = list()
            if len(subsets_names) > 0:
                assert len(subsets_names) == n_subsets, AssertionError(
                    "If list of prefixes was provided, its length must be "
                    "equal to the number of subsets.")
            elif len(subsets_names) == 0:
                subsets_names = list(range(n_subsets))
        
        # Assume default values
        cluster_criteria = subsets_criteria if cluster_criteria is None else cluster_criteria
        if n_clusters > 0:
            assert cluster_criteria is not None, ("If speckle cluster is requested, "
                    "must provide subset criteria. No default value was found.")
        
        # ------ Resolve nodes to use as reference ---------------------
        # determine nodes to use (entire mesh or specified nodeset)
        if from_nodeset is None: 
            logger.debug("Using all avaiable nodes in mesh.")
            # keep track of nodes data
            nodes = self.nodes()
            # keep track of nodes ids
            ids = np.arange(1, len(nodes) + 1, dtype=np.int32)
        else:
            logger.debug("Using nodes from nodeset %s" % from_nodeset)
            ids = self.get_nodeset(from_nodeset)
            nodes = self.nodes(mask=ids)
        # exclude nodes from other nodeset
        if exclude_nodeset is not None:
            logger.debug("Excluding nodes from nodeset %s" % exclude_nodeset)
            if isinstance(exclude_nodeset, (list, tuple)):
                for exc_ndset in exclude_nodeset:
                    ids_to_exclude = self.get_nodeset(exc_ndset)
                    ids = np.setdiff1d(ids, ids_to_exclude)
                    nodes = self.nodes(mask=ids)
            else:
                ids_to_exclude = self.get_nodeset(exclude_nodeset)
                ids = np.setdiff1d(ids, ids_to_exclude)
                nodes = self.nodes(mask=ids)
            # check for possible errors when exluding nodes (empty)
            if len(ids) == 0:
                raise RuntimeError("No aviable nodes found after excluding nodeset(s) '{}'."
                                   .format(exclude_nodeset))
        # ------  get longitudinal line -----------------------------------
        long_line = self.get_long_line()
        logger.debug("long_line: {}".format(long_line))

        # ------- Compute speckle LA center --------------------------------
        # LA center defines speckle position along longitudinal axis.
        # Determine point along longitudinal line to be used as reference
        if extrapolate_k is True:
            assert k >= -0.2 and k <= 1.12, (
                "For safety, extrapolation is not allowed for k values larger than 10%. "
                "k range is [-0.2, 1.2]. k received: {}".format(k))
        p = get_p_along_line(k, long_line, extrapolate=extrapolate_k)
        spk_center = p
        logger.debug("ref p (spk center): {}".format(p))
        
        
        # ----- Resolve nodes and speckle normal -------------------------
        if not use_all_nodes:
            logger.debug("Using nodes close to plane [k={}] at p={} with d={}."
                         .format(k,p,d))
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
                    
            # get points close to plane at height k (from p) and threshold d
            ioi = get_pts_close_to_plane(
                nodes, d, normal, p)
            pts = nodes[ioi]
            logger.debug("pts close to plane: {}".format(len(pts)))
        else:
            logger.debug("Using all nodes in mesh OR nodeset (not computing plane).")
            pts = nodes
            normal, _ = fit_plane(pts)
            ioi = np.arange(0, len(pts), dtype=np.int32)

        # ------- Check for k boundaries --------------------------------
        # kmin and kmax limit speckle regions based on apex and base
        adjusted_due_to_k_bounds = False # this flag is used for debug purposes
        if kmin is not None:
            assert isinstance(kmin, (int, float)), "kmin must be an integer or float."
            if kmin != -1.0:
                adjusted_due_to_k_bounds = True
                logger.debug(
                    "Adjusting avaiable nodes based on kmin: {}".format(kmin))
                p = get_p_along_line(kmin, long_line, extrapolate=extrapolate_k)
                ioi = np.setdiff1d(ioi, np.where(nodes[:, 2] < p[2]))
                pts = nodes[ioi]
        if kmax is not None:
            assert isinstance(kmax, (int, float)), "kmax must be an integer or float."
            if kmax != -1.0:
                adjusted_due_to_k_bounds = True
                logger.debug(
                    "Adjusting avaiable nodes based on kmax: {}".format(kmax))
                p = get_p_along_line(kmax, long_line, extrapolate=extrapolate_k)
                ioi = np.setdiff1d(ioi, np.where(nodes[:, 2] > p[2]))
                pts = nodes[ioi]
        # show debug text if adjustment occured.
        if adjusted_due_to_k_bounds:
            logger.debug(
                "New spk center and pts found after adjustment.")
            logger.debug("ref p (spk center): {}".format(p))
            logger.debug("pts close to plane: {}".format(len(pts)))
        # check number of points found.
        if len(pts) == 0:
            raise RuntimeError("Found number of points is zero. Try checking kmin and kmax.")

        # --------------------------------
        
        valid_ids = ids[ioi]  # get valid ids
        mask = ioi   # global mask
        
        if n_subsets <= 1:
            logger.debug("Adding single subset.")
            
            k_ids = None
            non_empty_buckets_l = None
            if n_clusters > 0:
                sub_pts = self.nodes()[valid_ids]
                k_ids, non_empty_buckets_l = self._subdivide_speckles(sub_pts, n_clusters, 
                                                        cluster_criteria, 
                                                        normal, valid_ids,
                                                        spk_center,
                                                        True, #check for orientation
                                                        log_level,
                                                        ignore_unmatch_number_of_clusters)
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
                center=spk_center,
                k_ids=k_ids,
                k_local_ids=non_empty_buckets_l
            )
        else:
            logger.debug("pts: {}".format(len(pts)))
            # cehck if subset names was provided
            logger.debug("Adding multiple subsets: {} -> {}"
                         .format(n_subsets, subsets_names))
            # apply speckle subdivision (smart clustering)
            non_empty_buckets, _ = self._subdivide_speckles(pts, n_subsets, 
                                                         subsets_criteria, 
                                                         normal, valid_ids,
                                                         spk_center,
                                                         False, # dont check for orientation
                                                         log_level)
                        
            for subname, sub_ids in zip(subsets_names, non_empty_buckets):
                logger.debug("Subname: {}".format(subname))

                k_ids = None
                non_empty_buckets_l = None
                if n_clusters > 0:
                    logger.debug("len(pts): {}".format(len(pts)))
                    logger.debug("sub_ids[:5]: {}".format(np.array(sub_ids)[:5]))
                    sub_pts = self.nodes()[sub_ids]
                    k_ids, non_empty_buckets_l= self._subdivide_speckles(sub_pts, 
                                                     n_clusters,
                                                     cluster_criteria, 
                                                     normal, sub_ids,
                                                     spk_center,
                                                     False, # dont check for orientation
                                                     log_level,
                                                     ignore_unmatch_number_of_clusters,
                                                     )
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
                    center=spk_center,
                    k_ids=k_ids,
                    k_local_ids=non_empty_buckets_l
                )

        return self.speckles.get(
            spk_name=name,
            spk_group=group,
            spk_collection=collection,
            t=t
        )

    def _subdivide_speckles(self, 
                            pts, 
                            n_subsets, 
                            subsets_criteria, 
                            normal, 
                            valid_ids,
                            ref_center, # used for angles
                            check_orientation=False,
                            log_level=logging.WARNING,
                            ignore_unmatch_number_of_clusters=False):
        logger = logging.getLogger('_subdivide_speckles')
        logger.setLevel(log_level)
        logger.debug("Subdividing Speckles into '{}' buckets based on subsets_criteria: '{}'."
                     .format(n_subsets, subsets_criteria))
        
        # create buckets        
        buckets = deque([deque() for _ in range(n_subsets)])   # holds global ids (within valid_ids -> mesh)
        buckets_l = deque([deque() for _ in range(n_subsets)]) # holds local ids (within spk)
        logger.debug("Number of buckets: {}.".format(len(buckets)))
        # selecte subdivition based on subsets_criteria
        if subsets_criteria == "z": # regular 'z' axis subdvision.
            zs = pts[:, 2]
            min_z, max_z = np.min(zs), np.max(zs)
            bins = np.digitize(zs, np.linspace(min_z-1, max_z+1, n_subsets+1))
        elif subsets_criteria == "z2": # complex 'z' axis subdvision.
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
            angle_y = angle_between(np.cross(self._Z, normal)[:2], self._Y[:2], check_orientation=False)

            if abs(angle_y) <= np.radians(45) or abs(angle_y) >= np.radians(135):
                ys = pts[:, 1]
                arr = np.sign(ys)*(zs-np.min(zs))
            else:
                xs = pts[:, 0]
                arr = np.sign(xs)*(zs-np.min(zs))
            # get ranges of of modified z-axis and compute bins
            min_z, max_z = np.min(arr), np.max(arr)
            bins = np.digitize(arr, np.linspace(
                min_z-1, max_z+1, n_subsets+1))

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
        elif subsets_criteria == "angles": # xy-plane angles subdvisiton.
            vecs = pts - ref_center#np.mean(pts, axis=0)
            angles = angle_between(vecs, self._X, check_orientation=True)
            bins = np.digitize(
                angles, np.linspace(-0.0001, 2*np.pi*1.001, n_subsets+1))
            logger.debug("Unique bins: {}.".format(np.unique(bins)))
        elif subsets_criteria == "angles2": # xy-plane angles subdvisiton.
            vecs = pts - ref_center #centroid(pts)
            angles = angle_between(vecs, self._X, check_orientation=True)
            min_a, max_a = np.min(angles), np.max(angles)
            logger.debug("min_a, max_a: {}, {}.".format(min_a, max_a))
            bins = np.digitize(
                angles, np.linspace(min_a*0.999, max_a*1.001, n_subsets+1))
            logger.debug("Unique bins: {}.".format(np.unique(bins)))
        elif subsets_criteria == "angles3": 
            if normal[2] < 0:
                    normal = -normal
                    logger.debug("using opposite normal for computation as negative "
                                "(on z) axis are not fully tested: {}.".format(normal))
            # project points onto a single plane based on spk normal
            logger.debug("pts.shape: {}.".format(pts.shape))
            logger.debug("pts[:5]: \n{}.".format(pts[:5]))
            plane_d = calc_plane_d(normal, ref_center)
            logger.debug("plane_d: {}.".format(plane_d))
            ppts = project_pts_onto_plane(pts, normal, plane_d)
            p_center = project_pts_onto_plane(ref_center, normal, plane_d)[0]
            logger.debug("p_center: {}.".format(p_center))
            # compute angles between a reference vector and other vectos
            vecs = ppts - p_center 
            vecs_sum = np.linalg.norm(vecs, axis=1)
            ref_max_pt = np.argmax(vecs_sum)
            ref_min_pt = np.argmin(vecs_sum)
            logger.debug("ref_max_pt: {}.".format(ref_max_pt))
            logger.debug("ref_max_pt: {}.".format(ref_min_pt))
            ref_max_vec = vecs[ref_max_pt]
            ref_min_vec = vecs[ref_min_pt]
            logger.debug("ref_max_vec: {}.".format(ref_max_vec))
            logger.debug("ref_min_vec: {}.".format(ref_min_vec))
            # compute angles based on 
            if check_orientation:
                logger.debug("Using a none or single subset, meaning large speckle. \n" 
                            "-> 'check_orientation' is set to True."
                            "-> Will try to eliminate possible discontinuities.")
                angles = angle_between(vecs, ref_max_vec, check_orientation=True, zaxis=normal)    
                search_for_jumps = True
                search_trial = 0
                max_search_trials = 20
                dist_step = 0.25
                while search_for_jumps:
                    logger.debug("Searching for discontinuity... {}".format(search_trial+1))
                    sort_ids = np.argsort(angles)
                    ppts_sorted_by_angles = ppts[sort_ids]
                    dists_ppts = np.linalg.norm(ppts_sorted_by_angles[1:] - ppts_sorted_by_angles[:-1], axis=1)
                    dists_grad = np.gradient(dists_ppts)
                    d_mean, d_stdev = np.mean(dists_grad), np.std(dists_grad)
                    dist_jump = np.where(np.abs(dists_grad - d_mean) > (4 + (search_trial*dist_step)) *d_stdev)[0]
                    if len(dist_jump) > 0:
                        logger.debug("Discontinuity Found: {}".format(len(dist_jump)))
                        dist_jump_val = dists_grad[dist_jump]
                        logger.debug("dist_jump: {} -> {}.".format(dist_jump, dist_jump_val))
                        if (search_trial % 2) == 0:
                            look_at = 2
                        else:
                            look_at = 0
                        idx_to_look_at =dist_jump[np.argmax(dist_jump_val)]+look_at
                        if idx_to_look_at >= len(sort_ids):
                            idx_to_look_at = len(sort_ids) - idx_to_look_at
                        new_max_pt = sort_ids[idx_to_look_at]
                        ref_max_vec = vecs[new_max_pt]
                        ref_max_pt = new_max_pt
                        angles = angle_between(vecs, ref_max_vec, check_orientation=True, zaxis=normal)
                    else:
                        logger.debug("No discontinuity found: {}".format(len(dist_jump)))
                        search_for_jumps = False
                    if search_trial >= max_search_trials:
                        logger.debug("Could not resolve jumps under {} trials. \n"
                                    "-> Continuing (errors might occur)".format(max_search_trials))
                        search_for_jumps = False
                    search_trial += 1
            else:
                logger.debug("Using angles without orientation [0, 180]")
                angles = angle_between(vecs, ref_max_vec, check_orientation=False)
            # get reference vecs for binarization bounds
            min_a, max_a = np.min(angles), np.max(angles)
            logger.debug("min_a: {}.".format(np.degrees(min_a)))
            logger.debug("max_a: {}.".format(np.degrees(max_a)))
            min_idx, max_idx = np.argmin(angles), np.argmax(angles)
            ref_max_vec = vecs[max_idx]
            ref_min_vec = vecs[min_idx]
            logger.debug("Updating reference vectors.")
            logger.debug("ref_max_vec: {}.".format(ref_max_vec))
            logger.debug("ref_min_vec: {}.".format(ref_min_vec))
            # create bins
            bins = np.digitize(
                angles, np.linspace(min_a*0.999, max_a*1.001, n_subsets+1))
            logger.debug("Number of bins found: {}.".format(len(np.unique(bins))))
            min_bin_id, max_bin_id = np.min(bins), np.max(bins)
            logger.debug("Min, Max bin Ids: [{}, {}].".format(min_bin_id, max_bin_id))
            # update reference vectors (make sure we have extreme pts)
            min_mask = np.where(bins==min_bin_id)[0]
            max_mask = np.where(bins==max_bin_id)[0]
            ref_last_bin_idx = max_mask[np.argmax(angles[max_mask])]
            ref_first_bin_idx = min_mask[np.argmin(angles[min_mask])]
            logger.debug("ref_last_bin_idx: {}.".format(ref_last_bin_idx))
            logger.debug("ref_first_bin_idx: {}.".format(ref_first_bin_idx))
            ref_last_vec = vecs[ref_last_bin_idx]
            ref_first_vec = vecs[ref_first_bin_idx] 
            logger.debug("ref_last_vec: {}.".format(ref_last_vec))
            logger.debug("ref_first_vec: {}.".format(ref_first_vec))
            # compute vector normal and cross vectors
            first_cross_last = np.cross(unit_vector(ref_first_vec), unit_vector(ref_last_vec)) 
            cross_normal = np.degrees(angle_between(normal, first_cross_last, check_orientation=False)) 
            first_last_angle = np.degrees(angle_between(ref_first_vec, ref_last_vec, check_orientation=True, zaxis=normal))
            logger.debug("normal: {}.".format(normal))
            logger.debug("first_cross_last: {}.".format(first_cross_last))
            logger.debug("cross_normal: {}.".format(cross_normal))
            logger.debug("first_last_angle: {}.".format(first_last_angle))
            # recompute bins to ensure we have full range (min to max vecs)
            # a_normal_Zaxis = np.degrees(angle_between(normal, [0.0,0.0,1.0], check_orientation=False))
            if not check_orientation:# and a_normal_Zaxis < 45: 
                # logger.debug("Re-computing bins as angle from normal to Z axis < 45: {}.".format(a_normal_Zaxis))
                logger.debug("Recomputing bins based on new vecs.")
                angles = angle_between(vecs, ref_last_vec, check_orientation=False)
                min_a, max_a = np.min(angles), np.max(angles)
                logger.debug("min_a: {}.".format(np.degrees(min_a)))
                logger.debug("max_a: {}.".format(np.degrees(max_a)))
                bins = np.digitize(
                    angles, np.linspace(min_a*0.999, max_a*1.001, n_subsets+1))
                # check gradients between gradient centers -> angle between first to other bins
                # we want ccw bins                
                centers = np.asarray([np.mean(ppts[bins==bin_id], axis=0) for bin_id in sorted(np.unique(bins))], dtype=np.int64)
                c_vecs = centers - p_center
                c_angles = np.degrees(angle_between(c_vecs, c_vecs[0], check_orientation=True, zaxis=normal))
                c_grads = np.gradient(c_angles[1:])
                mean_c_grads = np.mean(c_grads)
                logger.debug("mean_c_grads: {}.".format(mean_c_grads))
                # check for reversion
                should_reverse = False
                a_normal_Xaxis = np.round(np.degrees(angle_between(normal, [1.0,0.0,0.0], check_orientation=False)), 3)
                if a_normal_Xaxis < 45:
                    should_reverse = not should_reverse
                    logger.debug("Flip should_reverse due normal orientation -> [{}]: a_normal_Xaxis: {}.".format(should_reverse, a_normal_Xaxis))   
                # a_normal_Zaxis = np.round(np.degrees(angle_between(normal, [0.0,0.0,1.0], check_orientation=False)), 3)
                # if a_normal_Zaxis < 45:
                #     should_reverse = not should_reverse
                #     logger.debug("Flip should_reverse due normal orientation -> [{}]: a_normal_Zaxis: {}.".format(should_reverse, a_normal_Zaxis))   
                # if cross_normal < 90 and mean_c_grads > 0:
                #     should_reverse = not should_reverse
                #     logger.debug("bins should reverse [{}] as cross_normal < 90 and mean_c_grads > 0.".format(should_reverse))
                # if cross_normal < 90 and first_last_angle > 180:
                #     should_reverse = not should_reverse
                #     logger.debug("bins should reverse [{}] as cross_normal < 0 and first_last_angle > 180.".format(should_reverse))
                # elif cross_normal > 90 and first_last_angle > 180: #:
                #     should_reverse = not should_reverse
                #     logger.debug("bins should reverse [{}] as cross_normal > 0, first_last_angle > 180 and mean_c_grads > 0.".format(should_reverse))
                if first_last_angle > 180:
                    should_reverse = not should_reverse
                    logger.debug("bins should reverse [{}] as first_last_angle > 180 ".format(should_reverse))
                # if algorithm decided to reverse, reverse bins
                if should_reverse:
                    logger.debug("reversing bins.")
                    uvals = np.unique(bins)
                    logger.debug("uvals: {}.".format(uvals))
                    new_bins = np.zeros(len(bins), dtype=np.int64)
                    for i, u in enumerate(uvals[::-1]):
                        new_bins[bins==u] = i
                    bins = new_bins + 1
            
        elif subsets_criteria == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_subsets, n_init=5, random_state=0)
            bins = kmeans.fit_predict(pts)+1
        else:
            raise ValueError(
                "Unknown subset criteria. Valid options are: 'z', 'z2', or 'angles'")

        # add ids to each bucket
        for i, pool_idx in enumerate(bins):
            buckets[pool_idx-1].append(valid_ids[i])
            buckets_l[pool_idx-1].append(i)
            
        # check for valid buckets (must not be empty)
        non_empty_buckets = deque()
        non_empty_buckets_l = deque()
        for i, pool in enumerate(buckets):
            if len(pool) > 0:
                non_empty_buckets.append(buckets[i])
                non_empty_buckets_l.append(buckets_l[i])
                
        logger.debug("Number of buckets found (non_empty_buckets): {}.".format(
            len(non_empty_buckets)))
        
        if len(non_empty_buckets) != len(buckets) and not ignore_unmatch_number_of_clusters:
            logger.warn("Found empty buckels when performing "
                        "speckle subdivition, which might lead "
                        "to unexpected number of subsets or k_ids. "
                        "You may want to tweak speckle parameters. "
                        "Expected: '{}', Found: '{}'"
                        .format(len(buckets), len(non_empty_buckets)))
        
        return non_empty_buckets, non_empty_buckets_l
        
    def create_speckles_from_iterable(self, items):
        assert_iterable(items, False)
        for i, spk_args in enumerate(items):
            try:
                self.create_speckles(**spk_args)
            except:
                logger.error("Failed to create Speckles at id '{}'."
                "with the following arguments: \n{}.".format(i, spk_args))

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

    def get_speckles_xyz(self, spk_args, t=None) -> np.ndarray:
        spks_ids = self._resolve_spk_args(spk_args).stack_ids()
        if self.states.check_key(self.STATES.XYZ):
            xyz = self.states.get(self.STATES.XYZ, t=t, mask=spks_ids)
        else:
            xyz = self.nodes(mask=spks_ids)
        return xyz
    
    def get_speckles_centers(self, spk_args, t=None) -> np.ndarray:
        spk_deque = self._resolve_spk_args(spk_args)
        centers = deque()
        if self.states.check_key(self.STATES.XYZ):
            xyz = self.states.get(self.STATES.XYZ, t=t)
        else:
            xyz = self.nodes()
        for spk in spk_deque:
            centers.append(np.mean(xyz[spk.ids], axis=0))
        return np.vstack(centers)
    
    def get_speckles_la_centers(self, spk_args, t=None) -> np.ndarray:
        
        spk_deque = self._resolve_spk_args(spk_args)
        centers = deque()
        for spk in spk_deque:
            if not self.states.check_spk_key(spk, self.STATES.CENTERS):
                self.compute_spk_centers_over_timesteps(spk)
            c = self.states.get_spk_data(spk, self.STATES.CENTERS, t=t)
            centers.append(c)
        return np.vstack(centers)
    
    def get_speckles_k_centers(self, spk_args, t=None, **kwargs) -> np.ndarray:
        
        if self.states.check_key(self.STATES.XYZ) and t is not None:
            xyz = self.states.get(self.STATES.XYZ, t=t)
        else:
            xyz = self.nodes()
            
        spk_deque = self._resolve_spk_args(spk_args)
        centers = deque() 
        for spk in spk_deque:
            for kids in spk.k_ids:
                centers.append(np.mean(xyz[kids], axis=0))
        
        centers = np.asarray(centers)
        from project_heart.utils.spatial_utils import apply_filter_on_line_segment
        centers = apply_filter_on_line_segment(centers, **kwargs)
        return np.vstack(centers)

    def check_spk(self, spk):
        return isinstance(spk, Speckle)

    
    def _resolve_spk_args(self, spk_args):
        from collections.abc import Iterable
        if isinstance(spk_args, dict):
            spks = self.get_speckles(**spk_args)
            if len(spks) == 0:
                raise ValueError("No spks found for given spk_args: {}".format(spk_args))
        elif isinstance(spk_args, Iterable):
            if not issubclass(spk_args.__class__, SpeckeDeque):
                spks = SpeckeDeque([spk_args])
            else:
                spks = spk_args
        elif issubclass(spk_args.__class__, Speckle):
            spks = SpeckeDeque([spk_args])
        else:
            raise TypeError("'spks' must be a Iterable of spk objects."\
                            "The respective function argument can be either a "\
                            "dictionary containing 'get_speckles' args or one "\
                            "of the following types: 'list', 'tuple', 'np.ndarray'. "
                            "Received type: {}".format(type(spk_args))
                            )
        return spks