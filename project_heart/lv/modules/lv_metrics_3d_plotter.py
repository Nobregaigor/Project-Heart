import numpy as np
from .lv_metrics_geometrics_computations import LVGeometricsComputations
# from project_heart.enums import CONTAINER, STATES, LV_SURFS
from project_heart.utils.spatial_utils import centroid, radius
from project_heart.modules.speckles.speckle import Speckle, SpeckeDeque
from project_heart.utils.enum_utils import add_to_enum

from project_heart.utils.extended_classes import ExtendedDict

import logging

logger = logging.getLogger('LV3DMetricsPlotter')

from collections import deque


class LV3DMetricsPlotter(LVGeometricsComputations):
    def __init__(self, log_level=logging.INFO, *args, **kwargs):
        super(LV3DMetricsPlotter, self).__init__(log_level=log_level, *args, **kwargs)
        self.EPSILON = 1e-10
        self.explainable_metrics = ExtendedDict() # stores methods and approach used for metric computations
        logger.setLevel(log_level)
    
    def _reolve_window_size(self, ws):
        if ws is None:
            ws = (600,400)
        return ws

    def _get_metrics_3D_plotter(self, t=0.0, window_size=None, **kwargs):
        window_size = self._reolve_window_size(window_size)
            
        default_kwargs = dict(
                style='points', 
                color="gray", 
                opacity=0.6,
                window_size=window_size
            )
        default_kwargs.update(kwargs)
        #  plotmesh and virtual nodes
        return self.plot(t=t, re=True, **default_kwargs)
    
    def _resolve_plotter(self, plotter, plot_kwargs, t=0.0):
        # resolve plotter
        if plotter is None:
            if plot_kwargs is None:
                plot_kwargs = dict()
            plotter = self._get_metrics_3D_plotter(t=t, **plot_kwargs)
        return plotter
        
    def plot_longitudinal_line(self, 
                               t=0.0,
                               re=False, 
                               line_kwargs=None,
                               points_kwargs=None,
                               plot_kwargs=None, 
                               plotter=None,
                               window_size=None,
                               **kwargs):
        # resolve plotter and window size
        window_size = self._reolve_window_size(window_size)
        plotter = self._resolve_plotter(plotter, plot_kwargs, t=t)
        # get apex and base at given timestep
        apex = self.states.get(self.STATES.APEX_REF, t=t)
        base = self.states.get(self.STATES.BASE_REF, t=t)
        # create long line for plot
        from project_heart.utils import lines_from_points
        line = lines_from_points((apex, base))
        # set default args for long line and update if user provided
        d_kwargs = dict(color="cyan")
        if line_kwargs is not None:
            d_kwargs.update(kwargs)
        # add line mesh
        plotter.add_mesh(line, **d_kwargs)
        # add points
        d_kwargs = dict(color="orange", point_size=400)
        if points_kwargs is not None:
            d_kwargs.update(points_kwargs)
        plotter.add_points(np.vstack([apex, base]), **d_kwargs)
        
        # if requested, return plotter
        if re:
            return plotter
        # if not requested, show plot
        else:
            plotter.show(window_size=window_size)
    
    
    # metrics
    
    def _resolve_exm(self, key):
        all_exm = self.explainable_metrics.all(key)
        if len(all_exm) == 0:
            raise RuntimeError("No explainable metric for longitudinal distance was found. "
                               "Did you compute it or loaded from a file? "
                               "Only metrics compute with this package can be used for 3D plot.")
        return all_exm
    
    def plot_longitudinal_distance(self, t=0.0, colors=None, 
                                   window_size=None, plot_kwargs=None, log_level=logging.INFO):
        from project_heart.utils import lines_from_points
        log = logger.getChild("plot_longitudinal_distance")
        log.setLevel(log_level)
        
        key = self.STATES.LONGITUDINAL_DISTANCE
        all_exm = self._resolve_exm(key)
        
        use_exm = all_exm.all("base")
        if len(use_exm) == 0:
            log.warn("No 'base' explainable metric was found. Will use all values found, "
                     "but lines might be duplicated during plot.")
            use_exm = all_exm
        
        # resolve plotter and window size
        window_size = self._reolve_window_size(window_size)
        plotter = self._resolve_plotter(None, plot_kwargs, t=t)
        
        if colors is None:
            colors = ["green", "orange", "brown", "blue", "red"]
        idx = self.states.get_timestep_index(t)
        apex_ts = deque()
        base_ts = deque()
        for i, exm in enumerate(use_exm.values()):
            # get apex and base at given timestep
            apex = exm.apex[idx]
            base = exm.base[idx]
            # create long line for plot
            line = lines_from_points((apex, base))
            # add line mesh
            plotter.add_mesh(line, color=colors[i])
            # add points
            plotter.add_points(np.vstack([apex, base]), color=colors[i], point_size=400)
            apex_ts.append(apex)
            base_ts.append(base)
        # reduce values
        apex_reduced = np.mean(apex_ts, axis=0)
        base_reduced = np.mean(base_ts, axis=0)
        # create long line for plot
        line = lines_from_points((apex_reduced, base_reduced))
        # add line mesh
        plotter.add_mesh(line, color="magenta")
        # add points
        plotter.add_points(np.vstack([apex_reduced, base_reduced]), color="magenta", point_size=400)
        # show plot
        plotter.show(window_size=window_size)