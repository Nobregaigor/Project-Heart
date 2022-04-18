from .modules import LV_FiberEstimator, LV_Speckles, LVBaseMetricsComputations
import numpy as np
# from project_heart.enums import CONTAINER, STATES, LV_SURFS


class LV(LV_FiberEstimator, LVBaseMetricsComputations):
    def __init__(self, *args, **kwargs):
        super(LV, self).__init__(*args, **kwargs)

    def xyz(self, **kwargs) -> np.ndarray:
        """Return pointer to nodal position [xyz] state array.

        Raises:
            RuntimeError: If 'xyz' could not be extracted or automatically computed based on current state's data. 

        Returns:
            np.ndarray (n_timesteps, 3): Pointer to 'xyz' states array.
        """
        if self.states.check_key(self.STATES.XYZ):
            return self.states.get(self.STATES.XYZ, **kwargs)
        else:
            try:
                return self.compute_xyz_from_displacement(**kwargs)
            except:
                raise RuntimeError(
                    "Could not retrieve 'xyz' data from states. Did you compute it? Check if states has 'displacement' data.")

    def volume(self, **kwargs):
        if self.states.check_key(self.STATES.VOLUME):
            return self.states.get(self.STATES.VOLUME)
        else:
            try:
                return self.compute_volume_based_on_nodeset(**kwargs)
            except:
                raise RuntimeError(
                    "Could not retrieve 'xyz' data from states. Did you compute it? Check if states has 'displacement' data.")

    def ejection_fraction(self,
                          t_es: float = None,
                          t_ed: float = 0.0,
                          **kwargs) -> float:
        """Calculates the change in volume between end-systole and end-diatole.
            ES and ED are timesteps where the ES and ED occur.

        Args:
            t_es (float or None, optional): End-systole timestep. Defaults to None.
            t_ed (float, optional): End-diastole timestep. Defaults to 0.0.

        Returns:
            float (if t_es is specified): Ejection fraction (%) at t_es.
            np.ndarray (n_timesteps, 1): Volumetric fraction array.
        """
        # check if vf was computed
        if not self.states.check_key(self.STATES.VF):
            self.compute_volumetric_fraction(t_ed=t_ed, **kwargs)

        return self.states.get(self.STATES.VF, t=t_es)

    def longitudinal_shortening(self,
                                t_es: float = None,
                                t_ed: float = 0.0,
                                **kwargs) -> float:

        # check if ls was computed
        if not self.states.check_key(self.STATES.LS):
            self.compute_longitudinal_shortening(t_ed=t_ed, **kwargs)

        return self.states.get(self.STATES.LS, t=t_es)

    def radial_shortening(self):
        pass

    def wall_thickening(self):
        pass

    def longitudinal_strain(self):
        pass

    def circumferential_strain(self):
        pass

    def rotation(self):
        pass

    def twist(self):
        pass

    def torsion(self):
        pass
