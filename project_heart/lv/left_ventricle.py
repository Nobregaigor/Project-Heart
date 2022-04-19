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

    # ===============================
    # Spk computations
    # ===============================

    def radial_shortening(self,
                          spks,
                          t_es: float = None,
                          t_ed: float = 0.0,
                          recompute=False,
                          **kwargs) -> float:

        # check if RS was computed
        if not self.states.check_key(self.STATES.RS) or recompute:
            self._compute_metric_from_spks(spks,
                                           self.compute_spk_radial_shortening,
                                           self.STATES.RS,
                                           t_ed=t_ed,
                                           **kwargs)
        return self.states.get(self.STATES.RS, t=t_es)

    def wall_thickening(self,
                        endo_spks,
                        epi_spks,
                        t_es: float = None,
                        t_ed: float = 0.0,
                        recompute=False,
                        **kwargs) -> float:
        # check if wt was computed
        if not self.states.check_key(self.STATES.WT) or recompute:
            self._compute_metric_from_coupled_spks(
                endo_spks,
                epi_spks,
                self.compute_spk_thickening,
                self.STATES.WT,
                t_ed=t_ed,
                **kwargs)
        return self.states.get(self.STATES.WT, t=t_es)

    def longitudinal_strain(self,
                            spks,
                            t_es: float = None,
                            t_ed: float = 0.0,
                            recompute=False,
                            **kwargs) -> float:
        # check if SL was computed
        if not self.states.check_key(self.STATES.SL) or recompute:
            self._compute_metric_from_spks(spks,
                                           self.compute_spk_longitudinal_strain,
                                           self.STATES.SL,
                                           t_ed=t_ed,
                                           **kwargs)
        return self.states.get(self.STATES.SL, t=t_es)

    def circumferential_strain(self,
                               spks,
                               t_es: float = None,
                               t_ed: float = 0.0,
                               recompute=False,
                               **kwargs) -> float:
        # check if SC was computed
        if not self.states.check_key(self.STATES.SC) or recompute:
            self._compute_metric_from_spks(spks,
                                           self.compute_spk_circumferential_strain,
                                           self.STATES.SC,
                                           t_ed=t_ed,
                                           **kwargs)
        return self.states.get(self.STATES.SC, t=t_es)

    def rotation(self,
                 spks,
                 t_es: float = None,
                 t_ed: float = 0.0,
                 recompute=False,
                 **kwargs) -> float:
        # check if RO was computed
        if not self.states.check_key(self.STATES.RO) or recompute:
            self._compute_metric_from_spks(spks,
                                           self.compute_spk_radial_shortening,
                                           self.STATES.RO,
                                           t_ed=t_ed,
                                           **kwargs)
        return self.states.get(self.STATES.RO, t=t_es)

    def twist(self,
              apex_spks,
              base_spks,
              t_es: float = None,
              t_ed: float = 0.0,
              recompute=False,
              **kwargs) -> float:
        # check if tw was computed
        if not self.states.check_key(self.STATES.TW) or recompute:
            self._compute_metric_from_coupled_spks(
                apex_spks,
                base_spks,
                self.compute_spk_twist,
                self.STATES.TW,
                t_ed=t_ed,
                **kwargs)
        return self.states.get(self.STATES.TW, t=t_es)

    def torsion(self,
                apex_spks,
                base_spks,
                t_es: float = None,
                t_ed: float = 0.0,
                recompute=False,
                **kwargs) -> float:
        # check if TO was computed
        if not self.states.check_key(self.STATES.TO) or recompute:
            self._compute_metric_from_coupled_spks(
                apex_spks,
                base_spks,
                self.compute_spk_torsion,
                self.STATES.TO,
                t_ed=t_ed,
                **kwargs)
        return self.states.get(self.STATES.TO, t=t_es)

    # ===============================
    # Plots
    # ===============================

    def plot_metric(self, metric: str, kind="scatter", **kwargs):
        metric = self.check_enum(metric)
        if not self.states.check_key(metric):
            raise ValueError(
                "Metric '{}' not found in states." \
                "Did you compute it?".format(metric))
        data = self.states.get(metric)
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame({metric: data, "timesteps": self.states.timesteps})
        df.plot(x="timesteps", y=metric, kind=kind, **kwargs)
        plt.show()

        