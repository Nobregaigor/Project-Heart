


## Metric: Radius

- Updated centers: centers was fixed based on initial spk configuration
  - New: Compute centers based on 'k' height from moving longitudinal line
         defined from apex->base at each timestep.
    - k is defined when speckle is created
    - centers are computed using 'compute_spk_centers_over_timesteps'
    - moving longitudinal line is computed using 'compute_base_apex_ref_over_timesteps'
  - Future: Add plot method for displaying moving centers and long-line


## Metric: Thickness:
- Affected by change in radius computation

