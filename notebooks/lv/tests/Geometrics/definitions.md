

# Geometrics and respective clinical metrics:
Here is a brief description of each geometric:

- **radial_distance**: perpendicular distance surface to normal vector (defined from longitudinal line)
  - It can be though as the "2D" radial distance on an idealized geometry.
  - Two approaches:
    - fixed_vector: uses reference normal (created based on reference configuration).
    - moving_vector (default): uses temporal normals (normal is recomputed for all timesteps).
      - This method accounts for possible distortions if geometry changes orientation
  - Used for '**radial_shortening**"

- **radial_length**: Vector length from surface to specified centers at longitudinal line.
  - It is the absolute distance from a spk to its respective center.
  - Accounts for curvatures in the geometry.
  - Two approaches:
    - fixed_centers: uses reference relative center (created based on reference configuration).
    - moving_centers (default): uses temporal centers (center is recomputed for all timesteps).
      - This method accounts for possible distortions if geometry changes orientation
  - Used for '**radial_strain**'

- **wall_thickness**: Mean distance between Epicardium and Endocardium surfaces.
  - To account for changes in curvature, we consider "radial_length" when computing the wall thickness.
  - Used for '**wall_thickening**'

- **longitudinal_distance**: Vector length from apex to base reference points.
  - It can be though as the "2D" distance between top and bottom of an idealized geometry.
  - Used for '**longitudinal_shortening**'

- **longitudinal_length**: Curvature length of the surface along the longitudinal axis.
  - Used for '**longitudinal_strain**'

- **circumferential_length**: Curvature length of the surface along the circumferential axis.
  - Used for '**circumferential_strain**'
  