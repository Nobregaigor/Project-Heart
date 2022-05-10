
# Definitions for Left Ventricle Metrics

Recognizing the different definitions and usage of echocardiographic 
quantification of the left ventricle, this section aims to provide
information about basic definitions and assumptions used in this 
project.  
Our implementation considers the usage of high-quality meshes, with
minimal noise and high-fidelity geometry, allowing for high-precision
computations, rather than simplified assumptions used in the literature.
Is important to understand that, currently, there is no
consensus regading 3D speckle tracking. Consequently, we have
decided to follow some standard practices for 2D speckle tracking 
proposed by the European Association of Echocardiography in 2015 
[[1](https://core.ac.uk/reader/53744593?utm_source=linkout)]
while also implementing different approaches that let's the user
decide how the final metrics are computed.

## Basic Definitions:

- **Longitudinal Line**: Line formed between Apex and Base reference nodes.
- **Normal vector**: Vector based on longitudinal line as "Apex" to "Base"
  


## Geometrical Metrics (Geometrics):

- [**Radial distance**](./radial_metrics.md): Perpendicular distance from Endo/Epi surfaces to normal vector.
- [**Radial length**](./radial_metrics.md): Magnitude of vector from Endo/Epi surface to a specified center located at 'k' height on longitudinal line. The 'k' height is based on a percentage from Apex to Base representing the corresponding speckle height at longitudinal line.
- [**Wall thickness**](./wall_thickness.md): Distance between Epicardium and Endocardium surfaces (can be computed based on radial distance or radia length).
- [**Longitudinal distance**](./longitudinal_distance.md): Euclidean distance between Apex and Base reference nodes.
- [**Longitudinal length**](./longitudinal_length.md): Curvature length of Endo/Epi surfaces along longitudinal axis.
- [**Circumferential length**](./circumferential_length.md): Curvature length of Endo/Epi surfaces along circumferential axis. 



____
### References

- [1] [Definitions for a common standard for 2D speckletracking echocardiography: consensus documentof the EACVI/ASE/Industry Task Force tostandardize deformation imaging](https://core.ac.uk/reader/53744593?utm_source=linkout)


