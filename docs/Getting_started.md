
# File Structure

The source code of our library is located under `Project_heart` directory. The core objects and funtions are located in `Project_hear/modules`, while other higher-level objects will have their own direction, such as the left ventricle class `LV`, which is located at `Project_hear/lv`. Useful functions that are employed throughout our packages are located in `Project_hear/utils`. You can find the meaning and values for each Enum used in this project under `Project_hear/enums`.

To evaluate and visualize our code, we use jupyter notebooks, located at `notebooks` directory. You will find some demos and other useful notes to help understand and use our library.

# Code structure

## A) Mesh-related data:
---

The core of all mesh-related data structures is the `BaseContainerHandler`. This object is responsible for storing and handling all data related to mesh objects and temporal data (from simulation files). Higher-level classes should inherit this object for consistency and functionallity. The major structure of `BaseContainerHandler` is illustrated bellow: 

![BaseContainerHandler](/assets/images/basecontainerhandler_data_structure.png)

### Main containers:

There are three main containers: `mesh`, `surface_mesh` and `states`. The first are dataset objects from [PyVista](https://docs.pyvista.org/), while the third is a base `state` class. Here is a breakdown of the main containers:

- **mesh**: [pyvista.UnstructuredGrid](https://docs.pyvista.org/api/core/_autosummary/pyvista.UnstructuredGrid.html)
  - `point_data`: <em>Dictionary</em> containing data for each point in the mesh. It is used to store nodal information related to mesh coordinates.
  - `cell_data`: <em>Dictionary</em> containing data for each cell in the mesh. It is used to store element information.


- **surface_mesh**: [pyvista.PolyData](https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.html)
  - `point_data`: <em>Dictionary</em> containing data for each point at surface mesh. It is used to store nodal information related only to surface coordinates.
  - `cell_data`: <em>Dictionary</em> containing data for each cell at surface mesh. It is used to store facet information. 

- **states**: 

### Other containers:
- **bcs**: 
- **nodesets**: 
- **discretesets**: 
- **surfaces of interest**: 
- **Virtual node**: 
- **Virtual element**: 

### Container methods:
.

.

### Other Highlighed methods:
.

.

## B) Pressure-Volume loop analysis:
----
.

.

## C) Cardiovasculat data:
----
.