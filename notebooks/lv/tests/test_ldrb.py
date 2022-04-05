import dolfin
import ldrb
from project_heart.enums import *
from pathlib import Path
import os
from project_heart.lv import LV
import numpy as np
import pyvista as pv
# import project_heart as ph
from project_heart.modules.geometry import Geometry
pv.set_jupyter_backend("pythreejs")


if __name__ == "__main__":
    print("running test")

    lv = LV.from_pyvista_read("../sample_files/lvtetmesh.vtk")
    # lv.smooth_surface(n_iter=500)
    lv.identify_surfaces(
        endo_epi_args=dict(threshold=90.0),
        apex_base_args=dict(ab_ql=0.04, ab_qh=0.69),
        aortic_mitral_args=dict(a1=0.4,
                                a2=0.5,
                                a3=0.3,
                                a4=75,
                                a5=130,

                                m1=0.17,
                                m2=0.02,
                                m3=0.07,
                                m4=0.333
                                )
    )
    # lv.plot("surface", scalars=LV_MESH_DATA.SURFS_DETAILED.value,
    #         cmap="tab20_r")

    # transform point region ids into cell ids at surface level
    cellregionIdsSurf = lv.transform_point_data_to_cell_data(
        LV_MESH_DATA.SURFS_DETAILED.value, surface=True)
    # combine volumetric mesh with surface mesh
    mesh = lv.mesh.copy()
    mesh = mesh.merge(lv.get_surface_mesh())
    # adjust regions to include both surface and volume (with zeros)
    cellregionIds = np.hstack(
        (cellregionIdsSurf, np.zeros(mesh.n_cells - len(cellregionIdsSurf))))
    # add gmsh data
    mesh.clear_data()  # for some reason, no other info is accepted when loading in ldrb
    mesh.cell_data["gmsh:physical"] = cellregionIds
    mesh.cell_data["gmsh:geometrical"] = cellregionIds
    # save using meshio (I did not test other gmsh formats and binary files.)
    pv.save_meshio("../sample_files/lvtetmesh.msh", mesh,
                   file_format="gmsh22", binary=False)

    # Last argument here is the markers, but these are not used
    mesh, ffun, _ = ldrb.gmsh2dolfin(
        "../sample_files/lvtetmesh.msh",
        unlink=False,
    )
    # Run this first in serial and exit here
    # exit()

    markers = {
        "epi": LV_SURFS.EPI.value,
        "lv": LV_SURFS.ENDO.value,
        "base": LV_SURFS.MITRAL.value
    }

    ffun.array()[ffun.array() ==
                 LV_SURFS.EPI_AM_INTERCECTION] = LV_SURFS.EPI.value
    ffun.array()[ffun.array() == LV_SURFS.EPI_AORTIC] = LV_SURFS.EPI.value
    ffun.array()[ffun.array() == LV_SURFS.EPI_MITRAL] = LV_SURFS.EPI.value

    ffun.array()[ffun.array() ==
                 LV_SURFS.ENDO_AM_INTERCECTION] = LV_SURFS.ENDO.value
    ffun.array()[ffun.array() == LV_SURFS.ENDO_AORTIC] = LV_SURFS.ENDO.value
    ffun.array()[ffun.array() == LV_SURFS.ENDO_MITRAL] = LV_SURFS.ENDO.value

    ffun.array()[ffun.array() ==
                 LV_SURFS.BORDER_AORTIC] = LV_SURFS.MITRAL.value
    ffun.array()[ffun.array() ==
                 LV_SURFS.BORDER_MITRAL] = LV_SURFS.MITRAL.value

    fiber_space = "P_1"

    fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
        mesh=mesh,
        fiber_space=fiber_space,
        ffun=ffun,
        markers=markers,
        alpha_endo_lv=60,  # Fiber angle on the endocardium
        alpha_epi_lv=-60,  # Fiber angle on the epicardium
        beta_endo_lv=0,  # Sheet angle on the endocardium
        beta_epi_lv=0,  # Sheet angle on the epicardium
    )

    with dolfin.XDMFFile(mesh.mpi_comm(), "lvtetmesh_fiber.xdmf") as xdmf:
        xdmf.write(fiber)
