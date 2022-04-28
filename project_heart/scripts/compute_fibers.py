import numpy as np
import logging
import os
from pathlib import Path

from project_heart.enums import SCRIPT_TAGS, LV_SCRIPT_TAGS, VTK_ELEMENTS
from project_heart.modules.script_handler import ScriptHandler as sh
from project_heart.lv import LV
from project_heart.utils.enum_utils import enum_to_dict, assert_member, assert_value

logger = logging.getLogger('compute_fibers')
# logging.basicConfig()

def compute_fibers(**kwargs):

    # get log level for this module
    log_level = kwargs.get(SCRIPT_TAGS.LOG_LEVEL.value, 10)
    logger.setLevel(log_level)
    logger.info("Starting execution of 'compute_fibers'")

    # --------------------------------
    # resolve input data

    # get data either from kwargs or json input file
    input_data = sh.resolve_json_input_format(**kwargs)
    logger.debug("input_data '{}'".format(input_data))
    # check if input data is list of input values and execute them
    sh.resolve_recursive(input_data, compute_fibers)
    # check if function should be execute for all files in directory
    sh.resolve_multiple_input_files(input_data, compute_fibers)

    # this script does not handle multiple input arguments
    # therefore, we do not need to resolve 'resolve_multiple_input_arguments'

    # =================================================================
    # get arguments

    # required arguments
    input_file = input_data.get(SCRIPT_TAGS.INPUT_FILE.value, None)
    sh.assert_input_file(input_file)
    output_file = sh.resolve_output_filename(input_data, "with_fibers", ".vtk")
    sh.assert_input_exists(output_file, (str, Path))
    region_args = input_data.get(LV_SCRIPT_TAGS.IDENTIFY_REGIONS.value, None)
    sh.assert_input_exists(region_args, dict)
    # this indicates what type of compute-fibers schematics we should use.
    mesh_type = input_data.get(SCRIPT_TAGS.MESH_TYPE.value, None)
    sh.assert_input_exists(mesh_type, (int, str))
    if isinstance(mesh_type, str):
        assert_member(VTK_ELEMENTS, mesh_type)
        mesh_value = VTK_ELEMENTS[mesh_type].value
    elif isinstance(mesh_type, int): 
        assert_value(VTK_ELEMENTS, mesh_type)
        mesh_value = VTK_ELEMENTS(mesh_type).value
    alpha_endo = input_data.get(LV_SCRIPT_TAGS.ALPHA_ENDO.value, None)
    sh.assert_input_exists(alpha_endo, (int, float))
    alpha_epi = input_data.get(LV_SCRIPT_TAGS.ALPHA_EPI.value, None)
    sh.assert_input_exists(alpha_epi, (int, float))
    assert alpha_endo >= 0, "Fiber angle at Endo value should be positive."
    assert alpha_epi <= 0, "Fiber anfle at Epi value should be negative."

    # as of now, we are only able to write feb if a template is provided.
    if str(output_file).endswith(".feb"):
        feb_template = input_data.get(LV_SCRIPT_TAGS.FEB_TEMPLATE.value, None)
        sh.assert_input_exists(feb_template, str)

    # optional arguments
    prefix_map = input_data.get(SCRIPT_TAGS.PREFIX_MAP.value, None)
    save_vtk = input_data.get(SCRIPT_TAGS.SAVE_VTK.value, True)

    fiber_args = input_data.get(LV_SCRIPT_TAGS.FIBERS.value, {})
    tetrahedralize_args = input_data.get(LV_SCRIPT_TAGS.TETRAHEDRALIZE.value, {})

    interpolate_args = input_data.get(LV_SCRIPT_TAGS.INTERPOLATE.value, {})
    regress_args = input_data.get(LV_SCRIPT_TAGS.REGRESS.value, {})
    if len(interpolate_args) > 0 and len(regress_args) > 0:
        raise AssertionError(
            "One method for transfering data from a tetrahedral to non-tetrahedral "
            "mesh is allowed. Please specify either 'interpolate' or 'regress' "
            "but not both.")

    lv_log_level = kwargs.get(LV_SCRIPT_TAGS.LOG_LEVEL.value, logging.INFO)

    boundary_conditions = kwargs.get(LV_SCRIPT_TAGS.BOUNDARY_CONDITIONS.value, [])
    surfaces = kwargs.get(LV_SCRIPT_TAGS.SURFACES.value, [])

   
    # -------------------------------
    # resolve prefix data
    prefix_dict = {"endo": alpha_endo, "epi": alpha_epi}
    output_file = sh.resolve_prefix(output_file, prefix_dict, prefix_map)


    # =================================================================
    # start script execution:
    logger.info("File: {}".format(input_file))
    logger.info("Endo: {}, Epi: {}".format(alpha_endo, alpha_epi))
    
    if mesh_value == VTK_ELEMENTS.TETRA.value:
        logger.info("Using Tetrahedral mesh schematics.")
        logger.debug("Loading LV data...")
        lv = LV.from_file(input_file, log_level=lv_log_level)
        logger.debug("Identifying LV regions...")
        lv.identify_regions(**region_args)
        logger.debug("Computing fibers...")
        lv.compute_fibers(alpha_endo_lv=alpha_endo, alpha_epi_lv=alpha_epi, **fiber_args)
    else:
        logger.info("Using Non-Tetrahedral mesh schematics "
            "(Will apply tetrahedralization and fiber regression.")
        logger.debug("Loading LV data...")
        lv_tet = LV.from_file(input_file, log_level=lv_log_level)
        logger.debug("Apply tetrahedralization...")
        lv_tet.tetrahedralize(**tetrahedralize_args)
        logger.debug("Identifying LV regions...")
        lv_tet.identify_regions(**region_args)
        logger.debug("Computing fibers...")
        lv_tet.compute_fibers(alpha_endo_lv=alpha_endo, alpha_epi_lv=alpha_epi, **fiber_args)
        logger.debug("Transfering fiber data to non-tet LV...")
        logger.debug("Loading LV data (hex)...")
        lv_hex = LV.from_pyvista_read(input_file, log_level=lv_log_level) # reload hex mesh
        logger.debug("Identifying LV regions (hex)...")
        lv_hex.identify_regions(**region_args)
        if len(regress_args) > 0:
            logger.debug("Applying regression...")
            lv_hex.regress_fibers(lv_tet, **regress_args)
        else:
            logger.debug("Applying interpolation (default)...")
            lv_hex.interpolate_fibers(lv_tet, **interpolate_args)
        lv = lv_hex

    # create boundary conditions (if requested)
    if len(boundary_conditions) > 0:
        for bcargs in boundary_conditions:
            lv.create_spring_rim_bc(**bcargs)
    
    if len(surfaces) > 0:
        for surf in surfaces:
            lv.create_surface_oi_from_surface(surf)

    # saving file
    if str(output_file).endswith(".feb"):
        lv.to_feb_template(feb_template, output_file)
        if save_vtk:
            lv.mesh.save(sh.change_ext(output_file, ".vtk"))
    elif str(output_file).endswith(".vtk"):
        lv.mesh.save(output_file)
    else:
        import pyvista as pv
        pv.save_meshio(output_file, lv.mesh)
        if save_vtk:
            lv.mesh.save(sh.change_ext(output_file, ".vtk"))
