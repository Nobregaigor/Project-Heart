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
    output_file = input_data.get(SCRIPT_TAGS.OUTPUT_FILE.value, None)
    sh.assert_input_exists(output_file, (str))
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
    regress_args = input_data.get(LV_SCRIPT_TAGS.REGRESS.value, {})
    lv_log_level = kwargs.get(LV_SCRIPT_TAGS.LOG_LEVEL.value, logging.INFO)

    boundary_conditions = kwargs.get(LV_SCRIPT_TAGS.BOUNDARY_CONDITIONS.value, [])

    

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
        logger.debug("Regressing fibers...")
        logger.debug("-Loading LV data (hex)...")
        lv_hex = LV.from_pyvista_read(input_file, log_level=lv_log_level) # reload hex mesh
        logger.debug("-Identifying LV regions (hex)...")
        lv_hex.identify_regions(**region_args)
        logger.debug("-Applying regression...")
        lv_hex.regress_fibers(lv_tet, **regress_args)
        lv = lv_hex

    # create boundary conditions (if requested)
    if len(boundary_conditions) > 0:
        for bcargs in boundary_conditions:
            lv.create_spring_rim_bc(**bcargs)

    # saving file
    if str(output_file).endswith(".feb"):
        lv.to_feb_template(feb_template, output_file)
        if save_vtk:
            lv.mesh.save(output_file.replace(".feb", ".vtk"))
    elif str(output_file).endswith(".vtk"):
        lv.mesh.save(output_file)
    else:
        import pyvista as pv
        pv.save_meshio(output_file, lv.mesh)
        if save_vtk:
            ext = Path(output_file).suffix
            lv.mesh.save(output_file.replace(ext, ".vtk"))
