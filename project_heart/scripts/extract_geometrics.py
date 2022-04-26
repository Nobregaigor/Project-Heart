import numpy as np
import logging

from project_heart.enums import SCRIPT_TAGS, LV_SCRIPT_TAGS
from project_heart.modules.script_handler import ScriptHandler as sh
from project_heart.lv import LV

logger = logging.getLogger('extract_geometrics')
# logging.basicConfig()

def extract_geometrics(**kwargs):
    print(kwargs)
    log_level = kwargs.get(SCRIPT_TAGS.LOG_LEVEL.value, 10)
    logger.setLevel(log_level)
    logger.info("Starting execution of 'extract_geometrics'")

    # --------------------------------
    # resolve input data

    # get data either from kwargs or json input file
    input_data = sh.resolve_json_input_format(**kwargs)
    logger.debug("input_data '{}'".format(input_data))
    # check if input data is list of input values and execute them
    sh.resolve_recursive(input_data, extract_geometrics)
    # check if function should be execute for all files in directory
    sh.resolve_multiple_input_files(input_data, extract_geometrics)

    # this script does not handle multiple input arguments
    # therefore, we do not need to resolve 'resolve_multiple_input_arguments'

    # --------------------------------
    # get arguments

    # required arguments
    input_file = input_data.get(SCRIPT_TAGS.INPUT_FILE.value, None)
    sh.assert_input_file(input_file)
    output_file = input_data.get(SCRIPT_TAGS.OUTPUT_FILE.value, None)
    sh.assert_input_exists(output_file, (str))
    region_args = input_data.get(LV_SCRIPT_TAGS.IDENTIFY_REGIONS.value, None)
    sh.assert_input_exists(region_args, dict)
    spks = input_data.get(LV_SCRIPT_TAGS.SPECKLES.value, None)
    sh.assert_input_exists(spks, (dict, str, list))
    metrics = input_data.get(LV_SCRIPT_TAGS.METRICS.value, None)
    sh.assert_input_exists(metrics, dict)

    # optional arguments
    dtype = input_data.get(SCRIPT_TAGS.DTYPE.value, np.float32)
    log_level = input_data.get(SCRIPT_TAGS.LOG_LEVEL.value, logging.DEBUG)
    filename_map = input_data.get(SCRIPT_TAGS.FILENAME_MAP.value, None)
    merge_with_existing_file = input_data.get(SCRIPT_TAGS.MERGE_WITH_EXISTING_FILE.value, True)

    if filename_map is not None:
        sh.assert_filename_data(input_file, filename_map)

    # --------------------------------
    # start script execution:
    logger.info("File: {}".format(input_file))
    logger.info("Metrics: {}".format(list(metrics.keys())))
    # load spks (make sure it is a dictionary); can be retrieved from json.
    if isinstance(spks, str):
        spks = sh.get_json_data(spks)
    logger.info("Number of spks to create: {}".format(len(spks)))
    # start LV creation
    logger.debug("Loading LV data...")
    lv = LV.from_file(input_file, log_level=log_level)
    logger.debug("Identifying LV regions...")
    lv.identify_regions(**region_args)
    logger.debug("Creating spks...")
    lv.create_speckles_from_iterable(spks)
    logger.debug("Starting extraction...")
    df = lv.extract_geometrics(metrics, dtype=dtype)
    if filename_map is not None:
        logger.debug("Mapping filename to df...")
        df = sh.add_filename_data_to_df(df, str(input_file), filename_map)
    if merge_with_existing_file is True:
        logger.debug("Merging with existing file...")
        df = sh.merge_df_with_existing_at_file(df, output_file)
    sh.export_df(df, output_file, index=False)

