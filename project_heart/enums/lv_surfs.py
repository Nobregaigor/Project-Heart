from enum import IntEnum, Enum


class LV_SURFS(IntEnum):
    OTHER = 0

    ENDO = 1
    EPI = 2
    AORTIC = 3
    MITRAL = 4
    AM_INTERCECTION = 5

    BASE_REGION = 6
    APEX_REGION = 7


class LV_MESH_DATA(Enum):

    APEX_BASE_FILTER = "LV_APEX_BASE_FILTER"
    APEX_BASE_REGION = "LV_APEX_BASE_REGION"
    APEX_BASE_REGIONS = "LV_APEX_BASE_REGIONS"

    # This is for all srufaces except base and apex regions
    SURFS_EXPT_AB = "LV_SURFS_EXPT_AB"

    # This is for all srufaces (including apex and base regions)
    SURFS = "LV_SURFS"


class LV_VIRTUAL_NODES(Enum):
    APEX = "APEX"
    BASE = "BASE"

    AORTIC = "AORTIC"
    MITRAL = "MITRAL"
