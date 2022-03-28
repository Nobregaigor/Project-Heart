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

    ENDO_AORTIC = 8
    EPI_AORTIC = 9
    BORDER_AORTIC = 10
    BORDER_MITRAL = 11
    


class LV_MESH_DATA(Enum):

    APEX_BASE_FILTER = "LV_APEX_BASE_FILTER"
    APEX_BASE_REGION = "LV_APEX_BASE_REGION"
    APEX_BASE_REGIONS = "LV_APEX_BASE_REGIONS"

    EPI_ENDO_GUESS = "LV_EPI_ENDO_GUESS"
    AM_CLUSTERS = "LV_AORTIC_MITRAL_CLUSTERS"

    # This is for all srufaces except base and apex regions
    SURFS_EXPT_AB = "LV_SURFS_EXPT_AB"

    # This is for all srufaces (including apex and base regions)
    SURFS = "LV_SURFS"

    ENDO_AORTIC_MASK = "LV_ENDO_AORTIC_MASK"
    EPI_AORTIC_MASK = "LV_EPI_AORTIC_MASK"
    BORDER_AORTIC_MASK = "LV_BORDER_AORTIC_MASK"


class LV_VIRTUAL_NODES(Enum):
    APEX = "APEX"
    BASE = "BASE"

    AORTIC = "AORTIC"
    MITRAL = "MITRAL"


class LV_RIM(Enum):
    NODES = "RIM_NODES"
    CENTER = "RIM_CENTER"
    ELEMENTS = "RIM_ELEMENTS"
    RELATIONS = "RIM_RELATIONS"
    DISTS = "RIM_DISTS"
