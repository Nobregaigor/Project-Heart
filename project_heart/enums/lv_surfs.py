from enum import IntEnum, Enum


class LV_SURFS(IntEnum):
    OTHER = 0

    # ENDOCARDIUM, EPICARDIUM
    ENDO = 1
    EPI = 2

    # AORTIC VALUES
    AORTIC = 3
    ENDO_AORTIC = 4
    EPI_AORTIC = 5
    BORDER_AORTIC = 6

    # MITRAL VALUES
    MITRAL = 7
    ENDO_MITRAL = 8
    EPI_MITRAL = 9
    BORDER_MITRAL = 10

    # AORTIC, MITRAL INTERSECTION VALUES
    AM_INTERCECTION = 11
    ENDO_AM_INTERCECTION = 12
    EPI_AM_INTERCECTION = 13

    # AORTIC AND MITRAL REGIONS AS ENDO-EPI
    ENDO_AM_REGION = 14
    EPI_AM_REGION = 15

    # BASAL REGIONS
    BASE_REGION = 16
    ENDO_BASE_REGION = 17
    EPI_BASE_REGION = 18

    # APEX REGIONS
    APEX_REGION = 19
    ENDO_APEX_REGION = 20
    EPI_APEX_REGION = 21


class LV_MESH_DATA(Enum):

    # apex and base, no distinction between endo and epi
    APEX_BASE_REGIONS = "LV_APEX_BASE_REGIONS"
    # apex endo, apex epi, base endo, base epi
    AB_ENDO_EPI = "LV_APEX_BASE_REGIONS_ENDO_EPI"

    # 'guess' based on angle between surf normals and geo center
    EPI_ENDO_GUESS = "LV_EPI_ENDO_GUESS"
    EPI_ENDO = "LV_EPI_ENDO"  # final est. of epi and endo surfs

    # aortic, mitral and intersection (no detailed info)
    AM_SURFS = "LV_AM_SURFS"
    # detailed aortic (endo, epi, border), mitral aortic (endo, epi, border), am_intercection (endo, epi) ...
    AM_DETAILED = "LV_AORTIC_MITRAL_CLUSTERS"
    AM_EPI_ENDO = "LV_AM_EPI_ENDO"  # aortic and mitral region with endo-epi layers

    SURFS = "LV_SURFS"  # EPI_ENDO + AM_SURFS
    SURFS_DETAILED = "SURFS_DETAILED"  # EPI_ENDO + AM_DETAILED

    # Fibers
    FIBERS = "LV_FIBERS"


class LV_VIRTUAL_NODES(Enum):
    APEX = "APEX"
    BASE = "BASE"

    AORTIC = "AORTIC"
    MITRAL = "MITRAL"

    AORTIC_BORDER = "AORTIC_BORDER"


class LV_RIM(Enum):
    NODES = "RIM_NODES"
    CENTER = "RIM_CENTER"
    ELEMENTS = "RIM_ELEMENTS"
    RELATIONS = "RIM_RELATIONS"
    DISTS = "RIM_DISTS"
    REF_NODESET = "REF_NODESET"


class LV_AM_INFO(Enum):
    RADIUS = "R"
    CENTER = "C"
    SURF_IDS = "S"
    MESH_IDS = "M"

    BORDER_RADIUS = "RB"
    BORDER_CENTER = "CB"
    BORDER_SURF_IDS = "SB"
    BORDER_MESH_IDS = "MB"


class LV_FIBERS(Enum):
    FIBERS = "FIBERS"
    F0 = "FIBERS"
    SHEET = "SHEET"
    S0 = "SHEET"
    SHEET_NORMAL = "SHEET_NORMAL"
    N0 = "SHEET_NORMAL"
