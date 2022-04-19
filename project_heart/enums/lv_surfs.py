from enum import IntEnum, Enum


class LV_GEO_TYPES(IntEnum):
    IDEAL = 0

    NONIDEAL = 1
    TYPE_A = 1

    TYPE_B = 2


class LV_SURFS(IntEnum):
    OTHER = 0  # DO NOT MODIFY THIS ONE.

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

    # BASE BORDER
    BASE_BORDER = 23
    BASE = 23


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

    LDRB_1 = "LV_FIBERS_LDRB_1"
    LDRB_2 = "LV_FIBERS_LDRB_2"
    LDRB_3 = "LV_FIBERS_LDRB_3"

    # Fibers
    FIBERS = "LV_FIBERS"
    SHEET = "LV_SHEET"
    SHEET_NORMAL = "LV_SHEET_NORMAL"

    # FIBER ANGLES
    FIBER_ANGLES = "LV_FIBER_ANGLES"
    SHEET_ANGLES = "LV_SHEET_ANGLES"
    SHEET_NORMAL_ANGLES = "LV_SHEET_NORMAL_ANGLES"


class LV_VIRTUAL_NODES(Enum):
    APEX = "APEX"
    BASE = "BASE"

    AORTIC = "AORTIC"
    MITRAL = "MITRAL"

    AORTIC_BORDER = "AORTIC_BORDER"
    MITRAL_BORDER = "MITRAL_BORDER"


class LV_RIM(Enum):
    NODES = "RIM_NODES"
    CENTER = "RIM_CENTER"
    ELEMENTS = "RIM_ELEMENTS"
    RELATIONS = "RIM_RELATIONS"
    DISTS = "RIM_DISTS"
    REF_NODESET = "REF_NODESET"


class LV_BCS(Enum):
    RIM_SPRINGS = "RIM_SPRINGS"


class LV_AM_INFO(Enum):
    RADIUS = "R"
    CENTER = "C"
    SURF_IDS = "S"
    MESH_IDS = "M"

    BORDER_RADIUS = "RB"
    BORDER_CENTER = "CB"
    BORDER_SURF_IDS = "SB"
    BORDER_MESH_IDS = "MB"


class LV_BASE_INFO(Enum):
    RADIUS = "R"
    CENTER = "C"
    SURF_IDS = "S"
    MESH_IDS = "M"


class LV_FIBER_MODES(Enum):
    LDRB_1 = LV_MESH_DATA.LDRB_1.value
    LDRB_2 = LV_MESH_DATA.LDRB_2.value
    LDRB_3 = LV_MESH_DATA.LDRB_3.value


class LV_FIBERS(Enum):
    # FIBER DATA
    FIBERS = LV_MESH_DATA.FIBERS.value
    F0 = LV_MESH_DATA.FIBERS.value
    SHEET = LV_MESH_DATA.SHEET.value
    S0 = LV_MESH_DATA.SHEET.value
    SHEET_NORMAL = LV_MESH_DATA.SHEET_NORMAL.value
    N0 = LV_MESH_DATA.SHEET_NORMAL.value
    # ANGLES
    FIBER_ANGLES = LV_MESH_DATA.FIBER_ANGLES.value
    F0_ANGLES = LV_MESH_DATA.FIBER_ANGLES.value
    SHEET_ANGLES = LV_MESH_DATA.SHEET_ANGLES.value
    S0_ANGLES = LV_MESH_DATA.SHEET_ANGLES.value
    SHEET_NORMAL_ANGLES = LV_MESH_DATA.SHEET_NORMAL_ANGLES.value
    N0_ANGLES = LV_MESH_DATA.SHEET_NORMAL_ANGLES.value


class LV_STATES(Enum):
    DISP = "displacement"
    DISPLACEMENT = "displacement"
    STRESS = "stress"

    XYZ = "xyz"
    POS = "xyz"
    POSITION = "xyz"

    VOLUME = "volume"
    VOL = "volume"

    VOLUMETRIC_FRACTION = "volumetric_fraction"
    VF = "volumetric_fraction"
    EF = "volumetric_fraction"
    
    
    BASE_REF = "base_ref"
    APEX_REF = "apex_ref"
    

    LONGITUDINAL_DISTANCES = "longitudinal_distances"
    LONG_DISTS = "longitudinal_distances"

    LONGITUDINAL_SHORTENING = "longitudinal_shortening"
    LS = "longitudinal_shortening"

    RADIUS = "radius"

    RADIAL_SHORTENING = "radial_shortening"
    RS = "radial_shortening"

    THICKNESS = "thickness"

    WALL_THICKENING = "wall_thickening"
    THICKENING = "wall_thickening"
    WT = "wall_thickening"

    LONG_LENGTH = "longitudinal_length"
    LONG_STRAIN = "longitudinal_strain"
    SL = "longitudinal_strain"

    CIRC_LENGTH = "circumferential_length"
    CIRC_STRAIN = "circumferential_strain"
    SC = "circumferential_strain"
    
    SPK_VECS = "spckle_vectors"

    ROTATION = "angle_rotation"
    RO = "angle_rotation"

    TWIST = "twist"
    TW = "twist"

    TORSION = "torsion"
    TO = "torsion"


class LV_SPK_SETS(Enum):
    
    GROUP = "group"
    NAME = "name"

    GROUP_NAME = "group_name"
    NAME_GROUP = "group_name"