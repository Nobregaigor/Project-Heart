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
    ENDO_AM = 14
    EPI_AM = 15

    # -- BASE AND APEX REGIONS -- 
    # BASAL REGIONS
    BASE = 16
    BASE_EST = 16  # FOR DEBUGING
    BASE_ENDO = 17 # BASE + END0
    BASE_EPI = 18  # BASE + EPI

    # APEX REGIONS
    APEX = 19
    APEX_EST = 19   # FOR DEBUGING
    APEX_ENDO = 20  # APEX + ENDO
    APEX_EPI = 21   # APEX + EPI

    # BASE BORDER
    BASE_BORDER = 23 # 
    BASE_BORDER_ENDO = 24 # INTERSECTION OF BASE AND ENDO
    BASE_BORDER_EPI = 25  # INTERSECTION OF BASE AND EPI

    # BASE_EXCLUDE_ENDO = 26
    # BASE_EXCLUDE_EPI = 27

    
    

    

class LV_MESH_DATA(Enum):

    # apex and base, no distinction between endo and epi
    APEX_BASE_EST = "LV_APEX_BASE_EST"
    APEX_BASE = "LV_APEX_BASE"

    # apex endo, apex epi, base endo, base epi
    AB_ENDO_EPI = "LV_APEX_BASE_REGIONS_ENDO_EPI"

    # 'guess' based on angle between surf normals and geo center
    EPI_ENDO_EST = "LV_EPI_ENDO_EST"
    EPI_ENDO = "LV_EPI_ENDO"  # final est. of epi and endo surfs
    EPI_ENDO_EXCLUDE_BASE = "EPI_ENDO_EXCLUDE_BASE"
    
    
    # aortic, mitral and intersection (no detailed info)
    AM_SURFS = "LV_AM_SURFS"
    # detailed aortic (endo, epi, border), mitral aortic (endo, epi, border), am_intercection (endo, epi) ...
    AM_DETAILED = "LV_AORTIC_MITRAL_CLUSTERS"
    AM_EPI_ENDO = "LV_AM_EPI_ENDO"  # aortic and mitral region with endo-epi layers

    SURFS = "LV_SURFS"  # EPI_ENDO + AM_SURFS
    SURFS_DETAILED = "LV_SURFS_DETAILED"  # EPI_ENDO + AM_DETAILED

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
    APEX_BASE_OVER_TIMESTEPS = "apex_base_over_timesteps"
    
    LA_CENTERS = "la_centers" # used for spk centers

    LONGITUDINAL_DISTANCE = "longitudinal_distance"
    LONG_DISTS = "longitudinal_distance"

    LONGITUDINAL_SHORTENING = "longitudinal_shortening"
    LS = "longitudinal_shortening"

    # RADIUS = "radius"
    RADIAL_DISTANCE = "radial_distance"
    RADIAL_LENGTH = "radial_length"

    RADIAL_SHORTENING = "radial_shortening"
    RS = "radial_shortening"

    WALL_THICKNESS = "wall_thickness"
    THICKNESS = "wall_thickness"

    WALL_THICKENING = "wall_thickening"
    THICKENING = "wall_thickening"
    WT = "wall_thickening"
    
    LONG_LENGTH = "longitudinal_length"
    LONGITUDINAL_LENGTH = "longitudinal_length"
    GLOBAL_LONGITUDINAL_LENGTH = "global_longitudinal_length"
    GLOBAL_LONG_LENGTH = "global_longitudinal_length"

    LONGITUDINAL_STRAIN = "longitudinal_strain"
    LONG_STRAIN = "longitudinal_strain"
    SL = "longitudinal_strain"

    CIRC_LENGTH = "circumferential_length"
    CIRCUMFERENTIAL_LENGTH = "circumferential_length"
    GLOBAL_CIRCUMFERENTIAL_LENGTH = "global_circumferential_length"
    GLOBAL_CIRC_LENGTH = "global_circumferential_length"

    CIRCUMFERENTIAL_STRAIN = "circumferential_strain"
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
    SUBSET = "subset"

    GROUP_NAME = "group_name"
    NAME_GROUP = "group_name"
    
    

class LV_SPK_STATES(Enum):
    PLACEHOLDER = "placeholder"