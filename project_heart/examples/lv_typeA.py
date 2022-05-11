from pathlib import Path
from project_heart.lv import LV
from project_heart.enums import *
import numpy as np
import logging

import os
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
filepath = "./sample_files/lv_typeA_hex.vtk"
filepath = os.path.join(CURR_DIR, filepath)

statesfile = "./sample_files/sample_displacement_lv_typeA_hex_states_linear_press_incr.pbz2"
statesfile = os.path.join(CURR_DIR, statesfile)

def get_lv_typeA(filepath=filepath,statesfile=statesfile):

    lv = LV.from_file(Path(filepath)) 
    lv.states.from_pickle(str(statesfile))
    
    lv.identify_regions(LV_GEO_TYPES.TYPE_A,
        apex_base_args=dict(ab_ql=0.15, ab_qh=0.95),
        endo_epi_args={"threshold":80},
        recompute_apex_base=dict(ql=0.001, qh=0.75))
  
    endo_long = []
    epi_long = []
    for i, a in enumerate(np.linspace(0, np.pi, 6, endpoint=False)):
        
        spk = lv.create_speckles(
            collection="long-6",
            group="endo",
            name=str(i),
            from_nodeset=LV_SURFS.ENDO,
            exclude_nodeset=LV_SURFS.BASE, # does not afect ideal case
            d=1.75,
            k=0.5,
            normal_to=[np.cos(a),np.sin(a),0.0],
            n_subsets=6,
            subsets_criteria="z2",
            cluster_criteria="z2",
            n_clusters=10,
            t=0.0,
            kmin=-1,
            kmax=0.95,
            log_level=logging.WARN,
        )
        endo_long.append(spk)

        spk = lv.create_speckles(
            collection="long-6",
            group="epi",
            name=str(i),
            from_nodeset=LV_SURFS.EPI,
            exclude_nodeset=LV_SURFS.BASE, # does not afect ideal case
            d=2.4,
            k=0.5,
            normal_to=[np.cos(a),np.sin(a),0.0],
            n_subsets=6,
            subsets_criteria="z2",
            cluster_criteria="z2",
            n_clusters=10,
            t=0.0,
            kmin=-1,
            kmax=0.95,
            log_level=logging.WARN,
        )
        epi_long.append(spk)

    endo_circ = []
    epi_circ = []

    names = ["subapex", "apex", "superapex", "submid", "mid", "supermid", "subbase", "base", "superbase"]
    for i, a in enumerate(np.linspace(0.25, 0.95, len(names), endpoint=False)):
        
        spk = lv.create_speckles(
            collection="circ-6",
            group="endo",
            name=names[i],
            from_nodeset=LV_SURFS.ENDO,
            d=1.75,
            k=a,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=6,
            subsets_criteria="angles",
            cluster_criteria="angles2",
            n_clusters=8,
            t=0.0,
            kmin=-1.0,
            kmax=-1.0,
            log_level=logging.WARN,
        )
        endo_circ.append(spk)

        spk = lv.create_speckles(
            collection="circ-6",
            group="epi",
            name=names[i],
            from_nodeset=LV_SURFS.EPI,
            d=1.75,
            k=a,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=6,
            subsets_criteria="angles",
            cluster_criteria="angles2",
            n_clusters=8,
            t=0.0,
            kmin=-1.0,
            kmax=-1.0,
            log_level=logging.WARN,
        )
        epi_circ.append(spk)



    _ = lv.create_speckles(
            collection="SAMPLE",
            group="epi",
            name="SAMPLE",
            from_nodeset=LV_SURFS.EPI,
            d=4.0,
            k=0.8,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=6,
            subsets_criteria="angles",
            cluster_criteria="angles2",
            n_clusters=8,
            t=0.0,
            kmin=-1.0,
            kmax=-1.0,
            log_level=logging.WARN,
        )

    _ = lv.create_speckles(
            collection="SAMPLE",
            group="endo",
            name="SAMPLE",
            from_nodeset=LV_SURFS.ENDO,
            d=4.0,
            k=0.8,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=6,
            subsets_criteria="angles",
            cluster_criteria="angles2",
            n_clusters=8,
            t=0.0,
            kmin=-1.0,
            kmax=-1.0,
            log_level=logging.WARN,
        )
    
    
    _ = lv.create_speckles(
            collection="SAMPLE",
            group="endo",
            name="THICK",
            from_nodeset=LV_SURFS.ENDO,
            d=6.0,
            k=0.8,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=6,
            subsets_criteria="angles",
            cluster_criteria="angles2",
            n_clusters=8,
            t=0.0,
            kmin=-1.0,
            kmax=-1.0,
            log_level=logging.WARN,
        )

    _ = lv.create_speckles(
            collection="SAMPLE",
            group="endo",
            name="MID",
            from_nodeset=LV_SURFS.ENDO,
            d=3.0,
            k=0.8,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=6,
            subsets_criteria="angles",
            cluster_criteria="angles2",
            n_clusters=8,
            t=0.0,
            kmin=-1.0,
            kmax=-1.0,
            log_level=logging.WARN,
        )
    
    _ = lv.create_speckles(
            collection="SAMPLE",
            group="endo",
            name="THIN",
            from_nodeset=LV_SURFS.ENDO,
            d=1.75,
            k=0.8,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=6,
            subsets_criteria="angles",
            cluster_criteria="angles2",
            n_clusters=8,
            t=0.0,
            kmin=-1.0,
            kmax=-1.0,
            log_level=logging.WARN,
        )
    
    _ = lv.compute_base_apex_ref_over_timesteps()
    
    return lv