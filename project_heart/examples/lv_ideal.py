from pathlib import Path
from project_heart.lv import LV
from project_heart.enums import *
import numpy as np
import logging

import os
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
filepath = "./sample_files/ideal_linear_pressure_increase.xplt"
filepath = os.path.join(CURR_DIR, filepath)
    
def get_lv_ideal(filepath=filepath, save_spk_dict=False):

    lv = LV.from_file(Path(filepath)) 
    lv.identify_regions(geo_type=LV_GEO_TYPES.IDEAL, log_level=logging.INFO)

    # =========================================================================
    # LA SPECKLES
    
    _ = lv.create_speckles(
            collection="LA", # longitudinal axis collection
            group="endo",     # at endocardium
            name="base",    # base region
            from_nodeset=LV_SURFS.BASE_BORDER_ENDO, # using border (edge for ideal)
            use_all_nodes=True, # skip search for nodes close to 'plane'
            log_level=logging.WARN,
        )
    
    _ = lv.create_speckles(
            collection="LA", # longitudinal axis collection
            group="epi",     # at epicardium
            name="base",    # base region
            from_nodeset=LV_SURFS.BASE_BORDER_EPI, # using border (edge for ideal)
            use_all_nodes=True, # skip search for nodes close to 'plane'
            log_level=logging.WARN,
        )

    _ = lv.create_speckles(
            collection="LA", # longitudinal axis collection
            group="endo",     # at endocardium
            name="apex",    # base region
            from_nodeset=LV_SURFS.ENDO, # using border (edge for ideal)
            k=0.0,
            d=1.0,
            log_level=logging.WARN,
        )
    _ = lv.create_speckles(
            collection="LA", # longitudinal axis collection
            group="epi",     # at epicardium
            name="apex",    # base region
            from_nodeset=LV_SURFS.EPI, # using border (edge for ideal)
            use_local_k_ref=True,
            k=0.0,
            d=1.0,
            log_level=logging.WARN,
        )

    # =========================================================================
    # LONGITUDINAL SPECKLES
    
    for i, a in enumerate(np.linspace(0, np.pi, 6, endpoint=False)):
        
        _ = lv.create_speckles(
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
            cluster_criteria="angles3",
            n_clusters=10,
            t=0.0,
            log_level=logging.WARN,
        )
        
        _ = lv.create_speckles(
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
            cluster_criteria="angles3",
            n_clusters=10,
            t=0.0,
            log_level=logging.WARN,
        )
        
        _ = lv.create_speckles(
            collection="long-1",
            group="endo",
            name=str(np.round(np.degrees(a),3)),
            from_nodeset=LV_SURFS.ENDO,
            exclude_nodeset=LV_SURFS.BASE, # does not afect ideal case
            d=1.75,
            k=0.5,
            normal_to=[np.cos(a),np.sin(a),0.0],
            n_subsets=0,
            subsets_criteria="z2",
            cluster_criteria="z2",
            n_clusters=60,
            t=0.0,
            log_level=logging.WARN,
        )

        _ = lv.create_speckles(
            collection="long-1",
            group="epi",
            name=str(np.round(np.degrees(a),3)),
            from_nodeset=LV_SURFS.EPI,
            exclude_nodeset=LV_SURFS.BASE, # does not afect ideal case
            d=1.75,
            k=0.5,
            normal_to=[np.cos(a),np.sin(a),0.0],
            n_subsets=0,
            subsets_criteria="z2",
            cluster_criteria="z2",
            n_clusters=60,
            t=0.0,
            log_level=logging.WARN,
        )

    # =========================================================================
    # CIRCUMFERENTIAL AXIS SPECKLES
    
    names = ["subapex", "apex", "superapex", "submid", "mid", "supermid", "subbase", "base", "superbase"]
    for i, a in enumerate(np.linspace(0.1, 0.95, len(names), endpoint=True)):
        
        _ = lv.create_speckles(
            collection="circ-6",
            group="endo",
            name=names[i],
            from_nodeset=LV_SURFS.ENDO,
            d=2.25,
            k=a,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=6,
            subsets_criteria="angles",
            cluster_criteria="angles3",
            n_clusters=16,
            t=0.0,
            log_level=logging.WARN,
        )

        _ = lv.create_speckles(
            collection="circ-6",
            group="epi",
            name=names[i],
            from_nodeset=LV_SURFS.EPI,
            d=2.25,
            k=a,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=6,
            subsets_criteria="angles",
            cluster_criteria="angles3",
            n_clusters=16,
            t=0.0,
            log_level=logging.WARN,
        )
        
        _ = lv.create_speckles(
            collection="circ-1",
            group="endo",
            name=names[i],
            from_nodeset=LV_SURFS.ENDO,
            d=2.25,
            k=a,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=0,
            subsets_criteria="angles",
            cluster_criteria="angles3",
            n_clusters=36,
            t=0.0,
            log_level=logging.WARN,
        )


        _ = lv.create_speckles(
            collection="circ-1",
            group="epi",
            name=names[i],
            from_nodeset=LV_SURFS.EPI,
            d=2.25,
            k=a,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=0,
            subsets_criteria="angles",
            cluster_criteria="angles3",
            n_clusters=36,
            t=0.0,
            log_level=logging.WARN,
        )
   
    # =========================================================================
    # SAMPLE SPECKLES
    
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
            cluster_criteria="angles3",
            n_clusters=8,
            t=0.0,
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
            cluster_criteria="angles3",
            n_clusters=8,
            t=0.0,
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
            cluster_criteria="angles3",
            n_clusters=8,
            t=0.0,
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
            cluster_criteria="angles3",
            n_clusters=8,
            t=0.0,
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
            cluster_criteria="angles3",
            n_clusters=8,
            t=0.0,
            log_level=logging.WARN,
        )
    
    _ = lv.create_speckles(
            collection="SAMPLE",
            group="epi",
            name="THICK",
            from_nodeset=LV_SURFS.EPI,
            d=6.0,
            k=0.8,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=6,
            subsets_criteria="angles",
            cluster_criteria="angles3",
            n_clusters=8,
            t=0.0,
            log_level=logging.WARN,
        )

    _ = lv.create_speckles(
            collection="SAMPLE",
            group="epi",
            name="MID",
            from_nodeset=LV_SURFS.EPI,
            d=3.0,
            k=0.8,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=6,
            subsets_criteria="angles",
            cluster_criteria="angles3",
            n_clusters=8,
            t=0.0,
            log_level=logging.WARN,
        )
    
    _ = lv.create_speckles(
            collection="SAMPLE",
            group="epi",
            name="THIN",
            from_nodeset=LV_SURFS.EPI,
            d=1.75,
            k=0.8,
            normal_to=[0.0, 0.0, 1.0],
            n_subsets=6,
            subsets_criteria="angles",
            cluster_criteria="angles3",
            n_clusters=8,
            t=0.0,
            log_level=logging.WARN,
        )


    _ = lv.create_speckles(
            collection="SAMPLE",
            group="epi",
            name="SAMPLE-LONG",
            from_nodeset=LV_SURFS.EPI,
            exclude_nodeset=LV_SURFS.BASE, # does not afect ideal case
            d=4.0,
            k=0.5,
            normal_to=[1.0,0.0,0.0],
            n_subsets=6,
            subsets_criteria="z2",
            cluster_criteria="angles3",
            n_clusters=18,
            t=0.0,
            log_level=logging.WARN,
        )
    

    apex_spk = lv.get_speckles(spk_collection="LA", spk_group="endo", spk_name="apex")
    base_spk = lv.get_speckles(spk_collection="LA", spk_group="endo", spk_name="base")

    # lv.compute_base_apex_ref_over_timesteps(apex_spk, base_spk, log_level=logging.INFO)

    return lv