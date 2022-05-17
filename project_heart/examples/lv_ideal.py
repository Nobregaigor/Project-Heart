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
    lv.identify_regions(geo_type=LV_GEO_TYPES.IDEAL,
                            apex_base_args=dict(ab_ql=0.05, ab_qh=0.97),
                            recompute_apex_base=dict(ql=0.05, qh=0.95)
                            )

    spk_args = list()
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
            kmin=-1,
            kmax=0.95,
            log_level=logging.WARN,
        )
        
        spk_args.append(
            dict(
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
                kmin=-1,
                kmax=0.95,
                log_level=logging.WARN,
            )
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
            kmin=-1,
            kmax=0.95,
            log_level=logging.WARN,
        )
        
        spk_args.append(
            dict(
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
                kmin=-1,
                kmax=0.95,
                log_level=logging.WARN,
            )
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
            cluster_criteria="angles3",
            n_clusters=100,
            t=0.0,
            kmin=-1,
            kmax=0.92,
            log_level=logging.WARN,
        )
        
        spk_args.append(
            dict(
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
                cluster_criteria="angles3",
                n_clusters=100,
                t=0.0,
                kmin=-1,
                kmax=0.92,
                log_level=logging.WARN,
            )
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
            cluster_criteria="angles3",
            n_clusters=100,
            t=0.0,
            kmin=-1,
            kmax=0.92,
            log_level=logging.WARN,
        )
        
        spk_args.append(
            dict(
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
                cluster_criteria="angles3",
                n_clusters=100,
                t=0.0,
                kmin=-1,
                kmax=0.92,
                log_level=logging.WARN,
            )
        )

    names = ["subapex", "apex", "superapex", "submid", "mid", "supermid", "subbase", "base", "superbase"]
    for i, a in enumerate(np.linspace(0.1, 0.95, len(names), endpoint=False)):
        
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
            kmin=-1.0,
            kmax=-1.0,
            log_level=logging.WARN,
        )
        
        spk_args.append(
            dict(
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
                kmin=-1.0,
                kmax=-1.0,
                log_level=logging.WARN,
            )
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
            kmin=-1.0,
            kmax=-1.0,
            log_level=logging.WARN,
        )
        
        spk_args.append(
            dict(
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
                kmin=-1.0,
                kmax=-1.0,
                log_level=logging.WARN,
            )
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
            n_clusters=60,
            t=0.0,
            kmin=-1.0,
            kmax=-1.0,
            log_level=logging.WARN,
        )
        
        spk_args.append(
            dict(
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
                n_clusters=60,
                t=0.0,
                kmin=-1.0,
                kmax=-1.0,
                log_level=logging.WARN,
            )
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
            n_clusters=60,
            t=0.0,
            kmin=-1.0,
            kmax=-1.0,
            log_level=logging.WARN,
        )
        
        spk_args.append(
            dict(
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
                n_clusters=60,
                t=0.0,
                kmin=-1.0,
                kmax=-1.0,
                log_level=logging.WARN,
            )
        )



    
    
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
            cluster_criteria="angles3",
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
            cluster_criteria="angles3",
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
            cluster_criteria="angles3",
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
            cluster_criteria="angles3",
            n_clusters=8,
            t=0.0,
            kmin=-1.0,
            kmax=-1.0,
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
            kmin=-1.0,
            kmax=-1.0,
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
            kmin=-1.0,
            kmax=-1.0,
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
            kmin=-1.0,
            kmax=-1.0,
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
            kmin=-1,
            kmax=-1,
            log_level=logging.WARN,
        )
    
    _ = lv.compute_base_apex_ref_over_timesteps()
    
    if save_spk_dict:
        import json
        with open("./spk_args_lv_ideal.json", "w") as jfile:
            json.dump(spk_args, jfile)
    
    return lv