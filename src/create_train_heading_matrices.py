# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:57:03 2021

@author: mesar
"""

import numpy as np
import pandas as pd
from lib import utils
from progressbar import progressbar as pbar
from pathlib import Path
import sys

if __name__ == "__main__":
    max_distance = int(sys.argv[1])
    dwk = float(sys.argv[2])
    
    Path("../data/model_build_outputs/prob_matrices_heading").mkdir(parents=True, exist_ok=True)
    Path("../data/model_build_outputs/prob_matrices_angle").mkdir(parents=True, exist_ok=True)
    
    print("Max distance: " + str(max_distance))
    print("dwk: " + str(dwk))
    print("Calculating train_routes")
    train_routes = utils.get_routes()
    #hroutes = routes[['route_fid', 'score']].drop_duplicates()
    #hroutes = list(hroutes.route_fid)
    #all_zroutes = utils.get_routes_as_zones()
    #zroutes = all_zroutes[all_zroutes.route_fid.isin(hroutes)]
    print("Calculating train_zroutes")
    train_zroutes = utils.get_routes_as_zones()
    #max_distance = 200
    #dwk = 0.01
    print("Calculating z_route_fields")
    za = utils.ZrouteField(train_zroutes, max_distance=max_distance).compute_field(dwk=dwk)
    print("Calculating heading_matrices")
    h = za.get_estimated_headings(zroutes=train_zroutes)
    
    fname = f'../data/model_build_outputs/heading_estimations_md_{max_distance}_dwk_{dwk:.4f}.hdf'
    h.to_hdf(fname, "data")
    #h = pd.read_hdf("../data/model_apply_outputs/heading_estimations_md_200_dwk_0.1000.hdf")
    
    zroutes = train_zroutes.copy()
    print("Calculating prob_matrices")
    for route_fid in pbar(np.unique(h.route_fid)):
        probs = utils.get_heading_based_probmatrix(h, route_fid)
        probs = probs[~probs.index.str.contains("Station")]
        #probs.drop(probs.filter(regex='Station').columns, axis=1, inplace=True)
        probs.to_csv(f"../data/model_build_outputs/prob_matrices_heading/{route_fid}_probs.csv", sep=',', na_rep='nan')
        zones_id = zroutes.zone_id[zroutes.route_fid==route_fid]
        zones_id = zones_id[~zones_id.str.contains("Station")]
        zones_id.reset_index(inplace=True, drop=True)
        cities = zroutes.city[zroutes.route_fid==route_fid]
        cities.reset_index(inplace=True, drop=True)
        city = cities[0]
        city_size = len(city) + 2
        zones_id = [zones_id[i][city_size:] for i in range(0,len(zones_id))] #Empieza desde 1 para saltarse del Depot
        zones_df = pd.Series(zones_id)
        zones_df = zones_df.append(pd.Series("nan"))
        zones_df.to_csv(f"../data/model_build_outputs/prob_matrices_heading/{route_fid}_zroutes.csv", index=False, header=False, na_rep='nan')
        prob_matrix = utils.get_angle_based_probmatrix(h, route_fid)
        prob_matrix.to_csv(f"../data/model_build_outputs/prob_matrices_angle/{route_fid}_probs.csv", sep=',', na_rep='nan')
        #probs.to_hdf(f"data/prob_matrices_based_on_heading/{route_fid}_probs.hdf", "data")
        
    
    print("Done")