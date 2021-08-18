# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:04:34 2021

@author: mesar
"""

import numpy as np
import pandas as pd
from lib import utils
from progressbar import progressbar as pbar
from pathlib import Path
import itertools
from time import time
import csv

if __name__ == "__main__":
    
    Path("../data/model_build_outputs/all_prob_matrices_heading").mkdir(parents=True, exist_ok=True)
    Path("../data/model_build_outputs/prob_matrices_heading").mkdir(parents=True, exist_ok=True)
    Path("../data/model_build_outputs/prob_matrices_angle").mkdir(parents=True, exist_ok=True)
    
    print("Calculating train_routes")
    #train_routes = utils.get_train_routes()
    routes = utils.get_routes()
    hroutes = np.unique(routes.route_fid)
    all_zroutes = utils.get_routes_as_zones()
    zroutes = all_zroutes[all_zroutes.route_fid.isin(hroutes)]
    print("Done reading routes")
    
    t0 = time()
    max_distances = [50, 100, 150, 200, 250, 300]
    dwks = [0.01, 0.05, 0.1, 0.15]
    
    
    r = []
    for max_distance, dwk in itertools.product(max_distances, dwks):
        tt = time()
        #print ("\n----------\n%3d"%max_distance, "%.2f"%dwk, end=" || ", flush=True)
        za = utils.ZrouteField(zroutes, max_distance=max_distance).compute_field(dwk=dwk, use_pbar=True)
        h = za.get_estimated_headings(use_pbar=True)
        rr = za.heading_estimations_cosdistance(h)
        rr['max_distance'] = max_distance
        rr['dwk'] = dwk
        rr['zones_estimated'] = np.mean(h.cos_distance!=0)
        rr['time'] = time()-t0
        rr['nroutes'] = len(np.unique(za.zroutes.route_fid))
        t0 = time()
        r.append(rr)
        print ("maxd %3d, "%max_distance, "dwk %.2f, "%dwk, f'time {time()-tt:.4f}, cos_sim {rr["cos_distance_mean"]:.4f}', flush=True)
    r = pd.DataFrame(r)
    r.to_hdf("../data/model_build_outputs/md_dkw_exploration.hdf", "data")
    
    
    dwks = np.sort(np.unique(r.dwk))
    max_distances = np.sort(np.unique(r.max_distance))
    
    csims    = np.zeros((len(dwks), len(max_distances)))
    zcovered = np.zeros((len(dwks), len(max_distances)))
    
    for i,dwk in enumerate(dwks):
        for j,max_distance in enumerate(max_distances):
            k = r[(r.max_distance==max_distance)&(r.dwk==dwk)].iloc[0]
            csims[i,j] = k.cos_distance_mean
            zcovered[i,j] = k.zones_estimated
    
    for distance in max_distances:        
        k = r[r.max_distance==distance]
        print(k)
    
    estimated_zones_value = 1.0
    best_options = r[r.zones_estimated >= estimated_zones_value]
    if not best_options.empty:
        best_options = r[r.zones_estimated >= estimated_zones_value]
        best_combination = best_options[best_options.cos_distance_mean == best_options.cos_distance_mean.max()]
        selected_max_distance = best_combination.max_distance.values[0]
        selected_dwk = best_combination.dwk.values[0]
    
    while best_options.empty:
        print("Empty for value: " + str(estimated_zones_value))
        estimated_zones_value = estimated_zones_value - 0.1
        best_options = r[r.zones_estimated >= estimated_zones_value]
        best_combination = best_options[best_options.cos_distance_mean == best_options.cos_distance_mean.max()]
        selected_max_distance = best_combination.max_distance.values[0]
        selected_dwk = best_combination.dwk.values[0]
        print(selected_max_distance)
        print(selected_dwk)
    
    output_path = "../data/model_build_outputs/best_max_distance.csv"
    with open(output_path, "w") as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([selected_max_distance])
        
    output_path = "../data/model_build_outputs/best_dwk.csv"
    with open(output_path, "w") as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([selected_dwk])
        
    print("Max distance: " + str(selected_max_distance))
    print("dwk: " + str(selected_dwk))
    print("Calculating train_routes")
    train_routes = utils.get_routes()
    print("Calculating train_zroutes")
    train_zroutes = utils.get_routes_as_zones()
    print("Calculating z_route_fields")
    za = utils.ZrouteField(train_zroutes, max_distance=selected_max_distance).compute_field(dwk=selected_dwk)
    print("Calculating heading_matrices")
    h = za.get_estimated_headings(zroutes=train_zroutes)
    
    fname = f'../data/model_build_outputs/heading_estimations_md_{selected_max_distance}_dwk_{selected_dwk:.4f}.hdf'
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
    