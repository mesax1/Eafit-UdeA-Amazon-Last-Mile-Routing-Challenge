# -*- coding: utf-8 -*-
"""
Created on Sat May 29 21:44:12 2021

@author: mesar
"""
import pandas as pd
import numpy as np
#import timeit
from progressbar import progressbar as pbar
from os import listdir
from pathlib import Path
import sys
#from sklearn.neighbors import DistanceMetric
#from math import radians

def Haversine(v):
    """
    distance between two lat,lon coordinates 
    using the Haversine formula. Assumes one
    radius. r = 3,950 to 3,963 mi 
    Value taken from https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    #from timeit import default_timer as timer
    #start = timer()
    R = 6372.8 # Radius of earth in kilometers. Use 3959.87433 for miles
    v = np.radians(v)

    dlat = v[:, 0, np.newaxis] - v[:, 0]
    dlon = v[:, 1, np.newaxis] - v[:, 1]
    c = np.cos(v[:,0,None])

    a = np.sin(dlat / 2.0) ** 2 + c * c.T * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    result = R * c * 1000 #Multiplied by 1000 to get meters instead of km
    #print(round((timer() - start),3))
    return result

if __name__ == "__main__":
    in_test_phase = int(sys.argv[1])
    
    if in_test_phase == 0:
        Path("../data/model_build_outputs/haversine_distance_matrices").mkdir(parents=True, exist_ok=True)
        number_of_routes = len(listdir("../data/model_build_outputs/df_routes_files"))
    else:
        Path("../data/model_apply_outputs/haversine_distance_matrices").mkdir(parents=True, exist_ok=True)
        number_of_routes = len(listdir("../data/model_apply_outputs/df_routes_files_val"))
    
    
    print("Creating haversine distance matrices")
    na_values = [""]
    
    print(number_of_routes)
    for i in pbar(range(number_of_routes)):
        df = pd.DataFrame(None)
        if in_test_phase == 0:
            route_df = pd.read_csv("../data/model_build_outputs/df_routes_files/df_"+str(i)+".csv", na_values=na_values, keep_default_na=False)
            df_tt = pd.read_csv("../data/model_build_outputs/travel_times_files/travel_times_route_"+str(i)+".csv", na_values=na_values, keep_default_na=False )  
        else:
            route_df = pd.read_csv("../data/model_apply_outputs/df_routes_files_val/df_"+str(i)+".csv", na_values=na_values, keep_default_na=False)
            df_tt = pd.read_csv("../data/model_apply_outputs/travel_times_files_val/travel_times_route_"+str(i)+".csv", na_values=na_values, keep_default_na=False )  
      
        df_tt.set_index('key', inplace=True)
        #m_o = df_tt.index.values.tolist()
        m_o = ["".join(item) for item in df_tt.index.values.astype(str)]
        depot_id = route_df.loc[0, "customer_id"]
        depot_id_pos = np.where(df_tt.index == depot_id)[0][0]
        #matrix_order = np.insert(np.delete(m_o, depot_id_pos), 0, depot_id, 0)
        route_df.set_index("customer_id", inplace=True)
        route_df = route_df.reindex(index=df_tt.index)
        data = route_df[['latitude', 'longitude']]
        #starttime = timeit.default_timer()
        #print("The start time is :",starttime)
        distanceH = Haversine(data.values)
        #print("The time difference is :", timeit.default_timer() - starttime)
        distance_matrix = pd.DataFrame(distanceH)
        if in_test_phase == 0:
            distance_matrix.to_csv("../data/model_build_outputs/haversine_distance_matrices/"+str(i)+"_dist_matrix.csv", index_label="distances")
        else:
            distance_matrix.to_csv("../data/model_apply_outputs/haversine_distance_matrices/"+str(i)+"_dist_matrix.csv", index_label="distances")
    print("Done creating haversine distance matrices")