# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 20:59:04 2021

@author: mesar
"""

import pandas as pd
from os import path, listdir


    
df_final = pd.DataFrame(None)
number_of_routes = len(listdir("../data/model_apply_outputs/df_local_optima_analisis"))
print(number_of_routes)
for i in range(number_of_routes):
    try:
        df_results = pd.read_csv("../data/model_apply_outputs/df_local_optima_analisis/df_results_prob_route_"+str(i)+".csv", skiprows=[1], index_col=0)
        df_results.reset_index(inplace=True)
        df_results["iteration"] = df_results.index
        df_results.route_id = df_results.route_id.astype('int64')
        df_final = df_final.append(df_results[['route_id', 'iteration', 'total_cost', 'time_cost', 'heading_cost', 'angle_cost', 'time_window_cost', 'zone_time_cost', 'zone_distance_cost', 'evaluation_score', 'route_sequence' ]], ignore_index=True)
    except:
        pass


df_final.to_csv("../data/model_apply_outputs/df_all_routes_scores.csv", index=False)