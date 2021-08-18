# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:41:10 2021

@author: mesar
"""

import pandas as pd
import numpy as np
from csv import reader
import sys


def score(actual,sub,cost_mat,g=1000):
    '''
    Scores individual routes.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    cost_mat : dict
        Cost matrix.
    g : int/float, optional
        ERP gap penalty. Irrelevant if large and len(actual)==len(sub). The
        default is 1000.

    Returns
    -------
    float
        Accuracy score from comparing sub to actual.

    '''
    norm_mat=normalize_matrix(cost_mat)
    return seq_dev(actual,sub)*erp_per_edit(actual,sub,norm_mat,g)

def erp_per_edit(actual,sub,matrix,g=1000):
    '''
    Outputs ERP of comparing sub to actual divided by the number of edits involved
    in the ERP. If there are 0 edits, returns 0 instead.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : dict
        Normalized cost matrix.
    g : int/float, optional
        ERP gap penalty. The default is 1000.

    Returns
    -------
    int/float
        ERP divided by number of ERP edits or 0 if there are 0 edits.

    '''
    total,count=erp_per_edit_helper(actual,sub,matrix,g)
    if count==0:
        return 0
    else:
        return total/count

def erp_per_edit_helper(actual,sub,matrix,g=1000,memo=None):
    '''
    Calculates ERP and counts number of edits in the process.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : dict
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.
    memo : dict, optional
        For memoization. The default is None.

    Returns
    -------
    d : float
        ERP from comparing sub to actual.
    count : int
        Number of edits in ERP.

    '''
    if memo==None:
        memo={}
    actual_tuple=tuple(actual)
    sub_tuple=tuple(sub)
    if (actual_tuple,sub_tuple) in memo:
        d,count=memo[(actual_tuple,sub_tuple)]
        return d,count
    if len(sub)==0:
        d=gap_sum(actual,g)
        count=len(actual)
    elif len(actual)==0:
        d=gap_sum(sub,g)
        count=len(sub)
    else:
        head_actual=actual[0]
        head_sub=sub[0]
        rest_actual=actual[1:]
        rest_sub=sub[1:]
        score1,count1=erp_per_edit_helper(rest_actual,rest_sub,matrix,g,memo)
        score2,count2=erp_per_edit_helper(rest_actual,sub,matrix,g,memo)
        score3,count3=erp_per_edit_helper(actual,rest_sub,matrix,g,memo)
        option_1=score1+dist_erp(head_actual,head_sub,matrix,g)
        option_2=score2+dist_erp(head_actual,'gap',matrix,g)
        option_3=score3+dist_erp(head_sub,'gap',matrix,g)
        d=min(option_1,option_2,option_3)
        if d==option_1:
            if head_actual==head_sub:
                count=count1
            else:
                count=count1+1
        elif d==option_2:
            count=count2+1
        else:
            count=count3+1
    memo[(actual_tuple,sub_tuple)]=(d,count)
    return d,count

def normalize_matrix(mat):
    '''
    Normalizes cost matrix.

    Parameters
    ----------
    mat : dict
        Cost matrix.

    Returns
    -------
    new_mat : dict
        Normalized cost matrix.

    '''
    new_mat=mat.copy()
    time_list=[]
    for origin in mat:
        for destination in mat[origin]:
            time_list.append(mat[origin][destination])
    avg_time=np.mean(time_list)
    std_time=np.std(time_list)
    min_new_time=np.inf
    for origin in mat:
        for destination in mat[origin]:
            old_time=mat[origin][destination]
            new_time=(old_time-avg_time)/std_time
            if new_time<min_new_time:
                min_new_time=new_time
            new_mat[origin][destination]=new_time
    for origin in new_mat:
        for destination in new_mat[origin]:
            new_time=new_mat[origin][destination]
            shifted_time=new_time-min_new_time
            new_mat[origin][destination]=shifted_time
    return new_mat

def gap_sum(path,g):
    '''
    Calculates ERP between two sequences when at least one is empty.

    Parameters
    ----------
    path : list
        Sequence that is being compared to an empty sequence.
    g : int/float
        Gap penalty.

    Returns
    -------
    res : int/float
        ERP between path and an empty sequence.

    '''
    res=0
    for p in path:
        res+=g
    return res

def dist_erp(p_1,p_2,mat,g=1000):
    '''
    Finds cost between two points. Outputs g if either point is a gap.

    Parameters
    ----------
    p_1 : str
        ID of point.
    p_2 : str
        ID of other point.
    mat : dict
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.

    Returns
    -------
    dist : int/float
        Cost of substituting one point for the other.

    '''
    if p_1=='gap' or p_2=='gap':
        dist=g
    else:
        dist=mat[p_1][p_2]
    return dist

def seq_dev(actual,sub):
    '''
    Calculates sequence deviation.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.

    Returns
    -------
    float
        Sequence deviation.

    '''
    actual=actual[1:]
    sub=sub[1:-1]
    comp_list=[]
    for i in sub:
        comp_list.append(actual.index(i))
        comp_sum=0
    for ind in range(1,len(comp_list)):
        comp_sum+=abs(comp_list[ind]-comp_list[ind-1])-1
    n=len(actual)
    return (2/(n*(n-1)))*comp_sum

if __name__ == "__main__":
    
    
    k = int(sys.argv[1]) - 1
    
    
    
    new_routes_dict = {}
    na_values = [""]
    i = 0
    route_id = 1
    
    
    column_names = ['route_id', 'route_sequence',  'iteration', 
        'total_cost', 'time_cost',
       'heading_cost', 'angle_cost', 'time_window_cost', 'zone_time_cost', 'zone_distance_cost' ,'evaluation_score',]
    
    df_final = pd.DataFrame(None)
    
    
    df_parsed = pd.read_csv("../data/model_build_outputs/parsed_files/route_"+str(k)+".txt")
    
    df_1=pd.DataFrame(None)
    df_1 = pd.read_csv("../data/model_build_outputs/df_routes_files/df_"+str(k)+".csv", na_values=na_values, keep_default_na=False)
    #df_1.sort_values(by=['order'], inplace=True, ignore_index=True)
    #df_1.rename(columns={"Unnamed: 0": "customer_id"}, inplace=True)
    travel_times = pd.read_csv("../data/model_build_outputs/travel_times_files/travel_times_route_"+str(k)+".csv", na_values=na_values, keep_default_na=False)
    #Convertir travel_time_matrix al formato de ellos (dict)
    new_mat = travel_times.set_index('key')
    cost_mat = new_mat.to_dict()
    depot_number = int(df_parsed.at[4, 'number_of_clients'])-1
    
    tt_df_copy = travel_times.copy()
    tt_df_copy.set_index('key', inplace=True)
    time_matrix=tt_df_copy.to_numpy(dtype='float')
    average_time_between_customers = time_matrix.sum()/(len(time_matrix)*len(time_matrix))
    std_dev_time_between_customers = np.std(time_matrix, dtype=np.float64)
    
    average_time_customers_depot = sum(time_matrix[depot_number])/len(time_matrix)
    std_dev_time_customers_depot = np.std(time_matrix[depot_number], dtype=np.float64)
    df_results = pd.DataFrame(None)
    df_results.at[0, 'route_sequence'] = 0
    
    routes_set = set()
    
    multiple_skips = 0
    w=0
    while(multiple_skips <= 20):
        try:
            df_1 = pd.read_csv("../data/model_build_outputs/df_routes_files/df_"+str(k)+".csv", na_values=na_values, keep_default_na=False)
  
            stored_values = []
            with open("../data/model_build_outputs/local_optima_analisis/solution_route_"+str(k)+"_it_"+str(w), 'r') as read_obj:
                csv_reader = reader(read_obj, skipinitialspace=True,)
                # Iterate over each row in the csv using reader object
                
                
                for row in csv_reader:
                    # row variable is a list that represents a row in csv
                    #print(row)
                    stored_values.append(row)
                grasp_route = stored_values[0]
                zones_order = stored_values[1]
                total_cost = float(stored_values[2][0])
                time_cost = float(stored_values[3][0])
                heading_cost = float(stored_values[4][0])
                angle_cost = float(stored_values[5][0])
                time_window_cost = float(stored_values[6][0])
                zone_time_cost = float(stored_values[7][0])
                zone_distance_cost = float(stored_values[8][0])
            
            #travel_times.set_index("key", inplace=True)
            
            if tuple(grasp_route) in routes_set:
                w = w+1
                continue
            else:
                routes_set.add(tuple(grasp_route)) 
            
            i = i+1
            sub = grasp_route #Our submitted customer_id list
            actual = df_1.customer_id.to_list() #Actual customer_id list
            actual.append(actual[0])
            
        
            #Evaluate score
        
            route_score = score(actual,sub,cost_mat)
            
            
            
            print("Iteration: ", str(i), 'Score: ', route_score)
            
            
            
            number_of_customers = len(df_1)
            number_of_zones = len(df_1.zone_id.value_counts())
            
            
            
            df_results.at[i, 'route_id'] = k
            df_results.at[i, 'iteration'] = i
            df_results.at[i, 'total_cost'] = total_cost
            df_results.at[i, 'time_cost'] = time_cost
            df_results.at[i, 'heading_cost'] = heading_cost
            df_results.at[i, 'angle_cost'] = angle_cost
            df_results.at[i, 'time_window_cost'] = time_window_cost
            df_results.at[i, 'zone_time_cost'] = zone_time_cost
            df_results.at[i, 'zone_distance_cost'] = zone_distance_cost
            df_results.at[i, 'evaluation_score'] = route_score
            
            df_results['route_sequence'] = df_results['route_sequence'].astype(object)
            df_results.at[i, 'route_sequence'] = [sub]
            
            multiple_skips = 0
            
            
            
        except:
            multiple_skips = multiple_skips+1
            pass
        w = w+1

    print("Done with route " + str(k))
    df_results = df_results.reindex(columns=column_names)
    df_results.to_csv("../data/model_build_outputs/df_local_optima_analisis/df_results_prob_route_"+str(k)+".csv")
    
        
    