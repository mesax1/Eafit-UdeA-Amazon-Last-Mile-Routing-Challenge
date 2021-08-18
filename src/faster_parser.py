# -*- coding: utf-8 -*-


import pandas as pd
import json
from datetime import datetime, timedelta
import numpy as np
import csv
from pathlib import Path
from progressbar import progressbar as pbar
import time

Path("../data/model_build_outputs/parsed_files").mkdir(parents=True, exist_ok=True)
Path("../data/model_build_outputs/df_routes_files").mkdir(parents=True, exist_ok=True)
Path("../data/model_build_outputs/travel_times_files").mkdir(parents=True, exist_ok=True)
Path("../data/model_build_outputs/grasp_routes_prob").mkdir(parents=True, exist_ok=True)
def poprow(my_array,pr):
    """ row popping in numpy arrays
    Input: my_array - NumPy array, pr: row index to pop out
    Output: [new_array,popped_row] """
    i = pr
    pop = my_array[i]
    new_array = np.vstack((my_array[:i],my_array[i+1:]))
    return [new_array,pop]

def popcol(my_array,pc):
    """ column popping in numpy arrays
    Input: my_array: NumPy array, pc: column index to pop out
    Output: [new_array,popped_col] """
    i = pc
    pop = my_array[:,i]
    new_array = np.hstack((my_array[:,:i],my_array[:,i+1:]))
    return [new_array,pop]
    

def parse_json(i, key, number_of_clients, vehicle_capacity, package_data_list, route_data_list, travel_times_list, actual_sequences_list):       
    route_info = {}
    travel_time_matrix = []
    row_values = ['key']
    for k in travel_times_list.keys():           
        row_values.append(k)
    travel_time_matrix.append(row_values)
    for k in travel_times_list.keys():
        row_values = []
        row_values.append(k)
        for j in travel_times_list[k].keys():
            row_values.append(travel_times_list[k][j])
        travel_time_matrix.append(row_values)
    
    for k in actual_sequences_list.keys():
        for j in actual_sequences_list[k].keys():
            route_info[j] = {}
            route_info[j]['order'] = actual_sequences_list[k][j]
    
    latlongs = []
    zone_ids = []
    for k in route_data_list['stops'].keys():
        latlongs.append((route_data_list['stops'][k]['lat'], route_data_list['stops'][k]['lng']))
        depart_time_1 = str(route_data_list['date_YYYY_MM_DD']) + " " + str(route_data_list['departure_time_utc'])
        departure_time = datetime.strptime(depart_time_1, '%Y-%m-%d %H:%M:%S')
        departure_time_seconds = (departure_time.hour * 3600) + (departure_time.minute * 60) + departure_time.second
        
        route_info[k]['latitude'] = route_data_list['stops'][k]['lat']
        route_info[k]['longitude'] = route_data_list['stops'][k]['lng']
        route_info[k]['zone_id'] = route_data_list['stops'][k]['zone_id']
        id_zona = -1 if route_data_list['stops'][k]['zone_id'] == "NaN" else route_data_list['stops'][k]['zone_id']
        zone_ids.append(id_zona)
        route_info[k]['type'] = route_data_list['stops'][k]['type']
        route_info[k]['score'] = route_data_list['route_score']
        route_info[k]['departure_time'] = route_data_list['departure_time_utc']
        route_info[k]['departure_date'] = route_data_list['date_YYYY_MM_DD']
        route_info[k]['route_id'] = key
        route_info[k]['max_capacity'] = vehicle_capacity
    
    time_windows = []
    counter = 0
    planned_service_times = []
    dimensions = []
    package_id_and_client = {}
    double_visits = {}
    for k in package_data_list.keys():
        
        time_window1 = -1
        time_window2 = -1
        sum_dimensions = 0
        number_packages = 0
        max_depth = 0.0
        max_width = 0.0
        max_height = 0.0
        planned_service_time_client = 0.0
        package_status = ""
        for j in package_data_list[k].keys():
            if j in package_id_and_client:
                double_visits[k] = [package_id_and_client[j]]
                double_visits[package_id_and_client[j]] = [k]
            else:
                package_id_and_client[j] = k
            new_package_status = 'D' if str(package_data_list[k][j]['scan_status']) == 'DELIVERED' else 'A'
            package_status = package_status +'_' + new_package_status
            date_value1 = str(package_data_list[k][j]['time_window']['start_time_utc'])
            date_value2 = str(package_data_list[k][j]['time_window']['end_time_utc'])
            planned_service_time_client += package_data_list[k][j]['planned_service_time_seconds']
            if(date_value1 != 'nan' and date_value2 != 'nan'):
                real_date_1 = datetime.strptime(date_value1, '%Y-%m-%d %H:%M:%S')
                real_date_2 = datetime.strptime(date_value2, '%Y-%m-%d %H:%M:%S')
                date_1 = datetime.strptime(date_value1, '%Y-%m-%d %H:%M:%S') - departure_time
                date_2 = datetime.strptime(date_value2, '%Y-%m-%d %H:%M:%S') - departure_time
                if (real_date_1 <= departure_time):
                    time_window1 = 0
                    time_window2 = date_2.seconds
                else:
                    time_window1 = date_1.seconds
                    time_window2 = date_2.seconds
                
                #time_window1 = (date_1.hour * 3600) + (date_1.minute * 60) + date_1.second
                #time_window2 = (date_2.hour * 3600) + (date_2.minute * 60) + date_2.second
                #if(date_1.day != date_2.day):
                    #time_window2 += (24*3600)
            else:
                time_window1 = -1
                time_window2 = -1
                real_date_1 = -1
                real_date_2 = -1
            
            try:
                #if(time_windows[counter][0] == -1 and time_windows[counter][1] == -1):
                #time_windows[counter] = (time_window1, time_window2)
                route_info[k]['time_window_start_seconds'] = time_window1
                route_info[k]['time_window_end_seconds'] = time_window2
                route_info[k]['time_window_start'] = real_date_1
                route_info[k]['time_window_end'] = real_date_2
            except(BaseException):
                #time_windows.append((time_window1, time_window2))
                route_info[k]['time_window_start_seconds'] = time_window1
                route_info[k]['time_window_end_seconds'] = time_window2
                route_info[k]['time_window_start'] = real_date_1
                route_info[k]['time_window_end'] = real_date_2
            
            depth = float(-1 if package_data_list[k][j]['dimensions']['depth_cm'] == 'NaN' else package_data_list[k][j]['dimensions']['depth_cm'])
            height = float(-1 if package_data_list[k][j]['dimensions']['height_cm'] == 'NaN' else package_data_list[k][j]['dimensions']['height_cm'])
            width = float(-1 if package_data_list[k][j]['dimensions']['width_cm'] == 'NaN' else package_data_list[k][j]['dimensions']['width_cm'])
            max_depth = depth if ((depth >= max_depth) and (depth != -1)) else max_depth
            max_height = height if ((height >= max_height) and (height != -1)) else max_height
            max_width = width if ((width >= max_width) and (width != -1)) else max_width
            sum_dimensions += (depth * height * width)
            number_packages += 1
        planned_service_times.append(planned_service_time_client)
        dimensions.append(sum_dimensions)
        route_info[k]['service_time'] = planned_service_time_client
        route_info[k]['dimensions'] = sum_dimensions
        route_info[k]['number_packages'] = number_packages
        route_info[k]['package_status'] = package_status
        route_info[k]['max_depth'] = max_depth
        route_info[k]['max_height'] = max_height
        route_info[k]['max_width'] = max_width
        #route_info[k]['double_visit'] = double_visits[k] if k in double_visits else -1
        
        time_windows.append((time_window1, time_window2))
        counter += 1
    
    for k in route_info.keys():
        if route_info[k]['type'] == "Station":
            depot_key = k
    
    for z in range(len(travel_time_matrix)):
        if travel_time_matrix[z][0] == depot_key:
            depot_number =  z
            
    
    #f = open("../../parsed_files/"+key+".txt", "w")
    f = open("../data/model_build_outputs/parsed_files/route_"+str(i)+".txt", "w")
    f.write("number_of_clients\n")
    f.write(str(number_of_clients) + "\n")
    f.write("vehicle_capacity\n")
    f.write(str(vehicle_capacity) + "\n")
    f.write("depot_number\n")
    f.write(str(depot_number) + "\n")
    f.write("route_id\n")
    f.write(str(key) + "\n")
    f.write("travel_times\n")
    for k in travel_time_matrix:
        for j in k:
            f.write(str(j) + " ")
        f.write("\n")
    f.write("time_windows\n")
    for k in time_windows:
        f.write(str(k[0]) + " " + str(k[1]) + "\n")
    f.write("service_time\n")
    for k in planned_service_times:
        f.write(str(k) + "\n")
    f.write("dimensions\n")
    for k in dimensions:
        f.write(str(k) + "\n")
    f.write("latitude_longitude\n")
    for k in latlongs:
        f.write(str(k[0]) + " " + str(k[1]) + "\n")
    f.write("zone_id\n")
    for k in zone_ids:
        f.write(str(k) + "\n")
    
    f.close()
    
    with open("../data/model_build_outputs/travel_times_files/travel_times_route_"+str(i)+".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(travel_time_matrix)
    
    
    df = pd.DataFrame.from_dict(route_info, orient='index')
    travel_times = pd.DataFrame(travel_time_matrix[1:], columns=travel_time_matrix[0] )
    travel_times.set_index("key", inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index':'customer_id'}, inplace=True)
    df.sort_values(by=['order'], inplace=True, ignore_index=True)        
    #Calculate route times with HARD TIME WINDOWS
    number_of_waiting_times = 0
    number_of_late_times = 0
    capacity_used = 0
    df.at[0, 'visited_at'] = df.at[0, 'service_time'] 
    df.at[0, 'awaited_time_window'] = 0
    df.at[0, 'arrived_late'] = 0
    df.at[0, 'arrival_time'] = departure_time
    
    df.at[0, 'double_visit'] = "nan"
    df.at[0, 'has_reattempts'] = 0
    df['double_visit']=df['double_visit'].astype('str')
    for j in range(1, len(df)):
        
        arrival_time = df.at[j-1, 'visited_at'] + \
            travel_times[df.at[j, 'customer_id']][df.at[j-1, 'customer_id']] + \
                df.at[j-1, 'service_time']
        
        capacity_used += df.at[j, 'dimensions']
        
        if (arrival_time >= df.at[j, 'time_window_start_seconds']):
            df.at[j, 'visited_at'] = arrival_time
            df.at[j, 'awaited_time_window'] = 0
            if (arrival_time > df.at[j, 'time_window_end_seconds'])and(df.at[j, 'time_window_end_seconds'] != -1):
                df.at[j, 'arrived_late'] = 1
                number_of_late_times += 1
            else:            
                df.at[j, 'arrived_late'] = 0
        else:
            number_of_waiting_times += 1
            df.at[j, 'visited_at'] = df.at[j, 'time_window_start_seconds']
            df.at[j, 'awaited_time_window'] = 1
        
        df.at[j, 'arrival_time'] = departure_time + timedelta(seconds=df.at[j, 'visited_at'])
        df.at[j, 'double_visit'] = double_visits[df.at[j, 'customer_id']][0] if df.at[j, 'customer_id'] in double_visits else "nan"
        df.at[j, 'has_reattempts'] = 1 if df.at[j, 'customer_id'] in double_visits else 0
       
    df['has_reattempts'] = sum(df['has_reattempts'])/2
    df.set_index('customer_id', inplace=True)    
    df.to_csv("../data/model_build_outputs/df_routes_files/df_"+str(i)+".csv")
    return sum(df['has_reattempts'])/2


def read_json(number_of_records):
    parsing_time_start = time.time()
    print ("Reading files for parsing")
    with open('../data/model_build_inputs/travel_times.json') as f:
        travel_times_list = pd.DataFrame([json.loads(l) for l in f.readlines()])
    with open('../data/model_build_inputs/package_data.json') as f:
        package_data_list = pd.DataFrame([json.loads(l) for l in f.readlines()])
    with open('../data/model_build_inputs/route_data.json') as f:
        route_data_list = pd.DataFrame([json.loads(l) for l in f.readlines()])
    with open('../data/model_build_inputs/actual_sequences.json') as f:
        actual_sequences_list = pd.DataFrame([json.loads(l) for l in f.readlines()])
    
    parsing_time = time.time() - parsing_time_start
    print("Reading jsons time " + str(parsing_time))
    
    df_reattempts = pd.DataFrame(None) 
    df_reattempts.at[0, 'reattempts'] = 0
    total_number_routes = route_data_list.shape[1] if number_of_records == 0 else number_of_records
    print ("Parsing files into dataframes")
    parsing_time_start = time.time()
    for i in pbar(range(total_number_routes)):
    #for i in range(2167, 2170):
        #df = pd.DataFrame(None, columns=['latitude', 'longitude', 'time_window_start', 'time_window_end', ])
        
        key = package_data_list.keys()[i]
        #key = 'RouteID_fffd257c-3041-4736-be7a-5efea8af1173'
        number_of_clients = len(package_data_list[key][0].keys())
        
        vehicle_capacity = route_data_list[key][0]['executor_capacity_cm3']
        reattempts = parse_json(i, key, number_of_clients, vehicle_capacity, package_data_list[key][0], route_data_list[key][0], travel_times_list[key][0], actual_sequences_list[key][0])
        df_reattempts.at[i, 'reattempts'] = reattempts
    
    df_reattempts.to_csv("../data/model_build_outputs/df_has_reattempts.csv")
    print ("Parsing done")
    parsing_time = time.time() - parsing_time_start
    print("Parsing time " + str(parsing_time))
        
read_json(0)
#number_of_records = 100


