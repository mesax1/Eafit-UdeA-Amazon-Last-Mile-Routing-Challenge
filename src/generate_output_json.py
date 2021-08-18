from os import path, listdir
import sys, json, time
from csv import reader
import pandas as pd



if __name__ == "__main__":
    # Get Directory
    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
    
    # Read input data
    print('Reading Input Data')

    randomization = int(sys.argv[1])
    
    
    df_results = pd.DataFrame(None)
    df_results.at[0, 'route_sequence'] = 0
    new_routes_dict = {}
    na_values = [""]
    
    #number_of_routes = len(listdir('../data/model_build_outputs/df_routes_files'))
    number_of_routes = len(listdir('../data/model_apply_outputs/df_routes_files_val'))
    print(number_of_routes)
    for i in range(number_of_routes):
        try:
            df_1=pd.DataFrame(None)
            #df_1 = pd.read_csv("../data/model_build_outputs/df_routes_files/df_"+str(i)+".csv", na_values=na_values, keep_default_na=False)
            df_1 = pd.read_csv("../data/model_apply_outputs/df_routes_files_val/df_"+str(i)+".csv", na_values=na_values, keep_default_na=False)
            
            
           
            #with open("../data/model_build_outputs/grasp_routes_prob_random_"+str(randomization)+"/solution_route_"+str(i), 'r') as read_obj:
            with open("../data/model_apply_outputs/grasp_routes_prob_random_"+str(randomization)+"/solution_route_"+str(i), 'r') as read_obj:
                # pass the file object to reader() to get the reader object
                csv_reader = reader(read_obj, skipinitialspace=True,)
                # Iterate over each row in the csv using reader object
                for row in csv_reader:
                    # row variable is a list that represents a row in csv
                    #print(row)
                    grasp_route = row
            final_route = grasp_route[:-1] #Aqui se remueve el depot del final de la ruta antes de enviar la solucion final
            route_id = df_1['route_id'].values[1]
            
            counter = 0
            final_route_dict = {}
            for j in final_route:
                final_route_dict[j] = counter
                counter = counter+1
            #final_route_dict = {j:counter for (j, counter) in final_route}
            proposed_dict = {}
            proposed_dict['proposed'] = final_route_dict
            new_routes_dict[route_id] = proposed_dict
        except:
            pass
        
    print('Routes dict created')

    # Write output data
    #output_path=path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')
    output_path=path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')
    with open(output_path, 'w') as out_file:
        json.dump(new_routes_dict, out_file)
        print("Success: The '{}' file has been saved".format(output_path))

    print('Done!')
