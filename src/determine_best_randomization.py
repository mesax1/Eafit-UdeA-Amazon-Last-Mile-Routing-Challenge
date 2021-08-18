import pandas as pd
from datetime import datetime
import numpy as np
import csv
import sys
from os import path, listdir
from pathlib import Path
from progressbar import progressbar as pbar

df_results = pd.DataFrame(None)


na_values = [""]
number_of_files = len(listdir("../data/model_build_outputs/scores"))
print(number_of_files)
randomization_list = [2,3,4,5]
for i in pbar(range(number_of_files)):
    try:
        randomization = randomization_list[i]
        df = pd.read_csv("../data/model_build_outputs/scores/df_results_prob_random_"+str(randomization)+".csv", na_values=na_values, keep_default_na=False)
        df_results = pd.concat([df], axis=0, ignore_index=True)
    except:
        pass
    
best_row = df_results.final_mean_score.idxmin()
best_randomization = [df_results.loc[best_row, 'randomization']]
print("best_randomization")
print(best_randomization)

output_path = "../data/model_build_outputs/best_randomization.csv"
with open(output_path, "w") as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(best_randomization)

print("Done selecting best_randomization")



