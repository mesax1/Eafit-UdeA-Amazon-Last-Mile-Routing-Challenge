#!/bin/bash
set -eu

export PYTHONUNBUFFERED=1
#exec python src/model_apply.py
start=`date +%s`

cd src

TESTPHASE=1
#file="../data/model_build_outputs/best_randomization.csv"
#value=`cat $file`
#echo "$value"

#RANDOMIZATION="${value:-10}"
RANDOMIZATION=5

python faster_parser_model_apply_inputs.py $RANDOMIZATION

python create_haversine_dist_matrix.py $TESTPHASE

python create_test_heading_matrices.py

cd ..

folder_to_count=data/model_apply_outputs/df_routes_files_val

file_count=$(ls $folder_to_count | wc -l)
#file_count=$(($file_count+1))
echo $file_count files in $folder_to_count

mkdir -p data/model_apply_outputs/local_optima_analisis
mkdir -p data/model_apply_outputs/df_local_optima_analisis
mkdir -p data/model_apply_outputs/grasp_routes_prob_random_$RANDOMIZATION

cd src

cd GRASP-TSPTW
make clean
make

mv grasp_tw ..
cd ../..

cd src

for i in `seq 1 $file_count`
do
./grasp_tw $i $RANDOMIZATION $TESTPHASE
done

python generate_output_json.py $RANDOMIZATION

echo DONE EXECUTING GRASP RANDOMIZATION $RANDOMIZATION

end=`date +%s`

total_runtime=$(($end-$start))

echo TOTAL TIME IN SECONDS FOR MODEL_APPLY
echo $total_runtime
