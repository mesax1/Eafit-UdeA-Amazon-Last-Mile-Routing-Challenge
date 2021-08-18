#!/bin/bash
set -eu

export PYTHONUNBUFFERED=1
start=`date +%s`


TESTPHASE=0

cd src

python faster_parser.py
start_2=$(date +%s)

python create_haversine_dist_matrix.py $TESTPHASE
start_3=$(date +%s)

python create_all_train_heading_matrices.py
start_4=$(date +%s)

cd GRASP-TSPTW
make clean
make

mv grasp_tw ..
cd ../..

mkdir -p data/model_build_outputs/local_optima_analisis
mkdir -p data/model_build_outputs/df_local_optima_analisis


cd src

#Begin RANDOMIZATION parameter fitting

file_count=500
RANDOMIZATION=2
mkdir -p ../data/model_build_outputs/grasp_routes_prob_random_2
echo EXECUTING GRASP $RANDOMIZATION 
for i in `seq 1 $file_count`
do
./grasp_tw $i $RANDOMIZATION $TESTPHASE
done

RANDOMIZATION=3
mkdir -p ../data/model_build_outputs/grasp_routes_prob_random_3
echo EXECUTING GRASP $RANDOMIZATION 
for i in `seq 1 $file_count`
do
./grasp_tw $i $RANDOMIZATION $TESTPHASE
done

RANDOMIZATION=4
mkdir -p ../data/model_build_outputs/grasp_routes_prob_random_4
echo EXECUTING GRASP $RANDOMIZATION 
for i in `seq 1 $file_count`
do
./grasp_tw $i $RANDOMIZATION $TESTPHASE
done

RANDOMIZATION=5
mkdir -p ../data/model_build_outputs/grasp_routes_prob_random_5
echo EXECUTING GRASP $RANDOMIZATION 
for i in `seq 1 $file_count`
do
./grasp_tw $i $RANDOMIZATION $TESTPHASE
done

echo EVALUATING SCORES FOR DIFFERENT RANDOMIZATION VALUES
python prueba_scores_randomization.py 2 &
python prueba_scores_randomization.py 3 &
python prueba_scores_randomization.py 4 &
python prueba_scores_randomization.py 5 &
wait

python determine_best_randomization.py


start_5=$(date +%s)




<<'COMMENTS'

for i in `seq 1 $file_count`; do
    (
        # .. do your stuff here
        python prueba_scores_local_optimas_aws.py $i
    ) &

    # allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi
done
wait
COMMENTS


echo DONE EXECUTING GRASP RANDOMIZATION $RANDOMIZATION

end=`date +%s`

total_runtime=$(($end-$start))

parser_time=$(($start_2-$start))
haversine_time=$(($start_3-$start_2))
heading_time=$(($start_4-$start_3))
randomization_time=$(($start_5-$start_4))
parameter_tuning_time=$(($end-$start_5))

echo TOTAL TIME IN SECONDS FOR PARSING
echo $parser_time

echo TOTAL TIME IN SECONDS FOR HAVERSINE
echo $haversine_time

echo TOTAL TIME IN SECONDS FOR HEADING
echo $heading_time

echo TOTAL TIME IN SECONDS FOR RANDOMIZATION
echo $randomization_time

echo TOTAL TIME IN SECONDS FOR PARAMETER TUNING
echo $parameter_tuning_time

echo TOTAL TIME IN SECONDS FOR MODEL_BUILD
echo $total_runtime



echo DONE WITH MODEL_BUILD
