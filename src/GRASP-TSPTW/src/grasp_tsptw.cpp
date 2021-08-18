#include "metaheuristic/methods/giant_tour_creation.h"
#include "metaheuristic/methods/create_zone_matrix.h"
#include "metaheuristic/methods/neighborhood.h"
#include "metaheuristic/methods/vnd.h"
#include "metaheuristic/model/instance.h"
#include "metaheuristic/model/solution.h"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <omp.h>
#include <sstream>
#include <cmath>

#define length(x) (sizeof(x) / sizeof(x[0]))


using namespace std;

template < typename Type > std::string to_str (const Type & t)
{
    std::ostringstream os;
    os << t;
    return os.str ();
}

/*
 * RUN SEARCH IS THE MAIN ALGORITHM FUNCTION
 * INPUTS:
 * randomization --> int Indicate randomization value for construction of initial solutions with random_nearest_neighbor
 * seed --> int Random number generator state
 * instance_name --> string  "route_"+route_number+".txt";
 * in_test_phase --> 0 if currently Train phase, 1 if currently apply/test phase
 * neighborhood_combinations_sliced --> vector of vectors containing the Neighborhoods for VND
 * first_improvement --> Boolean indicating if first_improvement strategy or best_improvement strategy is used in the VND (Default=false, use best_improvement)
 * normalize --> Indicate if normalize costs of the objective function. Default=True
 */

void run_search(int randomization, int seed,  const string& instance_name, int in_test_phase,  vector<vector<TSPNeighborhood*>>neighborhood_combinations_sliced, bool first_improvement, bool normalize) {
    string direc_input = "";
    string direc_output = "";
    string timing_direc = "";
    string local_optima_direc = "";

    //WRITE OUTPUT CSV'S TO DIFFERENT FOLDERS, DEPENDING ON TRAINING PHASE OR APPLY PHASE
    if (in_test_phase == 0) {
        direc_input = "../data/model_build_outputs/parsed_files/";
        timing_direc = "../data/model_build_outputs/timing/";
        direc_output = "../data/model_build_outputs/grasp_routes_prob_random_"+ to_str(randomization)+"/";
        local_optima_direc = "../data/model_build_outputs/local_optima_analisis/";
    }
    else if (in_test_phase == 1) {
        direc_input = "../data/model_apply_outputs/parsed_files_val/";
        timing_direc = "../data/model_apply_outputs/timing/";
        direc_output = "../data/model_apply_outputs/grasp_routes_prob_random_"+ to_str(randomization)+"/";
        local_optima_direc = "../data/model_apply_outputs/local_optima_analisis/";

    }

    vector<string> splitting_result;
    boost::split(splitting_result, instance_name, boost::is_any_of("."));


    string global_results = direc_output +"solution_ " + splitting_result[0];


    vector<string> route_and_number;
    boost::split(route_and_number, splitting_result[0], boost::is_any_of("_"));
    TSPInstance instance = TSPInstance(direc_input + instance_name, route_and_number[1], in_test_phase);
    cout << "Parsing instance done" << endl;


    //ASSIGN RANDOM_SEED_STATE TO THE METHOD FOR CREATING RANDOM FURTHEST NEIGHBOR SOLUTION
    vector<TSPGiantTour *> giant_tour_heuristics;
    vector<TSPGiantTour *> furthest_heuristics = {new TSP_GiantTour_FN(seed)};  //Furthest neighbor for max initial_solution time

    // CREATE STRUCTURES FOR STORING INITIAL SOLUTIONS - CURRENT BEST SOLUTION - FINAL BEST SOLUTION
    //TSPSolution *giant_tour_solution;
    TSPSolution *initial_furthest_solution;
    TSPSolution *best_solution = new TSPSolution();
    TSPSolution *best_initial_solution = new TSPSolution();
    TSPSolution *train_best_initial_solution = new TSPSolution();
    TSPSolution *train_best_solution = new TSPSolution();
    bool is_train_best_solution_null = true;
    float train_best_solution_cost = std::numeric_limits<float>::max();
    LocalOptimaOutputs local_optima_storage;


    instance.normalize = normalize;

    if (in_test_phase == 1){
        instance.in_test_phase = true;
    }
    else
    {
        instance.in_test_phase = false;
    }

    // CREATE FURTHEST SOLUTION USING RANDOM FURTHEST NEIGHBOR HEURISTIC
    cout << "Computing initial solution" << endl;
    initial_furthest_solution = furthest_heuristics[0]->run(instance, randomization);
    float initial_furthest_cost = initial_furthest_solution->compute_cost(instance);
    cout << "Furthest cost: " << initial_furthest_cost << endl;
    // THIS COST OF THIS FURTHEST SOLUTION WILL BE USED TO NORMALIZE THE ROUTE-TIME COST OF THE OTHER SOLUTIONS GENERATED IN THIS GRASP
    instance.worst_initial_cost = initial_furthest_cost;

    //CREATE ZONE_TIME AND ZONE_DISTANCE PENALIZATION MATRICES
    TSPZoneMatrixSolution zones_info = create_zones_matrix(instance);

    //DETERMINE NUMBER OF GRASP ITERATIONS THAT WILL BE PERFORMED, ACCORDING TO NUMBER OF CUSTOMERS AND NUMBER OF ZONES IN THE ROUTE
    //int calc_iterations = instance.n_cust * log(instance.n_cust/(instance.zones_time_matrix_df.size()-1))/ sqrt(instance.zones_time_matrix_df.size()-1) ;
    //int number_of_zones = instance.zones_time_matrix_df.size()-1;
    //int total_iterations = max(calc_iterations,  number_of_zones);

    int total_iterations = omp_get_max_threads();
    cout << "number_of_grasp_iteration: " << total_iterations << endl;

    string local_optima_direction = local_optima_direc + "solution_" + splitting_result[0];
    int local_optima_counter = 0;
#pragma omp parallel
    {
        int seeds = omp_get_thread_num();

        //ASSIGN RANDOM_SEED_STATE TO EACH THREAD IN THE METHOD TO CREATE INITIAL SOLUTIONS WITH RANDOM NEAREST NEIGHBOR
        giant_tour_heuristics = {new TSPGiantTour_Prob_RNN(seeds)};
        bool is_best_solution_null = true;
        float best_solution_cost = std::numeric_limits<float>::max();

        //START GRASP ITERATIONS
#pragma omp for
        for (int it = 0; it < total_iterations; it++) {
            TSPSolution *giant_tour_solution;

            //CREATE INITIAL SOLUTION WITH CLUSTERED RANDOM NEAREST NEIGHBOR
            giant_tour_solution = giant_tour_heuristics[0]->run(instance, randomization);
            giant_tour_solution->cost = giant_tour_solution->compute_prob_cost(instance);

            float local_optima_cost;

            TSPSolution *local_optima;
            //local_optima = run_prob_tsp_vnd(*giant_tour_solution, instance, neighborhood_combinations_sliced[0], true);
            local_optima = run_vnd_generator(*giant_tour_solution, instance,  neighborhood_combinations_sliced[0], local_optima_direction, local_optima_counter, true);
            local_optima_cost = local_optima->cost;

            if (is_best_solution_null || local_optima_cost < best_solution_cost) {
                if (!is_best_solution_null) {
                    //delete best_solution;
                }
                //best_tsp_solution = giant_tour_solution;
                best_initial_solution = giant_tour_solution;
                best_solution = local_optima;

                best_solution_cost = local_optima_cost;
                is_best_solution_null = false;
            } else {
                //delete local_optima;
            }

#pragma omp critical
            {

                if (is_train_best_solution_null || best_solution_cost < train_best_solution_cost) {
                    train_best_solution = best_solution;
                    train_best_initial_solution = best_initial_solution;

                    train_best_solution_cost = best_solution_cost;
                    cout << train_best_solution_cost << endl;
                    is_train_best_solution_null = false;
                }
            }
        }

    }

    cout << "Final Cost: " << train_best_solution->cost << " Time Cost: " << train_best_solution->time_cost <<" Heading Cost: " << train_best_solution->heading_penalization_cost << " Angle cost: " << train_best_solution->angle_penalization_cost  <<endl;// << train_best_solution->time_window_cost << endl;
    cout << " Zone_time cost: " <<  train_best_solution->zone_time_penalization_cost << " Zone_distance cost: " <<  train_best_solution->zone_distance_penalization_cost  << " Time_window cost: " <<  train_best_solution->time_window_cost <<  endl; //best_solution->time_window_cost << endl;


    //global_output << train_best_solution->TSP_output_string_zones(instance).str() << endl;  //OUTPUT CSV WITH ZONES_IDS ACCORDING TO BEST SOLUTION FOUND
    ofstream global_output(global_results);
    global_output << train_best_solution->TSP_output_string_solutions(instance).str() << endl;
    // global_output << "Solution-ids: " << endl << train_best_solution->TSP_output_number_solutions().str() << endl;

    global_output.close();
}

int main(int argc, char* argv[])
{
    string route_number_str = argv[1];
    size_t pos;
    string n_randomization = argv[2];
    size_t pos9;
    int randomization = stoi(n_randomization, &pos9);
    int route = stoi(route_number_str, &pos) - 1; // Shell/bash/slurm job arrays start at 1, but routes start at 0
    string n_type = argv[3];
    size_t pos2;
    int in_test_phase = stoi(n_type, &pos2);
    int seed = 1;// = stoi(n_seed, &pos2);
    //int route=0;
    /*
    string p_beta = argv[5];
    size_t pos3;
    float penalization_beta = stof(p_beta, &pos3);
    */


    bool normalize = true;
    bool first_improvement = false;

    string instance_name = "route_"+to_string(route)+".txt";
    vector<TSPNeighborhood*> neighborhoods = {};

    int indexes[] = { 0, 1, 2, 3 };
    vector<string> neighborhoods_names = { "r", "s", "R", "S" };


    neighborhoods = {new TSPRelocateNeighborhood(first_improvement),
                     new TSPSwapNeighborhood(first_improvement),
                     new TSP_ProbZoneRelocateNeighborhood(first_improvement),
                     new TSP_ProbZoneSwapNeighborhood(first_improvement)
    };


    vector<vector<TSPNeighborhood*>> neighborhood_combinations;
    int until = 10;
    vector<vector<TSPNeighborhood*>> neighborhood_combinations_sliced;


    vector<vector<string>> neighborhood_names_combinations;
    vector<vector<string>> neighborhood_names_combinations_sliced;

    do {
        vector<TSPNeighborhood*> combination;
        vector<string> names_combination;
        for(unsigned int i = 0; i < neighborhoods_names.size(); i++) {
            combination.push_back(neighborhoods[indexes[i]]);
            names_combination.push_back(neighborhoods_names[indexes[i]]);
        }
        neighborhood_combinations.push_back(combination);
        neighborhood_names_combinations.push_back(names_combination);
    } while(std::next_permutation(begin(indexes), end(indexes)));

    for(unsigned int i = 0; i < neighborhood_combinations.size(); i += 1) {
        if (i == 0){
            neighborhood_combinations_sliced.push_back(neighborhood_combinations[i]);
            neighborhood_names_combinations_sliced.push_back(neighborhood_names_combinations[i]);
        }
    }

    cout << "In test phase type "<< in_test_phase << endl;

    run_search(randomization, seed, instance_name, in_test_phase,  neighborhood_combinations_sliced, first_improvement, normalize);

    cout << "Sali de run_search" << endl;
}
