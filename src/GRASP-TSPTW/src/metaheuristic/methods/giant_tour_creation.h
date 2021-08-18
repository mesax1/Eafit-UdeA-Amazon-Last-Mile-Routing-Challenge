#pragma once
#include "../model/solution.h"
#include "../model/instance.h"


//STARTS TSP with TW structures

class TSPGiantTour {
public:
    TSPSolution& giant_tour_solution;
    TSPInstance instance;

    TSPGiantTour();
    TSPGiantTour(int seed);
    ~TSPGiantTour();

    virtual TSPSolution* run(const TSPInstance& instance, int alpha = 3);

};


class TSPGiantTour_Prob_RNN:public TSPGiantTour {
public:

    TSPGiantTour_Prob_RNN();
    TSPGiantTour_Prob_RNN(int seed);
    ~TSPGiantTour_Prob_RNN();

    TSPSolution* run(const TSPInstance& instance, int alpha = 3);
};

class TSP_GiantTour_FN:public TSPGiantTour {
public:

    TSP_GiantTour_FN();
    TSP_GiantTour_FN(int seed);
    ~TSP_GiantTour_FN();

    TSPSolution* run(const TSPInstance& instance, int alpha = 3);
};
