#pragma once
#include "../model/solution.h"

class TSPNeighborhood{
public:
    bool first_improvement;
    std::string id;

    TSPNeighborhood(bool first_improvement = false, const std::string& id = "");
    ~TSPNeighborhood();

    virtual TSPSolution* search_neighbors(TSPSolution* initial_solution, const TSPInstance& instance, bool by_zones);
};

class TSPSwapNeighborhood: public TSPNeighborhood{
public:
    TSPSwapNeighborhood();

    TSPSwapNeighborhood(bool first_improvement = false);

    ~TSPSwapNeighborhood();

    TSPSolution* search_neighbors(TSPSolution* initial_solution, const TSPInstance& instance, bool  by_zones);
    TSPSolution* apply_best_move(TSPSolution* initial_solution, std::vector<int>& best_move, const TSPInstance& instance);
};

class TSPRelocateNeighborhood: public TSPNeighborhood{
public:
    TSPRelocateNeighborhood();

    TSPRelocateNeighborhood(bool first_improvement = false);

    ~TSPRelocateNeighborhood();

    TSPSolution* search_neighbors(TSPSolution* initial_solution, const TSPInstance& instance, bool  by_zones);
    TSPSolution* apply_best_move(TSPSolution* initial_solution, std::vector<int>& best_move, const TSPInstance& instance);
};

class TSP_ProbZoneSwapNeighborhood: public TSPNeighborhood{
public:
    TSP_ProbZoneSwapNeighborhood();

    TSP_ProbZoneSwapNeighborhood(bool first_improvement = false);

    ~TSP_ProbZoneSwapNeighborhood();

    TSPSolution* search_neighbors(TSPSolution* initial_solution, const TSPInstance& instance, bool  by_zones);
    TSPSolution* apply_best_move(TSPSolution* initial_solution, std::vector<int>& best_move, const TSPInstance& instance);
};

class TSP_ProbZoneRelocateNeighborhood: public TSPNeighborhood{
public:
    TSP_ProbZoneRelocateNeighborhood();

    TSP_ProbZoneRelocateNeighborhood(bool first_improvement = false);

    ~TSP_ProbZoneRelocateNeighborhood();

    TSPSolution* search_neighbors(TSPSolution* initial_solution, const TSPInstance& instance, bool  by_zones);
    TSPSolution* apply_best_move(TSPSolution* initial_solution, std::vector<int>& best_move, const TSPInstance& instance);
};