"""Pricing and optimization routines for the platform problem.

The optimizer implements a branch-and-price scheme to compute stable
matchings, prices, and intermediary profits on a network of farmers and
intermediaries.  The module contains helper classes for summarizing instance
runs, representing partial solutions, and performing column generation
via dual/primal models.  Most of the heavy lifting is done by
:class:`Optimizer`.
"""

import os
from farmers_intermediaries import Instance, Route, Farmer, Matching
import gurobipy as gp
import pickle
from copy import deepcopy
import time
from LP_solvers import VRPSolver
from dynamic_solvers import TSPSolver as dynamic_TSPSolver
import numpy as np
import json
from itertools import combinations
from pprint import pprint

def get_solver_threads() -> int:
    for var in ("SLURM_CPUS_PER_TASK", "GUROBI_THREADS"):
        value = os.environ.get(var)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return 1



class Branch:
    """Simple object representing a branching decision in the search tree.

    ``matching`` and ``unmatching`` are sets of intermediary identifiers that
    have been fixed to be included or excluded, respectively.  ``count_flag``
    controls whether the oracle call counter should be incremented when this
    branch is evaluated.
    """
    def __init__(self, matching, unmatching):
        self.matching = matching
        self.unmatching = unmatching
        self.count_flag = True


class InstanceSummary:
    """Data container for recording progress on a single instance run.

    Tracks timestamps, bound history, oracle calls, and final solutions.
    """
    def __init__(self, instance, parameters, solver_type):
        self.current_time = time.time()
        self.total_time = None
        self.instance = instance
        self.timestamps = []
        self.upper_bounds = []
        self.lower_bounds = []
        self.oracle_calls = []
        self.max_int_welf_sol = None
        self.min_int_welf_sol = None
        self.solver_type = solver_type
        self.total_oracle_calls = None
        self.parameters = parameters
        self.forced_lower_bound = None
        self.forced_upper_bound = None
        self.forced_cost = None

    def to_dict(self):
        return {
            "current_time": self.current_time,
            "total_time": self.total_time,
            "instance": self.instance.to_dict(),
            "timestamps": self.timestamps,
            "upper_bounds": self.upper_bounds,
            "lower_bounds": self.lower_bounds,
            "oracle_calls": self.oracle_calls,
            "max_int_welf_sol": self.max_int_welf_sol.return_dict() if self.max_int_welf_sol else None,
            "min_int_welf_sol": self.min_int_welf_sol.return_dict() if self.min_int_welf_sol else None,
            "solver_type": self.solver_type,
            "total_oracle_calls": self.total_oracle_calls,
            "het_costs": self.parameters["het_costs"],
            "epsilon": self.parameters["epsilon"],
            "forced_lower_bound": self.forced_lower_bound,
            "forced_upper_bound": self.forced_upper_bound,
            "forced_cost": self.forced_cost,
        }
        
    def save_to_json(self, file_path):
        data = self.to_dict()
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)


class Solution:
    """Encapsulates a pricing solution with farmer prices and intermediary
    profits.

    Provides convenience methods for computing welfare and converting to a
    dictionary for serialization.
    """
    def __init__(self, instance: Instance, matched_intermediaries: dict, farmer_prices: dict, intermediary_profits: dict, profit: float, matching_cost: float):
        self.instance = instance
        self.farmer_prices = farmer_prices
        self.intermediary_profits = intermediary_profits
        self.matched_intermediaries = matched_intermediaries
        self.profit = profit
        self.matched_set = set()
        self.matching_cost = matching_cost
        for int_id in matched_intermediaries:
            if matched_intermediaries[int_id] > 1-1e-9:
                self.matched_set.add(int_id)
            elif matched_intermediaries[int_id] < 1e-9:
                continue
            else:
                self.matched_set = None
                break
        self.price_per_quantity = -float('inf')
        self.price_per_mile_paved = -float('inf')
        self.price_per_mile_dirt = -float('inf')

    def farmer_welfare(self):
        return sum(self.farmer_prices[farmer.id] for farmer in self.instance.farmers)
    
    def intermediary_welfare(self):
        return sum(self.intermediary_profits[int_id] for int_id in self.intermediary_profits)
    
    def return_dict(self):
        data = {
            "farmer_prices": self.farmer_prices,
            "intermediary_profits": self.intermediary_profits,
            "matched_intermediaries": self.matched_intermediaries,
            "profit": self.profit,
            "matched_set": list(self.matched_set) if self.matched_set is not None else None,
            "matching_cost": self.matching_cost,
            "farmer_welfare": self.farmer_welfare(),
            "intermediary_welfare": self.intermediary_welfare(),
            "price_per_quantity": self.price_per_quantity,
            "price_per_mile_paved": self.price_per_mile_paved,
            "price_per_mile_dirt": self.price_per_mile_dirt,
        }
        return data
    

class PlatformSolution:
    """Represents a final platform outcome including payments and matching.

    This is a lightweight container used by higher-level scripts to record
    platform-wide results.
    """
    def __init__(self, instance: Instance, matching: Matching, farmer_payments: dict, intermediary_payments: dict):
        self.instance = instance
        self.matching = matching
        self.farmer_payments = farmer_payments
        self.intermediary_payments = intermediary_payments


class Optimizer:
    """Main optimization engine implementing branch-and-price for the
    platform model.

    The optimizer maintains a catalogue of matchings, solves primal and dual
    LPs, and handles branching decisions.  It exposes ``solve`` methods for
    heuristic and exact modes.
    """
    # Tolerance for row generation
    TOLERANCE = 1.0
    N_MATCHINGS = 20
    PAVED_THRESHOLD = 105000
    DIRT_THRESHOLD = 46000

    def __init__(self, instance: Instance, parameters: dict, base_matchings=None):
        
        self.instance = instance
        self.instance = instance

        if not parameters.get("disable_aggregation", False):
            self.feas_hist()
        else:
            self.validate_hist_sets()

        self.tsp_solver = dynamic_TSPSolver(self.instance)
        
        self.n_farmers = len(self.instance.farmers)
        self.n_intermediaries = len(self.instance.intermediaries)
        self.solver = parameters["solver"]

        if self.solver == "gurobi":
            self.vrp_solver = VRPSolver(self.instance)
        else:
            raise ValueError(f"Solver {self.solver} is not supported. Supported solvers are: 'gurobi'.")

        self.parameters = parameters
        self.het_costs = parameters["het_costs"]

        for int_id in self.het_costs:
            assert self.het_costs[int_id] + self.instance.truck_fixed_cost >= 0, f"Het cost for intermediary {int_id} is negative: {self.het_costs[int_id]}"
        
        self.dominance = self.calc_dominance()

        if base_matchings is not None:
            self.base_matchings = base_matchings
        else:
            self.base_matchings = self.initialize_base_matchings()
            
        self.all_matchings = self.initialize_all_matchings()
        self.all_routes = {}

        self.time_usage = {
            "tsp": 0,
            "adding_contraints_start": 0,
            "adding_contraints_found": 0,
            "solving": 0,
            "callback": 0,
            "total": 0,
        }


    def validate_hist_sets(self):
        for intermediary in self.instance.intermediaries:
            hist_sets = intermediary.additional_info.get("hist_sets", [])

            if not hist_sets:
                # Fallback: singleton empty sample
                intermediary.additional_info["hist_sets"] = [frozenset()]
            else:
                intermediary.additional_info["hist_sets"] = [
                    frozenset(h) for h in hist_sets
                ]

    def feas_hist(self):
        """Compute a single feasible history set for each intermediary.

        The original history sets may be large; this method reduces them to a
        single set containing all farmers currently assigned to that
        intermediary in the instance.  The result is stored back in
        ``intermediary.additional_info['hist_sets']``.
        """
        for intermediary in self.instance.intermediaries:
            feas_hist_sets = [set([f.id for f in self.instance.farmers if f.additional_info["intermediary_id"] == intermediary.id])]
            intermediary.additional_info["hist_sets"] = feas_hist_sets

    def calc_dominance(self):
        """Compute pairwise dominance relations between intermediaries.

        An intermediary ``i`` is said to dominate ``j`` if ``i`` has lower
        heterogeneous costs and at least as much expected historical fruit.
        The method returns a list of tuples *(i, j)* that satisfy this
        condition.  Dominance is used to enforce ordering constraints in the
        optimization models.
        """
        # optimize for speed - reduce recomputation of historical fruit

        # 1. Pre-compute historical fruit
        hist_fruit = {}
        for intermediary in self.instance.intermediaries:
            hist_sets = intermediary.additional_info["hist_sets"]

            int_hist_fruit = 0

            for hist_set in hist_sets:
                for farmer in self.instance.farmers:
                    if farmer.id in hist_set:
                        int_hist_fruit += farmer.quantity
            
            int_hist_fruit /= len(hist_sets)

            int_hist_fruit = 1 / len(intermediary.additional_info["hist_sets"]) *sum(sum(farmer.quantity for farmer in self.instance.farmers if farmer.id in hist_set) for hist_set in intermediary.additional_info["hist_sets"])
            hist_fruit[intermediary.id] = int_hist_fruit

        # 2. Compare intermediaries
        rels = []
        for intermediary1, intermediary2 in combinations(self.instance.intermediaries, 2):
            het_costs_1, het_costs_2 = self.het_costs[intermediary1.id], self.het_costs[intermediary2.id]
            hist_fruits_1, hist_fruits_2 = hist_fruit[intermediary1.id], hist_fruit[intermediary2.id]

            if het_costs_1 < het_costs_2 and hist_fruits_1 >= hist_fruits_2:
                rels.append((intermediary1.id, intermediary2.id))
            if het_costs_2 < het_costs_1 and hist_fruits_2 >= hist_fruits_1:
                rels.append((intermediary2.id, intermediary1.id))

    
        print(" Dominance relations ".center(80, '-'))
        pprint(rels)
        print()
        return rels

    def initialize_base_matchings(self):
        """Compute an initial set of matchings by solving simple VRPs.

        The method solves a sequence of vehicle routing problems with
        increasing minimum truck counts to populate ``self.base_matchings``.
        These provide starting points for column generation.  The results are
        also used to determine ``min_trucks`` and ``max_trucks``.
        """

        all_matchings = {}
        # We solve a VRP with 1 to n_intermediaries
        print(" Solving VRP ".center(80, '-'))

        if self.solver == "gurobi":
            min_cost_matching = self.vrp_solver.solve(1, self.n_intermediaries)
            print(f"\tObjective: {min_cost_matching.cost}")
            
            # We initialize the routing costs
            all_matchings[len(min_cost_matching.routes)] = min_cost_matching

            for lower_n_intermediaries in range(len(min_cost_matching.routes) + 1, min(len(min_cost_matching.routes) + self.N_MATCHINGS, self.n_intermediaries + 1)):
                print(f"Solving VRP with at least {lower_n_intermediaries} intermediaries")
                #matching = self.vrp_solver.solve(lower_n_intermediaries, self.n_intermediaries)
                matching = deepcopy(min_cost_matching)
                matching.cost = matching.cost + (lower_n_intermediaries-len(min_cost_matching.routes)) * self.instance.truck_fixed_cost
                all_matchings[lower_n_intermediaries] = matching
                print(f"\tObjective: {matching.cost}")
            print()
        elif self.solver == "ORTools":
            min_cost_matching = self.vrp_solver.solve()
            print(f"\tObjective: {min_cost_matching.cost}")
            all_matchings[len(min_cost_matching.routes)] = min_cost_matching
        
            
        self.min_trucks = min(all_matchings.keys())
        self.max_trucks = max(all_matchings.keys())

        matchings_cost = {k: v.cost for k, v in all_matchings.items()}
        print(' Matchings cost '.center(80, '-'))
        pprint(matchings_cost)
        print()
        return all_matchings
    
    def initialize_all_matchings(self):
        """Initialize the catalogue of all explored matchings.

        Begins with the cheapest matching for the minimum truck count and
        records its cost.  Additional sets will be added during column
        generation.
        """
        min_length = min(self.base_matchings.keys())
        # Order trucks by distance to mill
        ordered_intermediaries = sorted(self.instance.intermediaries, key=lambda x: self.het_costs[x.id], reverse=False)
        min_set = set([intermediary.id for intermediary in ordered_intermediaries[:min_length]])
        return {frozenset(min_set): self.matching_cost(min_set)}
    
    def matching_cost(self, int_set):
        """Return the total cost of a given set of intermediaries.

        Combines the base VRP cost for the appropriate truck count with the
        heterogeneous costs of the selected intermediaries.
        """
        base_cost = self.base_matchings[len(int_set)].cost
        for int_id in int_set:
            base_cost += self.het_costs[int_id]
        return base_cost

    def add_cuts(self, model, eta, kappa, farmer_prices, route, add_info=None):
        """Add stability cuts to the primal model for a given route.

        Each intermediary's constraints are augmented to prevent profitable
        deviations along ``route`` given the current dual variables.  The
        optional ``add_info`` dictionary may include values for debugging or
        incremental updates.
        """
        for intermediary in self.instance.intermediaries:
            for hist_set_index, hist_set in enumerate(intermediary.additional_info["hist_sets"]):
                left_cut = (
                    route.value 
                    - self.het_costs[intermediary.id] - gp.quicksum(farmer_prices[f.id] for f in route.farmers)
                    - eta[intermediary.id] * sum(farmer.quantity for farmer in route.farmers if farmer.id not in hist_set) 
                    - kappa[intermediary.id, hist_set_index]
                )
            
                model.addConstr(left_cut<=0)

    def solve_branch_exact(self, branch):
        """Perform an exact branch-and-price iteration on a given branch.

        Parameters
        ----------
        branch : Branch
            Branching restrictions to apply (fixed matches/unmatches).

        Returns
        -------
        dict
            Information about the branch outcome including status codes,
            potential branching variable, and bound values.
        """
        print(f"Solving branch with matching {branch.matching} and unmatching {branch.unmatching}")

        min_cost_set, _, min_cost = self.prize_matching({int_id: 0 for int_id in self.intermediary_ids}, branch)
        branch.min_cost_set = min_cost_set
        branch.min_cost = min_cost
        if min_cost_set not in self.all_matchings:
            self.all_matchings[min_cost_set] = min_cost

        # 1. compute lower bound LB^n
        forced_lb = self.solve_primal(branch, "forced_lower_bound")
        print(f"Forced LB: {forced_lb['profit']}")
        print("Min Cost Set:")
        pprint(branch.min_cost_set)
        print()

        if forced_lb["profit"] > self.best_lb:
            self.best_lb = forced_lb["profit"]
            self.best_lb_set = branch.min_cost_set
            self.best_lb_summary = forced_lb

            self.instance_summary.lower_bounds.append(self.best_lb)
            self.instance_summary.upper_bounds.append(self.best_ub)
            self.instance_summary.timestamps.append(time.time() - self.instance_summary.current_time)
            self.instance_summary.oracle_calls.append(self.oracle_calls)

            print("Found a better lower bound through forcing, updating best lower bound to ", forced_lb["profit"], "new gap:", (self.best_ub - self.best_lb) / np.abs(self.best_lb))

        # 2. compute upper bound UB^n
        forced_ub = self.solve_primal(branch, "forced_upper_bound")
        print(f"Forced UB: {forced_ub['profit']}")
        print("Min Cost Set:")
        pprint(branch.min_cost_set)
        print()

        if forced_ub["profit"] < self.best_lb + Optimizer.TOLERANCE:
            print("\tForced upper bound is less than best lower bound, stopping branch")
            return {"status": "stop"}

        iteration = 0
        while True:
            print(f"\tStarting exact iteration {iteration}")
            dual_sol = self.solve_dual(branch)
            primal_sol = self.solve_primal(branch, "exact")

            if dual_sol["n_added_cols"] == 0 and primal_sol["n_cuts"] == 0:
                break
            else:
                iteration += 1
        
        print(f"Dual solution: {dual_sol['profit']}, Primal solution: {primal_sol['profit']}")
        print("Matching probabilities:", primal_sol["matching_probs"])

        if primal_sol["profit"] < self.best_lb + Optimizer.TOLERANCE:
            print("\tPrimal solution is less than best lower bound, stopping branch")
            return {"status": "stop"}
        else:
            integral = True
            for int_id in primal_sol["matching_probs"]:
                if primal_sol["matching_probs"][int_id] > 1e-9 and primal_sol["matching_probs"][int_id] < 1-1e-9:
                    integral = False
                    break
            if integral:
                self.best_lb = primal_sol["profit"]
                self.best_lb_set = set([int_id for int_id in primal_sol["matching_probs"] if primal_sol["matching_probs"][int_id] > 1-1e-9])
                self.best_lb_summary = primal_sol

                self.instance_summary.lower_bounds.append(self.best_lb)
                self.instance_summary.upper_bounds.append(self.best_ub)
                self.instance_summary.timestamps.append(time.time() - self.instance_summary.current_time)
                self.instance_summary.oracle_calls.append(self.oracle_calls)

                print("Found an integral solution through exact solving, updating best lower bound to ", primal_sol["profit"], "new gap:", (self.best_ub - self.best_lb) / np.abs(self.best_lb))
                
                return {"status": "integral"}
            else:
                dist_to_middle = {int_id: np.abs(0.5 - primal_sol["matching_probs"][int_id]) for int_id in primal_sol["matching_probs"]}
                branch_on = min(dist_to_middle, key=dist_to_middle.get)
                return {"status": "fractional", "branch_on":branch_on, "branch_value": primal_sol["matching_probs"][branch_on], "upper_bound": primal_sol["profit"], "branch": branch}

    def solve_branch_heuristic(self, branch, optimize):
        """Heuristic version of branch evaluation used in greedy search.

        ``optimize`` toggles whether to use the updated intermediary profits.
        Returns a dict similar to :meth:`solve_branch_exact` but may propose
        heuristic branching decisions.
        """
        print("Matching:")
        pprint(branch.matching)
        print("Unmatching:")
        pprint(branch.unmatching)
        print()

        min_cost_set, _, min_cost = self.prize_matching({int_id: 0 for int_id in self.intermediary_ids}, branch)
        branch.min_cost_set = min_cost_set
        branch.min_cost = min_cost
        if min_cost_set not in self.all_matchings:
            self.all_matchings[min_cost_set] = min_cost

        forced_lb = self.solve_primal(branch, "forced_lower_bound")
        if len(branch.matching) == 0 and len(branch.unmatching) == 0:
            self.instance_summary.forced_lower_bound = forced_lb["profit"]
            self.instance_summary.forced_cost = forced_lb["matching_cost"]

        print(f"Forced LB: {forced_lb['profit']}")
        print("Min Cost Set:")
        pprint(branch.min_cost_set)
        print()

        if forced_lb["profit"] > self.best_lb:
            self.best_lb = forced_lb["profit"]
            self.best_lb_set = branch.min_cost_set
            self.best_lb_summary = forced_lb

            self.instance_summary.lower_bounds.append(self.best_lb)
            self.instance_summary.upper_bounds.append(self.best_ub)
            self.instance_summary.timestamps.append(time.time() - self.instance_summary.current_time)
            self.instance_summary.oracle_calls.append(self.oracle_calls)

            print("Found a better lower bound through forcing, updating best lower bound to ", forced_lb["profit"], "new gap:", (self.best_ub - self.best_lb) / np.abs(self.best_lb))

        forced_ub = self.solve_primal(branch, "forced_upper_bound")
        if len(branch.matching) == 0 and len(branch.unmatching) == 0:
            self.instance_summary.forced_upper_bound = forced_ub["profit"]

        print(f"Forced UB {forced_ub['profit']}")
        print("Min Cost Set:")
        pprint(branch.min_cost_set)
        print()

        if forced_ub["profit"] < self.best_lb + Optimizer.TOLERANCE:
            print("\tForced upper bound is less than best lower bound, stopping branch\n")
            return {"status": "stop"}
        
        if optimize:
            int_profits = forced_ub["updated_int_profits"]
        else:
            int_profits = {int_id: np.random.uniform(0, 1) if forced_ub["updated_int_profits"][int_id]>Optimizer.TOLERANCE else 0 for int_id in self.intermediary_ids}
        max_profit = -float('inf')
        for int_id in int_profits:
            if int_id not in branch.min_cost_set and int_id not in branch.matching and int_id not in branch.unmatching:
                if int_profits[int_id] > max_profit:
                    max_profit = int_profits[int_id]
                    branch_on = int_id
        
        return {"status": "heuristic", "branch_on": branch_on, "branch_profits": int_profits, "upper_bound": forced_ub["profit"], "branch": branch}

    def solve_heuristic(self, optimize):
        """Entry point for a heuristic branch-and-price search.

        Parameters
        ----------
        optimize : bool
            If ``True`` perform profit optimization during branching, otherwise
            use random heuristics.

        Returns
        -------
        InstanceSummary
            Summary of the best lower-bound solution found.
        """
        farmer_ids = [farmer.id for farmer in self.instance.farmers]
        intermediary_ids = [intermediary.id for intermediary in self.instance.intermediaries]
        self.farmer_ids = farmer_ids
        self.intermediary_ids = intermediary_ids
        self.best_lb = -float('inf')
        self.best_ub = float('inf')

        root_branch = Branch(set(), set())
        branches_queue = [root_branch]
        active_branches = []

        while True:
            for branch in branches_queue:
                branch_sol = self.solve_branch_heuristic(branch, optimize)
                if branch_sol["status"] in ["stop","integral"]:
                    continue
                elif branch_sol["status"] in ["heuristic"]:
                    active_branches.append(branch_sol)
            
            if not active_branches or (("early_stop" in self.options) and self.options["early_stop"]):
                print("No more active branches, stopping\n")
                break
            print(f"Currently a total of {len(active_branches)} branches in the queue")
            print(f"Upper bounds of active branches: {[branch['branch_profits'][branch['branch_on']] for branch in active_branches]}")

            for branch in active_branches:
                if branch["upper_bound"] < self.best_lb:
                    active_branches.remove(branch)
            if not active_branches:
                print("No more active branches with upper bound greater than best lower bound, stopping\n")
                break

            max_branch = max(active_branches, key=lambda x: x["branch_profits"][x["branch_on"]])

            current_max_upper_bound = -float('inf')
            for branch in active_branches:
                if branch["upper_bound"] > current_max_upper_bound:
                    current_max_upper_bound = branch["upper_bound"]
            if current_max_upper_bound < self.best_ub:
                self.best_ub = current_max_upper_bound

                self.instance_summary.lower_bounds.append(self.best_lb)
                self.instance_summary.upper_bounds.append(self.best_ub)
                self.instance_summary.timestamps.append(time.time() - self.instance_summary.current_time)
                self.instance_summary.oracle_calls.append(self.oracle_calls)

                print(f"New upper bound: {self.best_ub}, current gap: {(self.best_ub - self.best_lb) / np.abs(self.best_lb)}")
                print("Summary:", self.instance_summary.upper_bounds)
                print()
                
            # pop the max branch from the active branches
            father_branch = max_branch["branch"]
            active_branches.remove(max_branch)

            for branch in active_branches:
                if branch["upper_bound"] < self.best_lb:
                    active_branches.remove(branch)

            print(f" Branching on {max_branch['branch_on']} with value = {max_branch['branch_profits'][max_branch['branch_on']]} ".center(80, '-'))
            left_branch = Branch(father_branch.matching.union({max_branch["branch_on"]}), father_branch.unmatching)
            right_branch = Branch(father_branch.matching, father_branch.unmatching.union({max_branch["branch_on"]}))
            right_branch.count_flag = False

            branches_queue = [left_branch, right_branch]

        return self.best_lb_summary

    def solve(self, solver, options={}):
        """Public entrypoint to run optimization with a given strategy.

        Parameters
        ----------
        solver : {'heuristic_unoptimized','exact','heuristic_optimized'}
            Choice of algorithm to execute.
        options : dict, optional
            Additional flags controlling price structure and dominance rules.

        Returns
        -------
        InstanceSummary
            Object containing run statistics and final solution summaries.
        """
        self.oracle_calls = 0
        self.options = options
        self.instance_summary = InstanceSummary(self.instance, self.parameters, solver)
        if solver == "heuristic_unoptimized":
            best_lb = self.solve_heuristic(optimize=False)
        elif solver == "exact":
            best_lb = self.solve_exact()
        elif solver == "heuristic_optimized":
            best_lb = self.solve_heuristic(optimize=True)

        self.instance_summary.total_time = time.time() - self.instance_summary.current_time
        self.instance_summary.max_int_welf_sol = best_lb["solution_max_int_welfare"]
        self.instance_summary.min_int_welf_sol = best_lb["solution_min_int_welfare"]
        self.instance_summary.total_oracle_calls = self.oracle_calls

        return self.instance_summary

    def solve_exact(self):
        """Perform a full exact branch-and-price solver.

        This method orchestrates branching on fractional variables and
        maintains global best bounds, returning a summary of the best
        integral solution found.
        """
        farmer_ids = [farmer.id for farmer in self.instance.farmers]
        intermediary_ids = [intermediary.id for intermediary in self.instance.intermediaries]
        self.farmer_ids = farmer_ids
        self.intermediary_ids = intermediary_ids

        self.best_lb = -float('inf')
        self.best_ub = float('inf')

        root_branch = Branch(set(), set())
        branches_queue = [root_branch] # queue prioritized by optimality gap
        active_branches = []
        while True:
            for branch in branches_queue:
                branch_sol = self.solve_branch_exact(branch)
                if branch_sol["status"] in ["stop","integral"]:
                    continue
                elif branch_sol["status"] in ["fractional"]:
                    active_branches.append(branch_sol)
            
            if not active_branches:
                print("No more active branches, stopping\n")
                break
            print(f"Currently a total of {len(active_branches)} branches in the queue")
            print(f"Upper bounds of active branches: {[branch['upper_bound'] for branch in active_branches]}")


            max_branch = max(active_branches, key=lambda x: x["upper_bound"])
            if max_branch["upper_bound"] < self.best_ub:
                self.best_ub = max_branch["upper_bound"]

                self.instance_summary.lower_bounds.append(self.best_lb)
                self.instance_summary.upper_bounds.append(self.best_ub)
                self.instance_summary.timestamps.append(time.time() - self.instance_summary.current_time)
                self.instance_summary.oracle_calls.append(self.oracle_calls)

                print(f"New upper bound: {max_branch['upper_bound']}, current gap: {(self.best_ub - self.best_lb) / np.abs(self.best_lb)}")
            # pop the max branch from the active branches
            father_branch = max_branch["branch"]
            active_branches.remove(max_branch)

            # prune node if LB^n > UB
            for branch in active_branches:
                if branch["upper_bound"] < self.best_lb:
                    active_branches.remove(branch)

            print(f"Branching on {max_branch['branch_on']} with value {max_branch['branch_value']}")
            left_branch = Branch(father_branch.matching.union({max_branch["branch_on"]}), father_branch.unmatching)
            right_branch = Branch(father_branch.matching, father_branch.unmatching.union({max_branch["branch_on"]}))
            branches_queue = [left_branch, right_branch]

        return self.best_lb_summary

    def prize_matching(self, prizes, branch):
        """Compute the best intermediary set given prize values and branch.

        Solves a simplified selection problem over ``prizes`` taking into
        account forced matches/unmatches.  Also increments the oracle counter
        if appropriate.

        Returns
        -------
        max_int_set : frozenset
            Best set of intermediaries.
        max_obj : float
            Objective value corresponding to that set.
        max_cost : float
            Cost associated with the set.
        """

        if branch.count_flag:
            self.oracle_calls += 1


        net_prizes = {int_id: prizes[int_id] - self.het_costs[int_id] for int_id in prizes.keys()}

        mandatory_intermediaries = set(branch.matching)
        mandatory_prizes_sum = sum(net_prizes[int_id] for int_id in mandatory_intermediaries)
        len_mandatory = len(mandatory_intermediaries)
        for int_id in branch.unmatching:
            del net_prizes[int_id]
        for int_id in mandatory_intermediaries:
            del net_prizes[int_id]
        
        ordered_ints = sorted(net_prizes.keys(), key=lambda x: net_prizes[x], reverse=True)
        objs = {}
        for n_trucks in self.base_matchings:
            if n_trucks >= len_mandatory:
                prizes_sum = sum(net_prizes[int_id] for int_id in ordered_ints[:n_trucks-len_mandatory])
                objs[frozenset(set(ordered_ints[:n_trucks-len_mandatory]).union(mandatory_intermediaries))] = prizes_sum + mandatory_prizes_sum - self.base_matchings[n_trucks].cost


        max_obj = max(objs.values())
        max_int_set = [int_set for int_set, obj in objs.items() if obj == max_obj]
        max_cost = self.base_matchings[len(max_int_set[0])].cost + sum(self.het_costs[int_id] for int_id in max_int_set[0])

        return max_int_set[0], max_obj, max_cost
    
    def valid_matching(self, branch, int_set):
        """Check whether a candidate matching respects a branch's restrictions.

        Returns ``True`` if ``int_set`` does not violate any forced match or
        unmatch assignments, ``False`` otherwise.
        """
        for int_id in branch.unmatching:
            if int_id in int_set:
                return False
        for int_id in branch.matching:
            if int_id not in int_set:
                return False
        return True

    def solve_dual(self, branch: Branch):
        """Construct and solve the dual LP for a given branch.

        The dual problem is used to generate new columns (matchings) via a
        pricing subproblem.  It returns the objective value and the number of
        columns added.
        """

        model = gp.Model("Dual")
        model.setParam("Threads", get_solver_threads())
        model.setParam('OutputFlag', 0)

        beta = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-float('inf'), name="beta")
        lamb = model.addVars(self.intermediary_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="lamb")
        mu = model.addVars(self.intermediary_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="mu")
        alpha = model.addVars(
            [(intermediary.id, hist_set_index, route_set_index) 
             for intermediary in self.instance.intermediaries
             for hist_set_index in range(len(intermediary.additional_info["hist_sets"])) 
             for route_set_index in self.all_routes],
            vtype=gp.GRB.CONTINUOUS, lb=0.0, name="alpha"
        )

        if self.options:
            if self.options["structured_farmer_prices"]:
                gamma = model.addVars(self.farmer_ids, vtype=gp.GRB.CONTINUOUS, lb=-float('inf'), name="gamma")
                model.addConstr(gp.quicksum(gamma[farmer.id] for farmer in self.instance.farmers) == 0, "gamma_sum_zero")
                model.addConstr(gp.quicksum(-gamma[farmer.id] * farmer.quantity for farmer in self.instance.farmers) <= 0, "gamma_quantity_zero")
                model.addConstr(gp.quicksum(gamma[farmer.id] * (farmer.paved_to_mill > self.PAVED_THRESHOLD) * farmer.quantity for farmer in self.instance.farmers) <= 0, "gamma_dist_zero_1")
                model.addConstr(gp.quicksum(gamma[farmer.id] * (farmer.dirt_to_mill > self.DIRT_THRESHOLD) * farmer.quantity for farmer in self.instance.farmers) <= 0, "gamma_dist_zero_2")
            if self.options["domination"]:
                D = model.addVars(self.dominance, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="D")


        objective = sum(self.instance.fruit_price * farmer.quantity for farmer in self.instance.farmers) + beta + gp.quicksum(alpha[intermediary.id, hist_set_index, route_set_index] * (self.all_routes[route_set_index].cost + self.het_costs[intermediary.id] - sum(self.instance.fruit_price * farmer.quantity for farmer in self.all_routes[route_set_index].farmers)) for intermediary in self.instance.intermediaries for hist_set_index in range(len(intermediary.additional_info["hist_sets"])) for route_set_index in self.all_routes)

        # Add constraints for the dual variables
        if self.options["domination"]:
            for intermediary in self.instance.intermediaries:
                sum_dominating = 0
                sum_dominated = 0
                for (int_id1, int_id2) in self.dominance:
                    if int_id1 == intermediary.id:
                        sum_dominating += D[int_id1, int_id2]
                    elif int_id2 == intermediary.id:
                        sum_dominated += D[int_id1, int_id2]
                model.addConstr(-1 + mu[intermediary.id] - lamb[intermediary.id] + sum_dominating - sum_dominated <= 0, f"domination_constraint_{intermediary.id}")                
        else:
            model.addConstrs(-1 + mu[intermediary.id] - lamb[intermediary.id]<= 0 for intermediary in self.instance.intermediaries)
        model.addConstrs(-self.parameters["epsilon"][intermediary.id]*mu[intermediary.id] + gp.quicksum(alpha[intermediary.id, hist_set_index, route_set_index]*sum(farmer.quantity for farmer in self.all_routes[route_set_index].farmers if farmer.id not in intermediary.additional_info["hist_sets"][hist_set_index]) for hist_set_index in range(len(intermediary.additional_info["hist_sets"])) for route_set_index in self.all_routes) <= 0 for intermediary in self.instance.intermediaries)
        model.addConstrs(-self.all_matchings[int_set_index] + gp.quicksum(lamb[intermediary_id] * self.instance.truck_capacity * self.instance.fruit_price for intermediary_id in int_set_index) - beta <= 0 for int_set_index in self.all_matchings if self.valid_matching(branch, int_set_index))
        
        if self.options:
            if self.options["structured_farmer_prices"]:
                model.addConstrs(gamma[farmer.id] -1 + gp.quicksum(alpha[intermediary.id, hist_set_index, route_set_index]*(farmer.id in route_set_index) for intermediary in self.instance.intermediaries for hist_set_index in range(len(intermediary.additional_info["hist_sets"])) for route_set_index in self.all_routes) <= 0 for farmer in self.instance.farmers)
            else:
                model.addConstrs(-1 + gp.quicksum(alpha[intermediary.id, hist_set_index, route_set_index]*(farmer.id in route_set_index) for intermediary in self.instance.intermediaries for hist_set_index in range(len(intermediary.additional_info["hist_sets"])) for route_set_index in self.all_routes) <= 0 for farmer in self.instance.farmers)
        model.addConstrs(-1/len(intermediary.additional_info["hist_sets"]) * mu[intermediary.id] + gp.quicksum(alpha[intermediary.id, hist_set_index, route_set_index] for route_set_index in self.all_routes) == 0 for intermediary in self.instance.intermediaries  for hist_set_index in range(len(intermediary.additional_info["hist_sets"])))

        model.setObjective(objective, gp.GRB.MINIMIZE)
        model.update()
        model.optimize()
        added_cols = 0

        # We prize new matchings
        while True:
            prizes = {self.intermediary_ids[i]: lamb[self.intermediary_ids[i]].X * self.instance.truck_capacity * self.instance.fruit_price for i in range(len(self.intermediary_ids))}
            max_set, max_obj, max_cost = self.prize_matching(prizes, branch)
            if max_obj > beta.X + Optimizer.TOLERANCE:
                constr = gp.quicksum(lamb[int_id] * self.instance.truck_capacity * self.instance.fruit_price for int_id in max_set) - beta - max_cost <= 0
                assert max_set not in self.all_matchings, f"Matching {max_set} already exists"
                assert self.valid_matching(branch, max_set), f"Matching {max_set} is not valid for branch {branch}"
                self.all_matchings[frozenset(max_set)] = max_cost
                model.addConstr(constr, "new_matching_constraint")
                added_cols += 1
                model.optimize()
            else:
                break

        print(f"\tDual yielded {added_cols} new columns, objective: {model.ObjVal}")

        return {
            "profit": model.ObjVal,
            "n_added_cols": added_cols,
        }

    def solve_primal(self, branch: Branch, sol_type):
        """Solve the primal LP under given branch and solution type.

        ``sol_type`` may be ``'exact'``, ``'forced_lower_bound'`` or
        ``'forced_upper_bound'`` and controls additional constraints.  The
        method returns a dictionary summarizing the solution including profits
        and cut information.
        """

        valid_matchings = [int_set for int_set in self.all_matchings if self.valid_matching(branch, int_set)]

        initial_time = time.time()

        # Create a new model
        model = gp.Model("Primal")
        model.setParam("Threads", get_solver_threads())
        model.setParam('OutputFlag', 0)

        gap_ratio = sum(farmer.quantity for farmer in self.instance.farmers) / self.instance.truck_capacity

        
        # Create pricing variables for each farmer
        farmer_prices = model.addVars(self.farmer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="farmer_price")
        intermediary_profits = model.addVars(self.intermediary_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="intermediary_profit")

        if self.options:
            if self.options["structured_farmer_prices"]:
                price_per_quantity = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name="price_per_quantity")
                price_per_mile_paved = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name="price_per_mile_paved")
                price_per_mile_dirt = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name="price_per_mile_dirt")

                fixed_price = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-float('inf'), name="fixed_price")
                for farmer in self.instance.farmers:
                    model.addConstr(farmer_prices[farmer.id] == price_per_quantity * farmer.quantity - price_per_mile_paved * (farmer.paved_to_mill > self.PAVED_THRESHOLD) * farmer.quantity - price_per_mile_dirt * (farmer.dirt_to_mill > self.DIRT_THRESHOLD) * farmer.quantity + fixed_price, f"farmer_price_{farmer.id}")
        if self.options["domination"]:
            for (int_id1, int_id2) in self.dominance:
                model.addConstr(intermediary_profits[int_id1] >= intermediary_profits[int_id2], f"domination_constraint_{int_id1}_{int_id2}")


        matching_used = model.addVars(list(range(len(valid_matchings))), vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name="matching_used")
        model.addConstr(gp.quicksum(matching_used[k] for k in range(len(valid_matchings))) == 1, "matching_used_sum")
        matching_cost = gp.quicksum(matching_used[k] * self.all_matchings[valid_matchings[k]] for k in range(len(valid_matchings)))

        intermediary_matched = {}
        for int_id in self.intermediary_ids:
            intermediary_matched[int_id] = gp.quicksum(matching_used[k] for k in range(len(valid_matchings)) if int_id in valid_matchings[k])

        for int_id in branch.matching:
            model.addConstr(intermediary_matched[int_id] == 1, f"intermediary_matched_{int_id}")
        for int_id in branch.unmatching:
            model.addConstr(intermediary_matched[int_id] == 0, f"intermediary_unmatched_{int_id}")

        if sol_type in ["exact", "forced_lower_bound"]:
            model.addConstrs(intermediary_profits[intermediary.id] <= intermediary_matched[intermediary.id] * intermediary.capacity * self.instance.fruit_price for intermediary in self.instance.intermediaries)

        model.addConstrs(intermediary_profits[int_id] <=0 for int_id in branch.unmatching)

        if sol_type in ["forced_upper_bound", "forced_lower_bound"]:
            index_min_cost_set = valid_matchings.index(branch.min_cost_set)
            model.addConstr(matching_used[index_min_cost_set] == 1, "forced_matching_used")


        fruit_value = sum(farmer.quantity * self.instance.fruit_price for farmer in self.instance.farmers)

        profit = fruit_value - \
                sum(farmer_prices[farmer.id] for farmer in self.instance.farmers) - \
                sum(intermediary_profits[intermediary.id] for intermediary in self.instance.intermediaries) - \
                matching_cost
        
        # Extract epsilon
        epsilon = self.parameters["epsilon"]

        # Add dual variables
        eta = model.addVars(self.intermediary_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="eta")
        kappa = model.addVars(
            [(intermediary.id, hist_set_index) for intermediary in self.instance.intermediaries for hist_set_index in range(len(intermediary.additional_info["hist_sets"]))], 
            vtype=gp.GRB.CONTINUOUS, lb=-float("inf"), name="kappa"
        )

        # Add stability contraints
        for intermediary in self.instance.intermediaries:
            model.addConstr(intermediary_profits[intermediary.id] >= eta[intermediary.id] * epsilon[intermediary.id] + 1/len(intermediary.additional_info["hist_sets"]) * gp.quicksum(kappa[intermediary.id, hist_set_index] for hist_set_index in range(len(intermediary.additional_info["hist_sets"]))), f"stability_{intermediary.id}")
        
        # Add cuts
        for route in self.all_routes:
            self.add_cuts(model, eta, kappa, farmer_prices, self.all_routes[route])
        
        model.setObjective(profit, gp.GRB.MAXIMIZE)
        model.update()

        time_optimization_start = time.time()
        model.optimize()
        self.time_usage["solving"] += time.time() - time_optimization_start

        def verify_feasibility():
            current_obj = model.ObjVal

            #print("Time usage", self.time_usage)

            start_callback_time = time.time()
            eta_val = model.getAttr("X", eta)
            kappa_val = model.getAttr("X", kappa)
            farmer_prices_val = model.getAttr("X", farmer_prices)
            
            # We construct the prizes
            iteration_routes = {}
            all_violations = []
            total_cuts = 0
            for ind in range(len(self.instance.intermediaries)):
                intermediary = self.instance.intermediaries[ind]
                for hist_set_index, hist_set in enumerate(intermediary.additional_info["hist_sets"]):
                    prizes = {}
                    for farmer in self.instance.farmers:
                        prizes[farmer.id] = farmer.quantity * self.instance.fruit_price - farmer_prices_val[farmer.id] - eta_val[intermediary.id] * (farmer.quantity if (farmer.id not in hist_set) else 0)
                    start_tsp_time = time.time()
                    row_routes, row_objs = self.tsp_solver.solve(prizes)

                    # TO REMOVE
                    if len(row_routes) == 0:
                        print(
                            f"WARNING: TSP returned no feasible routes for {intermediary.id}, "
                            f"hist_set_index={hist_set_index}"
                        )
                    
                    self.time_usage["tsp"] += time.time() - start_tsp_time
                    for row_route, row_obj in zip(row_routes, row_objs):
                        
                        if len(row_route.farmers) > 0:
                            violation = row_obj - kappa_val[intermediary.id, hist_set_index] - self.het_costs[intermediary.id]
                            # Make sure that the route is not empty
                            if (violation > Optimizer.TOLERANCE):
                                all_violations.append(violation)
                                if frozenset([f.id for f in row_route.farmers]) not in iteration_routes:
                                    start_time = time.time()
                                    add_info = {
                                        "eta": eta_val,
                                        "kappa": kappa_val,
                                        "farmer_prices": farmer_prices_val,
                                    }
                                    self.add_cuts(model, eta, kappa, farmer_prices, row_route, add_info)
                                    total_cuts += 1
                                    self.time_usage["adding_contraints_found"] += time.time() - start_time
                                    iteration_routes[frozenset([f.id for f in row_route.farmers])] = row_route
            
            max_violation = max(all_violations) if all_violations else 0
            # gap = max_violation * gap_ratio
            # ub = current_obj
            # lb = current_obj - gap
            # if (ub * lb < 0):
            #     print(f"Upper bound: {current_obj}, Lower bound: {current_obj - gap}")
            # else:
            #     print(f"Upper bound: {current_obj}, Lower bound: {current_obj - gap}, Gap factor: {(ub - lb) / np.abs(lb)}")
            # print(f"Upper bound: {current_obj}, Total cuts: {total_cuts}")
                                        
            start_time = time.time()
            self.all_routes.update(iteration_routes)
                            
            self.time_usage["callback"] += time.time() - start_callback_time
            

            if max_violation < Optimizer.TOLERANCE:
                return 0
            else:
                return total_cuts

        all_cuts_iter = 0
            
        while True:
            new_cuts = verify_feasibility()
            time_optimization_start = time.time()
            model.optimize()
            self.time_usage["solving"] += time.time() - time_optimization_start
            if new_cuts == 0:
                break
            else:
                all_cuts_iter += new_cuts

        initial_int_profits = {intermediary.id: intermediary_profits[intermediary.id].X for intermediary in self.instance.intermediaries}
        matchings_used_val = model.getAttr("X", matching_used)
        matching_probabilities = {int_id: 0 for int_id in self.intermediary_ids}
        for k in range(len(valid_matchings)):
            for int_id in valid_matchings[k]:
                matching_probabilities[int_id] += matchings_used_val[k]

        profit_val = model.ObjVal
        solution_summary = {
            "profit": model.ObjVal,
            "n_cuts": all_cuts_iter,
            "matching_probs": matching_probabilities,
            "int_profits": initial_int_profits,
            "matching_cost": matching_cost.getValue(),
        }

        model.addConstr(profit >= model.ObjVal - Optimizer.TOLERANCE, "optimality")
        model.setObjective(gp.quicksum(intermediary_profits[intermediary_id] for intermediary_id in self.intermediary_ids), gp.GRB.MAXIMIZE)
        model.optimize()

        solution_max_int_welfare = Solution(self.instance, matching_probabilities, {farmer.id: farmer_prices[farmer.id].X for farmer in self.instance.farmers}, {intermediary.id: intermediary_profits[intermediary.id].X for intermediary in self.instance.intermediaries}, profit_val, solution_summary["matching_cost"])
        if self.options and self.options["structured_farmer_prices"]:
            solution_max_int_welfare.price_per_quantity = price_per_quantity.X
            solution_max_int_welfare.price_per_mile_paved = price_per_mile_paved.X
            solution_max_int_welfare.price_per_mile_dirt = price_per_mile_dirt.X

        solution_summary.update({
            "max_intermediary_profit": model.ObjVal,
            "min_farmer_welfare": sum(farmer_prices[farmer.id].X for farmer in self.instance.farmers),
            "solution_max_int_welfare": solution_max_int_welfare,
        })


        model.setObjective(gp.quicksum(farmer_prices[farmer.id] for farmer in self.instance.farmers), gp.GRB.MAXIMIZE)
        model.optimize()

        solution_min_int_welfare = Solution(self.instance, matching_probabilities, {farmer.id: farmer_prices[farmer.id].X for farmer in self.instance.farmers}, {intermediary.id: intermediary_profits[intermediary.id].X for intermediary in self.instance.intermediaries}, profit_val, solution_summary["matching_cost"])
        if self.options and self.options["structured_farmer_prices"]:
            solution_min_int_welfare.price_per_quantity = price_per_quantity.X
            solution_min_int_welfare.price_per_mile_paved = price_per_mile_paved.X
            solution_min_int_welfare.price_per_mile_dirt = price_per_mile_dirt.X


        solution_summary.update({
            "max_farmer_welfare": model.ObjVal,
            "min_intermediary_profit": sum(intermediary_profits[intermediary.id].X for intermediary in self.instance.intermediaries),
            "updated_int_profits": {intermediary.id: intermediary_profits[intermediary.id].X for intermediary in self.instance.intermediaries},
            "solution_min_int_welfare": solution_min_int_welfare,
        }) 

        print(f"\tPrimal yielded {all_cuts_iter} cuts, new objective: {model.ObjVal}")

        # if solution_type in ["lower_bound_knapsack"]:
        #     s = model.addVars([(intermediary.id, hist_set_index) for intermediary in self.instance.intermediaries for hist_set_index in range(len(intermediary.additional_info["hist_sets"]))], vtype=gp.GRB.CONTINUOUS, lb=0, name="s")
        #     lamb = model.addVars([(intermediary.id, hist_set_index) for intermediary in self.instance.intermediaries for hist_set_index in range(len(intermediary.additional_info["hist_sets"]))], vtype=gp.GRB.CONTINUOUS, lb=0, name="lamb")
        #     gamma = model.addVars([(intermediary.id, hist_set_index, f.id) for intermediary in self.instance.intermediaries for hist_set_index in range(len(intermediary.additional_info["hist_sets"])) for f in self.instance.farmers], vtype=gp.GRB.CONTINUOUS, lb=0, name="gamma")
        #     x = model.addVars([(intermediary.id, hist_set_index, f.id, self.instance.edge_to_index[e_id]) for intermediary in self.instance.intermediaries for hist_set_index in range(len(intermediary.additional_info["hist_sets"])) for f in self.instance.farmers for e_id in self.instance.root_edges[f.id]], vtype=gp.GRB.CONTINUOUS, lb=0, name="x")

        #     for intermediary in self.instance.intermediaries:
        #         for hist_set_index, hist_set in enumerate(intermediary.additional_info["hist_sets"]):
        #             model.addConstr(kappa[intermediary.id, hist_set_index] >= -s[intermediary.id, hist_set_index] + lamb[intermediary.id, hist_set_index] * self.instance.TRUCK_CAPACITY_CONSTRAINT + gp.quicksum(gamma[intermediary.id, hist_set_index, f.id] for f in self.instance.farmers) - self.instance.truck_fixed_cost, f"relaxation_{intermediary.id}_{hist_set_index}")
        #             for f in self.instance.farmers:
        #                 w = self.instance.fruit_price * f.quantity - farmer_prices[f.id] - eta[intermediary.id] * (f.quantity if f.id not in hist_set else 0)
        #                 z = gp.quicksum(2 * x[intermediary.id, hist_set_index, f.id, self.instance.edge_to_index[e_id]] * self.instance.tree[e_id[0]][e_id[1]]["weight"] * self.instance.cost_per_meter for e_id in self.instance.root_edges[f.id])
        #                 model.addConstr(w - z - lamb[intermediary.id, hist_set_index] * f.quantity - gamma[intermediary.id, hist_set_index, f.id] + s[intermediary.id, hist_set_index] <= 0, f"relaxation_constr_{intermediary.id}_{hist_set_index}_{f.id}")
        #             for e_id, e_index in self.instance.edge_to_index.items():
        #                 model.addConstr(gp.quicksum(x[intermediary.id, hist_set_index, f.id, e_index] for f in self.instance.edge_to_root_farmers[e_id]) <= 1, f"submodularity_{intermediary.id}_{hist_set_index}_{e_id}")

        return solution_summary





