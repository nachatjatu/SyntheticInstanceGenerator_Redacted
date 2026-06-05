"""Linear programming based solvers for routing problems.

This module provides wrappers around Gurobi models to solve the travelling
salesperson subproblem (TSP) and the vehicle routing problem (VRP) on the
platform instance tree.  There is also a placeholder for an OR-Tools based
implementation.
"""

import os
import gurobipy as gp
from farmers_intermediaries import Instance, Matching
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def get_solver_threads() -> int:
    for var in ("SLURM_CPUS_PER_TASK", "GUROBI_THREADS"):
        value = os.environ.get(var)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return 1


class TSPSolver:
    """Gurobi-based solver for the prize-collecting TSP defined on the tree.

    The solver takes a prize for each farmer and maximizes prize minus the
    twice-traversed edge costs and fixed truck cost.  The returned route is
    constructed using :meth:`Instance.calculate_tree_path`.
    """
    TOLERANCE = 1.0
    def __init__(self, instance: Instance):
        """Initialize with a platform :class:`Instance`.
        """
        self.instance = instance

    def solve(self, prizes):
        """Solve the prize-collecting TSP.

        Parameters
        ----------
        prizes : dict or sequence
            Prize value associated with each farmer (indexed by farmer id).

        Returns
        -------
        route : Route
            Best route found by the model.
        objective : float
            Objective value corresponding to the route.
        """
        if len(prizes) != len(self.instance.farmers):
            raise ValueError(f"Prizes length mismatch: {len(prizes)} != {len(self.instance.farmers)}")
        model = gp.Model("TSP")
        model.setParam("Threads", get_solver_threads())
        model.setParam("OutputFlag", 0)
        
        # Add binary variables for each farmer
        farmer_ids = [farmer.id for farmer in self.instance.farmers]
        farmer_vars = model.addVars(farmer_ids, vtype=gp.GRB.BINARY, name="visit")

        # Add a continuous variable for the used truck
        used = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name="used")

        # Add continuous variables for each edge in the tree
        edge_vars = model.addVars(self.instance.tree_edges, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="edge")

        # Make sure that all edges of a node are transversed
        for farmer in self.instance.farmers:
            for edge in self.instance.root_edges[farmer.id]:
                model.addConstr(edge_vars[edge] >= farmer_vars[farmer.id])

        # Make sure that if any farmer is picked up, then used is equal to one
        model.addConstrs((used >= farmer_vars[farmer.id] for farmer in self.instance.farmers), "used")

        # Make sure that at least one farmer is picked up
        model.addConstr(gp.quicksum(farmer_vars[farmer.id] for farmer in self.instance.farmers) >= 1, "one_farmer")

        # Add capacity constraint
        model.addConstr(gp.quicksum(farmer_vars[farmer.id] * farmer.quantity for farmer in self.instance.farmers) <= self.instance.truck_capacity, "capacity")

        # Objective: maximize the total prize collected minus the cost of each edge twice
        model.setObjective(
            gp.quicksum(prizes[farmer.id] * farmer_vars[farmer.id] for farmer in self.instance.farmers)
            - 2 * gp.quicksum(edge_vars[edge] * self.instance.tree[edge[0]][edge[1]]["weight"] * self.instance.cost_per_meter for edge in self.instance.tree_edges)
            - used * self.instance.truck_fixed_cost,
            gp.GRB.MAXIMIZE)

        model.optimize()
        objective = model.ObjVal

        # Extract the matching
        selected_farmers = [farmer.id for farmer in self.instance.farmers if farmer_vars[farmer.id].X > 0.5]
        route = self.instance.calculate_tree_path(selected_farmers)

        # Verify that the objectives correspond
        total_prize = sum(prizes[farmer_id] for farmer_id in selected_farmers)
        alt_objective = total_prize - route.cost
        # Check if the objective matches
        if abs(objective - alt_objective) > self.TOLERANCE:
            raise ValueError(f"Objective mismatch: {objective} != {alt_objective}")

        return route, objective


class VRPSolver:
    """Gurobi-based solver for the vehicle routing problem (VRP).

    This formulation selects a set of farmers for each truck (intermediary)
    respecting capacity and attempts to minimize total routing and fixed costs.
    It also handles lower- and upper-bounds on the number of vehicles.
    """

    TOLERANCE = 1.0
    def __init__(self, instance: Instance):
        """Create a VRP solver for the given platform instance."""
        self.instance = instance

    def solve(self, num_vehicles_lower_bound: int, num_vehicles_upper_bound: int):
        """Solve the VRP with a bound on number of vehicles.

        Parameters
        ----------
        num_vehicles_lower_bound : int
            Minimum number of trucks that must be used.
        num_vehicles_upper_bound : int
            Maximum number of trucks available in the fleet.

        Returns
        -------
        matching : Matching
            A :class:`Matching` object describing the chosen routes.
        """
        model = gp.Model("VRP")
        model.setParam("Threads", get_solver_threads())
        model.setParam("TimeLimit", 2000) 

        # Add binary variables for each farmer and intermediary
        farmer_ids = [farmer.id for farmer in self.instance.farmers]
        intermediary_ids = list(range(num_vehicles_upper_bound))

        # Add binary variables for each truck and each farmer
        matching_vars = model.addVars(intermediary_ids, farmer_ids, vtype=gp.GRB.BINARY, name="visit")
        
        # Add continuous variables for each edge in the tree and each intermediary
        edge_vars = model.addVars(intermediary_ids, list(range(len(self.instance.tree_edges))), vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name="edge")
        edge_to_index = {edge: index for index, edge in enumerate(self.instance.tree_edges)}

        # Add continuous variables for each used intermediary
        used = model.addVars(intermediary_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name="used")

        # Make sure that all edges of a node are transversed
        for intermediary in intermediary_ids:
            for farmer in self.instance.farmers:
                for edge in self.instance.root_edges[farmer.id]:
                    model.addConstr(edge_vars[intermediary, edge_to_index[edge]] >= matching_vars[intermediary, farmer.id])

        # Make sure that if a truck picks up a farmer, then it is used
        for intermediary in intermediary_ids:
            model.addConstrs((used[intermediary] >= matching_vars[intermediary, farmer.id] for farmer in self.instance.farmers), f"used_lower_{intermediary}")
        # Make sure that if a truck is used, then it picks up at least one farmer
        for intermediary in intermediary_ids:
            model.addConstr(gp.quicksum(matching_vars[intermediary, farmer.id] for farmer in self.instance.farmers) >= used[intermediary], f"used_upper_{intermediary}")
        # Make sure that at least num_vehicles_lower_bound trucks are used
        model.addConstr(gp.quicksum(used[intermediary] for intermediary in intermediary_ids) >= num_vehicles_lower_bound, "num_vehicles")

        # Make sure that each farmer is picked up by exactly one truck
        model.addConstrs((gp.quicksum(matching_vars[intermediary, farmer.id] for intermediary in intermediary_ids) == 1 for farmer in self.instance.farmers), "one_truck")

        # Add capacity constraint for each truck
        for intermediary in intermediary_ids:
            model.addConstr(gp.quicksum(matching_vars[intermediary, farmer.id] * farmer.quantity for farmer in self.instance.farmers) <= self.instance.truck_capacity, "capacity")

        # Add an ordering of used trucks to break symmetry
        for intermediary in intermediary_ids[1:]:
            model.addConstr(used[intermediary] <= used[intermediary - 1], f"order_{intermediary}")

        # Objective: minimize the total cost
        model.setObjective(
            gp.quicksum(2 * edge_vars[intermediary,  edge_index] * self.instance.tree[edge[0]][edge[1]]["weight"] * self.instance.cost_per_meter for intermediary in intermediary_ids for edge_index, edge in enumerate(self.instance.tree_edges)) + \
            gp.quicksum(used[intermediary] * self.instance.truck_fixed_cost for intermediary in intermediary_ids),
            gp.GRB.MINIMIZE)
        model.optimize()

        total_cost = model.ObjVal

        # Extract the matching
        alt_cost = 0
        routes = []
        for intermediary in intermediary_ids:
            selected_farmers = [farmer.id for farmer in self.instance.farmers if matching_vars[intermediary, farmer.id].X > 0.5]
            if len(selected_farmers) > 0:
                route = self.instance.calculate_tree_path(selected_farmers)
                routes.append(route)

        matching = Matching(self.instance, routes)
        alt_cost = matching.cost

        # Check if the objective match
        print("Gap", total_cost, alt_cost)
        if abs(total_cost - alt_cost) > self.TOLERANCE:
            raise ValueError(f"Objective mismatch: {total_cost} != {alt_cost}")
        
        return matching
    
    
class ORToolsVRPSolver:
    """Placeholder for an OR-Tools based VRP solver.

    The implementation was intended to provide an alternative to Gurobi but
    remains unimplemented.  Users should continue using
    :class:`VRPSolver` unless this class is extended in the future.
    """

    def __init__(self, instance: Instance):
        """Store the platform instance."""
        self.instance = instance

    def solve(self):
        """Solve the VRP using OR-Tools.

        Raises
        ------
        NotImplementedError
            Always, because this method is a stub.
        """
        raise NotImplementedError("ORToolsVRPSolver.solve is not implemented")