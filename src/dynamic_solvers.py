"""Dynamic programming and LP based solvers accelerating routing subproblems.

Contains a tailored prize matrix approach to quickly evaluate candidate routes
on the tree and wrappers combining this with Gurobi models.  The classes here
are used by the pricing algorithms to generate and value routes efficiently.
"""

import gurobipy as gp
from farmers_intermediaries import Instance
import numpy as np
import math

class PrizeMatrix:
    """Helper structure for rapid computation of best subsets of farmers.

    The matrix stores dynamic programming information on a tree derived from
    the platform graph.  It is primarily used by :class:`TSPSolver` to find
    high‑value farmer sets without solving a full integer programme.
    """
    TOP_N = 5
    def __init__(self, instance: Instance, q_max: int, n_nodes: int, 
                 farmer_to_node, ordering, quantities, 
                 parents, root_node, costs):
        """Create a prize matrix for the given tree.

        Parameters mirror those computed in :class:`TSPSolver.__init__`.
        """
        self.instance = instance
        self.matrix = np.zeros((n_nodes, q_max + 1))
        self.farmers = np.zeros((n_nodes, q_max + 1, len(farmer_to_node)), dtype=bool)
        self.farmer_to_node = farmer_to_node
        self.ordering = ordering
        self.quantities = quantities
        self.parents = parents
        self.root_node = root_node
        self.costs = costs

    def reset(self, prizes: list) -> None:
        """Reset internal arrays and seed them using a prize vector.

        Parameters
        ----------
        prizes : list
            Prize associated with each farmer index.
        """
        self.matrix.fill(-np.inf)
        self.matrix[:, 0] = 0
        self.farmers.fill(False)
        for i in range(len(prizes)):
            self.matrix[self.farmer_to_node[i], self.quantities[i]] = prizes[i]
            self.farmers[self.farmer_to_node[i], self.quantities[i], i] = True

    def merge(self, i: int, j: int, k: int, cost: float) -> None:
        """Combine two nodes during the recursive DP computation.

        Parameters
        ----------
        i : int
            index of child node in matrix
        j : int
            index of parent node in matrix
        k : int
            index of merged node where result is stored
        cost : float
            Edge cost between i and j
        """
        matrix_aux = np.zeros(self.matrix.shape[1]) - np.inf
        matrix_aux[0] = 0
        farmers_aux = np.zeros((self.matrix.shape[1], self.farmers.shape[2]), dtype=bool)

        for q1 in range(self.matrix.shape[1]):
            effective_cost = cost if q1 > 0 else 0
            if (q1 == 0) or (q1 > 0 and self.matrix[i, q1] > -np.inf):
                for q2 in range(self.matrix.shape[1] - q1):
                    if (q2 == 0) or (q2 > 0 and self.matrix[j, q2] > -np.inf):
                        agg_q = q1 + q2
                        agg_p = self.matrix[i, q1] + self.matrix[j, q2] - effective_cost
                        if matrix_aux[agg_q] < agg_p:
                            matrix_aux[agg_q] = agg_p
                            farmers_aux[agg_q] = self.farmers[i, q1] | self.farmers[j, q2]
        self.matrix[k] = matrix_aux
        self.farmers[k] = farmers_aux

    def solve(self, prizes: list, threshold: float) -> tuple[list, list]:
        """Evaluate the prize matrix and return top routes exceeding threshold.

        Parameters
        ----------
        prizes : dict
            Prize associated with each farmer.
        threshold : float
            Minimum objective value to consider.

        Returns
        -------
        objs : list
            Sorted objectives (descending) above the threshold.
        farmer_sets : list
            Corresponding sets of farmer indices.
        """
        self.reset(prizes)
        if len(prizes) != len(self.instance.farmers):
            raise ValueError(f"Prizes length mismatch: {len(prizes)} != {len(self.instance.farmers)}")

        # Perform DFS of the tree
        for node in self.ordering:
            if node != self.root_node:
                parent_node = self.parents[node]
                self.merge(node, parent_node, parent_node, 2 * self.costs[node, parent_node])
                
        # Extract all prizes
        objs = []
        farmer_sets = []
        for i in range(1, self.matrix.shape[1]):
            if (self.matrix[self.root_node, i] > threshold) and (np.sum(self.farmers[self.root_node, i]) > 0):
                objs.append(self.matrix[self.root_node, i])
                farmer_sets.append(np.where(self.farmers[self.root_node, i])[0])

        # Sort the prizes in descending order
        if len(objs) >= 1:
            objs, farmer_sets = zip(*sorted(zip(objs, farmer_sets), key=lambda x: x[0], reverse=True))
            return objs[0: min(self.TOP_N, len(objs))], farmer_sets[0: min(self.TOP_N, len(objs))]
        else:
            return [], []


class TSPSolver:
    """Fast prize-collecting TSP formulation using dynamic programming.

    Wraps a :class:`PrizeMatrix` together with a Gurobi model to generate
    candidate routes and verify optimality.
    """
    TOLERANCE = 1.0
    MULTIPLIER = 10
    def __init__(self, instance: Instance) -> None:
        """Construct solver and precompute matrices from the instance tree."""
        self.instance = instance
        self.prize_matrix = self.init_prize_matrix()

    def init_prize_matrix(self):
        q_max = int(self.instance.truck_capacity * self.MULTIPLIER)
        n_nodes = len(self.instance.tree.nodes())
        nodes_indices = {node: i for i, node in enumerate(self.instance.tree.nodes())}
        index_to_node = {i: node for i, node in enumerate(self.instance.tree.nodes())}
        farmer_index_to_node_index = {i: nodes_indices[farmer.id] for i, farmer in enumerate(self.instance.farmers)}
        ordering = [nodes_indices[node] for node in list(reversed(self.instance.tree_order))]
        quantities = [int(farmer.quantity * self.MULTIPLIER) for farmer in self.instance.farmers]
        # quantities = [
        #     math.ceil(farmer.quantity * self.MULTIPLIER - 1e-9)
        #     for farmer in self.instance.farmers
        # ]
        parents = [nodes_indices[self.instance.node_to_parent[node]] if node != self.instance.mill.id else -1 for node in self.instance.tree.nodes()]
        root_node = nodes_indices[self.instance.mill.id]
        costs = np.zeros((n_nodes, n_nodes))
        for i, n1 in enumerate(self.instance.tree.nodes()):
            if n1 != self.instance.mill.id:
                costs[i, parents[i]] = self.instance.tree[n1][index_to_node[parents[i]]]["weight"] * self.instance.cost_per_meter
                costs[parents[i],i] = self.instance.tree[n1][index_to_node[parents[i]]]["weight"] * self.instance.cost_per_meter
        
        return PrizeMatrix(self.instance, q_max, n_nodes, farmer_index_to_node_index, ordering, quantities, parents, root_node, costs)
            
    def solve(self, prizes_dict: dict) -> tuple[list, list]:
        """Solve the simplified TSP problem.

        Parameters
        ----------
        prizes_dict : dict
            Prize for each farmer id.

        Returns
        -------
        final_routes : list of Route
            Best candidate routes found.
        final_objectives : list of float
            Corresponding objective values.
        """
        # Check that # of prizes = # of farmers
        if len(prizes_dict) != len(self.instance.farmers):
            raise ValueError(f"Prizes length mismatch: {len(prizes_dict)} != {len(self.instance.farmers)}")
        
        # Convert prizes into a list and solve
        prizes = [prizes_dict[farmer.id] for farmer in self.instance.farmers]
        total_prizes, farmer_sets = self.prize_matrix.solve(prizes, -np.inf)

        selected_farmers_sets = [[self.instance.farmers[i].id for i in farmer_set] for farmer_set in farmer_sets]

        feasible_candidates = []

        for total_prize, selected_farmers in zip(total_prizes, selected_farmers_sets):
            if len(selected_farmers) == 0:
                raise ValueError("Zero farmers selected")

            route = self.instance.calculate_tree_path(selected_farmers)

            # Skip infeasible DP candidates, but keep searching through the larger pool
            if not route.is_feasible:
                continue

            objective = total_prize - self.instance.truck_fixed_cost

            # Objective consistency check
            alt_objective = (
                sum(prizes_dict[farmer_id] for farmer_id in selected_farmers)
                - route.cost
            )

            if abs(objective - alt_objective) > 1e-4:
                raise ValueError(f"Objective mismatch: {objective} != {alt_objective}")

            feasible_candidates.append((objective, route))

        # Keep top N feasible routes, not top N approximate DP candidates
        feasible_candidates.sort(key=lambda x: x[0], reverse=True)
        feasible_candidates = feasible_candidates[:self.prize_matrix.TOP_N]

        final_routes = [route for objective, route in feasible_candidates]
        final_objectives = [objective for objective, route in feasible_candidates]

        if len(final_routes) == 0:
            raise RuntimeError(
                "Pricing returned no feasible routes after DP candidate generation. "
                "This can underconstrain the primal and produce inflated profit."
            )

        return final_routes, final_objectives

        # Loop over farmers and compute their profit
        final_routes, final_objectives = [], []
        for i in range(len(total_prizes)):
            # Make sure that the prize is larger than the fixed cost
            if len(selected_farmers_sets[i]) == 0:
                raise ValueError(f"Zero farmers selected")
            else:
                objective = total_prizes[i] - self.instance.truck_fixed_cost
            route = self.instance.calculate_tree_path(selected_farmers_sets[i])

            # Only perform the strict objective check if the route is actually feasible
            if route.is_feasible:
                alt_objective = sum(prizes_dict[farmer_id] for farmer_id in selected_farmers_sets[i]) - route.cost
                if abs(objective - alt_objective) > 1e-4: # Use a small tolerance for floats
                    raise ValueError(f"Objective mismatch: {objective} != {alt_objective}")
                
                final_routes.append(route)
                final_objectives.append(objective)            

        return final_routes, final_objectives