"""Utilities for building and manipulating platform instances.

This module defines the basic objects used to describe a platform of farmers,
intermediaries, and a mill.  It also contains functionality for loading
instances from YAML files, mapping nodes to a road network graph, and
computing simple routes and matchings.  The core classes are :class:`Instance`,
:class:`Route` and :class:`Matching`, with helper node types for farmers,
intermediaries, and the mill.

Example
-------
>>> from farmers_intermediaries import Instance
>>> inst = Instance.from_yaml('data/instances/2020-08-27.yaml')
"""
from __future__ import annotations
import networkx as nx
from road_graphs import RoadGraph
import yaml
import pickle
import os



class Node:
    """Base node type representing an entity with an identifier and location.

    Subclasses specialize this for farmers, intermediaries and mills but the
    basic attributes ``id`` and ``location`` are shared.  ``location`` is a
    (latitude, longitude) tuple.
    """

    def __init__(self, id: str, location: tuple):
        self.id = id
        self.location = location

    def __repr__(self):
        return f"Node(id={self.id}, location={self.location})"
    

class Farmer(Node):
    """A farmer supplying fruit with a quantity and optional metadata.

    Parameters
    ----------
    id : str
        Unique identifier for the farmer.
    quantity : float
        Amount of fruit available (in tonnes).
    location : tuple
        Geographic coordinate of the farmer (lat, lon).
    additional_info : dict, optional
        Arbitrary extra information, such as status quo intermediary.
    """

    def __init__(self, id: str, quantity: float, location: tuple, additional_info: dict = None):
        super().__init__(id, location)
        self.quantity = quantity
        self.additional_info = additional_info

    def __repr__(self):
        return f"Farmer(id={self.id}, quantity={self.quantity}, location={self.location})"
    
    
class Intermediary(Node):
    """An intermediary that collects fruit from farmers and delivers to the mill.

    Parameters
    ----------
    id : str
        Unique identifier for the intermediary.
    capacity : float
        Maximum load (in the same units as farmers' quantities).
    location : tuple
        Geographic coordinate of the intermediary.
    additional_info : dict, optional
        Supporting data such as history sets used in the analysis.
    """

    def __init__(self, id: str, capacity: float, location: tuple, additional_info: dict = None):
        super().__init__(id, location)
        self.capacity = capacity
        self.additional_info = additional_info

    def __repr__(self):
        return f"Intermediary(id={self.id}, capacity={self.capacity}, location={self.location})"
    
    
class Mill(Node):
    """Simple node to represent the processing mill."""

    def __init__(self, id: str, location: tuple):
        super().__init__(id, location)

    def __repr__(self):
        return f"Mill(id={self.id}, location={self.location})"
    

class Instance:
    """Container for a platform instance consisting of farmers,
    intermediaries and a mill.

    Provides methods for loading from YAML, mapping to road graphs, and
    generating routes based on a precomputed tree.  Several economic and
    cost-related constants are defined as class attributes and can be
    overridden per-instance after construction.
    """

    MILL_KEY = "SKIP" # Default value for mill key
    TRUCK_CAPACITY_CONSTRAINT = 9.0 # Default value for capacity constraint
    TRUCK_FIXED_COST = 800000 #295591.5532073739
    COST_PER_METER = 2625/1000.0  #(2625.0+2065.0)/2.0/1000.0
    FRUIT_PRICE = 2513.0 * 1000.0
    N_HIST_SETS = 5
    USD = 14500.0

    def __init__(self, instance_id: str, farmers: list, intermediaries: list, mill: Mill) -> None:
        self.instance_id = instance_id
        self.farmers = farmers
        self.farmer_by_id = {farmer.id: farmer for farmer in farmers}
        self.intermediaries = intermediaries
        self.mill = mill
        self.nodes = farmers + intermediaries + [mill]
        self.truck_capacity = self.TRUCK_CAPACITY_CONSTRAINT
        self.truck_fixed_cost = self.TRUCK_FIXED_COST
        self.cost_per_meter = self.COST_PER_METER
        self.fruit_price = self.FRUIT_PRICE
        self.source = None

    def calculate_tree_path(self, farmer_ids: list) -> Route:
        """Calculate the shortest path in the tree graph for a list of nodes, starting at the mill"""
        tour = []
        for graph_node in self.tree_order:
            if graph_node in self.graph_node_to_id:
                ids_in_node = self.graph_node_to_id[graph_node]
                for n in ids_in_node:
                    if n in farmer_ids:
                        tour.append(self.farmer_by_id[n])
        if len(tour) != len(farmer_ids):
            raise ValueError("The length of the tour does not match the length of farmer_ids")
        return Route(tour, self)

    def to_dict(self) -> dict:
        """Serialize the instance to a JSON-serializable dictionary.

        The returned structure includes basic meta-data, farmer quantities,
        and status-quo intermediary assignments useful for logging and
        exporting results.
        """
        return {
            "instance_id": self.instance_id,
            "farmer_quantities": {farmer.id: farmer.quantity for farmer in self.farmers},
            "truck_fixed_cost": self.truck_fixed_cost,
            "fruit_price": self.fruit_price,
            "usd": self.USD,
            "status_quo_quantities": self._calculate_sq_quantities(),
            "n_intermediaries": len(self.intermediaries),
            "n_farmers": len(self.farmers),
            "status_quo_intermediary": {farmer.id: farmer.additional_info["intermediary_id"] for farmer in self.farmers},
        }

    @classmethod
    def from_dict(cls, data: dict, opt_quantities: dict = None):
        farmers = []

        for f in data['farmers']:
            if opt_quantities:
                farmers.append(Farmer(f['id'], opt_quantities[f['id']], tuple(f['location']), {"intermediary_id":f['intermediary']}))
            else:
                farmers.append(Farmer(f['id'], f['quantity'], tuple(f['location']), {"intermediary_id":f['intermediary']}))

        intermediaries = []
        for int_id in data['intermediaries']:
            hist_sets = [frozenset(r) for r in int_id['routes']]
            if len(hist_sets) == 0:
                hist_sets.append(frozenset())
            intermediaries.append(Intermediary(
                int_id['id'], int_id['capacity'], tuple(int_id['location']), {"hist_sets":hist_sets}
            ))

        for m in data["mills"]:
            if m["id"] == cls.MILL_KEY:
                mill = Mill(m['id'], tuple(m['location']))
                break

        instance = cls(data['instance_id'], farmers, intermediaries, mill)
        return instance

        

    @classmethod
    def from_yaml(cls, yaml_file: str, opt_quantities: dict = None):
        """Load platform instance data from a YAML file."""
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        instance = cls.from_dict(data, opt_quantities)
        # Remove the .yaml extension from the file name
        instance.source = yaml_file[:-5]
        return instance

    def set_graph(self, graph: RoadGraph):
        """Attach a road network and precompute auxiliary data.

        The graph is used to compute distances between nodes.  This method
        updates internal mappings and builds a spanning tree that serves as
        the base for route cost computations.
        """
        self.graph = graph
        self._precompute_mappings()
        self.tree, list_root_edges = self.graph.build_tree(list(self.id_to_graph_node.keys()), list(self.id_to_graph_node.values()), self.MILL_KEY, plot=False)
        self.root_edges = dict(zip(self.id_to_graph_node.keys(), list_root_edges))
        self.tree_edges = []
        for f_id in self.root_edges:
            for edge in self.root_edges[f_id]:
                if edge not in self.tree_edges:
                    self.tree_edges.append(edge)
        self.tree_order = list(nx.dfs_preorder_nodes(self.tree, source=self.MILL_KEY))
        self.node_to_parent = nx.dfs_predecessors(self.tree, source=self.MILL_KEY)
        self.edge_to_index = {edge: i for i, edge in enumerate(self.tree_edges)}
        self.edge_to_root_farmers = {edge: [] for edge in self.tree_edges}
        for farmer in self.farmers:
            for edge in self.root_edges[farmer.id]:
                self.edge_to_root_farmers[edge].append(farmer)

        # Calculate distances to root for each intermediary
        self.dist_to_mill = {}
        for intermediary in self.intermediaries:
            graph_node = self.id_to_graph_node[intermediary.id]
            self.dist_to_mill[intermediary.id] = nx.shortest_path_length(self.tree, source=graph_node, target=self.id_to_graph_node[self.mill.id], weight='weight') * self.cost_per_meter
            dirt_to_mill = nx.shortest_path_length(self.tree, source=graph_node, target=self.id_to_graph_node[self.mill.id], weight='weight_dirt') * self.cost_per_meter
            intermediary.dist_to_mill = self.dist_to_mill[intermediary.id]
            intermediary.dirt_to_mill = dirt_to_mill
        for farmer in self.farmers:
            graph_node = self.id_to_graph_node[farmer.id]
            self.dist_to_mill[farmer.id] = nx.shortest_path_length(self.tree, source=graph_node, target=self.id_to_graph_node[self.mill.id], weight='weight') * self.cost_per_meter
            farmer.dist_to_mill = self.dist_to_mill[farmer.id]
            dirt_to_mill = nx.shortest_path_length(self.tree, source=graph_node, target=self.id_to_graph_node[self.mill.id], weight='weight_dirt') * self.cost_per_meter
            farmer.dirt_to_mill = dirt_to_mill
            paved_to_mill = nx.shortest_path_length(self.tree, source=graph_node, target=self.id_to_graph_node[self.mill.id], weight='weight_paved') * self.cost_per_meter
            farmer.paved_to_mill = paved_to_mill

    def save_graph_data(self, filename: str):
        """Persist precomputed graph-related data to a pickle file.

        This speeds up repeated experiments by avoiding costly graph
        computations on subsequent loads.
        """
        with open(filename, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'tree': self.tree,
                'root_edges': self.root_edges,
                'tree_edges': self.tree_edges,
                'tree_order': self.tree_order,
                'id_to_graph_node': self.id_to_graph_node,
                'graph_node_to_id': self.graph_node_to_id,
                'node_to_parent': self.node_to_parent,
                'edge_to_index': self.edge_to_index,
                'edge_to_root_farmers': self.edge_to_root_farmers,
                'dist_to_mill': self.dist_to_mill,
            }, f)

    def load_graph_data(self, filename: str):
        """Load previously saved graph data and restore internal state.

        Returns ``True`` if the file existed and the load succeeded, ``False``
        otherwise.
        """
        print(filename)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.graph = data['graph']
                self.tree = data['tree']
                self.root_edges = data['root_edges']
                self.tree_edges = data['tree_edges']
                self.tree_order = data['tree_order']
                self.id_to_graph_node = data['id_to_graph_node']
                self.graph_node_to_id = data['graph_node_to_id']
                self.node_to_parent = data['node_to_parent']
                self.edge_to_index = data['edge_to_index']
                self.edge_to_root_farmers = data['edge_to_root_farmers']
                self.dist_to_mill = data['dist_to_mill']
                return True
        return False

    def __repr__(self):
        """Human-readable representation of the instance."""
        return f"PlatformInstance(id={self.instance_id})"

    def _precompute_mappings(self):
        """Internal helper to map instance nodes onto graph nodes.

        The mappings ``id_to_graph_node`` and ``graph_node_to_id`` are
        populated so that later distance queries can be made efficiently.
        """
        locations = [node.location for node in self.nodes]
        node_ids = [node.id for node in self.nodes]
        self.id_to_graph_node = dict(zip(node_ids, self.graph.closest_points(locations)))
        self.graph_node_to_id = {}
        for node_id, graph_node in self.id_to_graph_node.items():
            if graph_node not in self.graph_node_to_id:
                self.graph_node_to_id[graph_node] = [node_id]
            else:
                self.graph_node_to_id[graph_node].append(node_id)
    
    def _calculate_sq_quantities(self) -> dict[str, float]:
        sq_map = {}

        for inter in self.intermediaries:
            hist_sets = inter.additional_info.get("hist_sets", [])
            if not hist_sets:
                sq_map[inter.id] = 0.0
                continue
                
            # Calculate total quantity across all historical sets
            print(hist_sets)
            total_hist_qty = sum(
                self.farmer_by_id[f_id].quantity 
                for h_set in hist_sets 
                for f_id in h_set
            )
            # Average it out by the number of sets
            sq_map[inter.id] = total_hist_qty / len(hist_sets)

        return sq_map

class Route:
    """Represents a collection of farmers visited by a single truck.

    The route object computes the total quantity, verifies capacity, and
    determines the associated cost and value relative to the instance
    parameters.
    """

    def __init__(self, farmers: list, instance: Instance) -> None:
        self.farmers = farmers
        self.total_quantity = sum(farmer.quantity for farmer in farmers)
        self.instance = instance
        
        self.cost = self.calculate_route_tree_cost()
        self.is_feasible = (self.total_quantity <= (self.instance.truck_capacity + 1e-6))
        self.value = self.total_quantity * self.instance.fruit_price - self.cost

    def __repr__(self) -> str:
        return f"Route(farmers={[farmer.id for farmer in self.farmers]})"


    def calculate_route_tree_cost(self) -> float:
        """Calculate the cost of traversing the route using the precomputed tree.

        The cost consists of a fixed truck cost plus the weighted distance along
        the tree between successive points.
        """
        if len(self.farmers) == 0:
            return 0
        else:
            cost = self.instance.truck_fixed_cost
            graph_nodes = [self.instance.id_to_graph_node[self.instance.mill.id]] + \
                    [self.instance.id_to_graph_node[f.id] for f in self.farmers] + \
                    [self.instance.id_to_graph_node[self.instance.mill.id]]
            
            for i in range(len(graph_nodes) - 1):
                cost += nx.shortest_path_length(self.instance.tree, source=graph_nodes[i], target=graph_nodes[i + 1], weight='weight') * self.instance.cost_per_meter
        return cost

class Matching:
    """Collection of routes covering all farmers exactly once.

    Validates that each farmer appears in one and only one route and computes
    the total cost of the matching.
    """

    def __init__(self, instance: Instance, routes: list) -> None:
        self.instance = instance
        self.routes = routes
        self._verify_routes()

    def _verify_routes(self) -> None:
        """Verify that each farmer is covered exactly once and compute cost.

        Raises
        ------
        ValueError
            If any farmer is missing or appears in multiple routes or a route is
            empty.
        """ 

        # 1. Collect all farmer IDs from the routes
        all_farmers_in_routes = [f.id for r in self.routes for f in r.farmers]
        
        # 2. Check for empty routes and coverage simultaneously
        if any(len(r.farmers) == 0 for r in self.routes):
            raise ValueError("Route cannot be empty.")

        # 3. Use Set math for validation
        instance_ids = {f.id for f in self.instance.farmers}
        route_ids_set = set(all_farmers_in_routes)

        if len(all_farmers_in_routes) != len(route_ids_set):
            raise ValueError("Duplicate farmers found across routes.")
        
        if route_ids_set != instance_ids:
            missing = instance_ids - route_ids_set
            extra = route_ids_set - instance_ids
            raise ValueError(f"Coverage mismatch. Missing: {missing}, Extra: {extra}")

        # 4. Sum costs
        self.cost = sum(r.cost for r in self.routes)
