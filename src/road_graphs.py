"""Utilities for creating and manipulating road network graphs.

The :class:`RoadGraph` class wraps an OSMnx graph and provides methods for
computing closest nodes, pruning, building approximated Steiner trees, and
plotting subgraphs annotated by surface type.
"""

import networkx as nx
from shapely.geometry import Point
import osmnx as ox
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.lines import Line2D

class RoadGraph:
    DIRT_FACTOR = 4 

    def __init__(self, graph: nx.MultiDiGraph):
        mapping_surfaces = {
            None: "dirt", # assume dirt by default
            'path': "dirt",
            'primary': "paved",
            'primary_link': "paved",
            'residential': "dirt",
            'secondary': "paved",
            'secondary_link': "paved",
            'service': "dirt",
            'tertiary': "paved",
            'track': "dirt",
            'trunk': "paved",
            'trunk_link': "paved",
            'unclassified': "dirt",
            'living_street': "dirt",
        }
        self.mapping_surfaces = mapping_surfaces
        self.graph = graph
        self.undirected_graph = self.get_undirected_graph()
        
    def closest_points(self, all_points: list) -> list:
        "Returns the closest nodes to a list of points in lat lon coordinates"
        # 1. Get a list of Shapely points from all_points
        shapely_points = [Point(p[1], p[0]) for p in all_points]
            
        # 2. Project points to graph crs 
        graph_crs = self.graph.graph['crs']
        gdf = gpd.GeoDataFrame(geometry=shapely_points, crs="EPSG:4326")
        gdf_proj = ox.projection.project_gdf(gdf, to_crs=graph_crs)

        # 3. Extract coordinates
        xs = gdf_proj.geometry.x
        ys = gdf_proj.geometry.y

        # 4. Find nearest nodes and return as list
        nearest_nodes = ox.nearest_nodes(self.graph, xs, ys)
        return nearest_nodes.tolist()
    
    def get_undirected_graph(self) -> nx.Graph:
        "Returns an undirected version of the graph"
        G = self.graph.to_undirected()
        # 1. Connect the graph
        #   a. Get all components as subgraphs
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

        #   b. Find the largest one by node count
        main_island = max(subgraphs, key=len)
        max_i = subgraphs.index(main_island)

        #   c. Connect subgraphs to main island
        connector = list(subgraphs[max_i].nodes())[0]
        for i in range(len(subgraphs)):
            if i != max_i:
                connection = list(subgraphs[i].nodes())[0]
                G.add_edge(connector, connection, highway="trunk", length=0) # clever, but does this create problems?

        # 2. Add additional weights for different surface types
        for v1, v2, _ in G.edges(data=True):
            for k in G[v1][v2]:
                # a. Get the highway attribute
                highway_attr = G[v1][v2][k].get('highway', [])

                # b. Ensure it's a list so we can iterate consistently
                highways = highway_attr if isinstance(highway_attr, list) else [highway_attr]

                # c. Use any() to see if any part of that road is paved
                is_paved = any(self.mapping_surfaces.get(h) == "paved" for h in highways)
                surface = "paved" if is_paved else "dirt"

                # d. Set edge attributes depending on surface type
                G[v1][v2][k]["weight"] = G[v1][v2][k]['length'] if surface=="paved" else self.DIRT_FACTOR * G[v1][v2][k]['length']
                G[v1][v2][k]["surface"] = surface
                G[v1][v2][k]["weight_paved"] = G[v1][v2][k]['length'] if surface=="paved" else 0
                G[v1][v2][k]["weight_dirt"] = self.DIRT_FACTOR * G[v1][v2][k]['length'] if surface!="paved" else 0

        # 3. Remove multi edges
        for v1, v2 in list(G.edges()):
            if G.has_edge(v1, v2): # Check because we might have deleted the pair already
                # Find the key of the edge with the minimum weight
                min_key = min(G[v1][v2].items(), key=lambda x: x[1]["weight"])[0]
                
                # Remove all edges except the winner
                for k in list(G[v1][v2]):
                    if k != min_key:
                        G.remove_edge(v1, v2, key=k)

        # We construct a new undirected graph from G
        new_G = nx.Graph()
        for v1, v2, data in G.edges(data=True):
            keys = list(G[v1][v2].keys())
            assert(len(keys) == 1) # sanity check that multi edges removed
            new_G.add_edge(v1, v2, **data)

        assert(nx.is_connected(new_G)) # sanity check that graph is connected

        return new_G

    @staticmethod
    def _prune_graph(G: nx.Graph, not_to_touch: set) -> bool:
        "Takes a graph and prunes it, returns True if the graph was pruned completely"
        initial_len = len(G)

        # 1. Remove all unprotected degree-1 nodes (dead ends)
        dead_ends = [node for node, deg in G.degree() if (deg == 1) and (node not in not_to_touch)]
        G.remove_nodes_from(dead_ends)

        # 2. Repeat the same procedure for nodes of degree 2, merging them
        while True:
            target = None
            for node, deg in G.degree():
                if (deg == 2) and (node not in not_to_touch):
                    target = node
                    break
            
            # Exit if no removable nodes of degree 2 found
            if target is None:
                break
            
            target_neighbors = list(G.neighbors(target))
            assert(len(target_neighbors) == 2) # confirm node is degree 2

            # Get connecting edges from target to neighbors
            edge1 = (target, target_neighbors[0])
            edge2 = (target, target_neighbors[1])

            # Sanity checks
            assert(edge1 in G.edges())
            assert(edge2 in G.edges())
            assert(edge1[0] == edge2[0]) # check both edges feed into node

            # Combine edge attributes
            weight1, weight2 = G[edge1[0]][edge1[1]]["weight"], G[edge1[0]][edge2[1]]["weight"]
            new_weight = weight1 + weight2
            new_weight_paved = G[edge1[0]][edge1[1]]["weight_paved"] + G[edge2[0]][edge2[1]]["weight_paved"]
            new_weight_dirt = G[edge1[0]][edge1[1]]["weight_dirt"] + G[edge2[0]][edge2[1]]["weight_dirt"]

            # Remove node
            G.remove_node(target)

            # Check if there are any edges between neighbors before adding a new edge
            if G.has_edge(target_neighbors[0], target_neighbors[1]):
                existing_weight = G[target_neighbors[0]][target_neighbors[1]]["weight"]
                # If new weight is less than existing weight, replace the existing weight
                if existing_weight > new_weight:
                    G.remove_edge(target_neighbors[0], target_neighbors[1])
                    G.add_edge(
                        target_neighbors[0],
                        target_neighbors[1],
                        weight=new_weight, 
                        weight_dirt=new_weight_dirt, 
                        weight_paved=new_weight_paved
                    )
            else:
                G.add_edge(
                    target_neighbors[0], 
                    target_neighbors[1], 
                    weight=new_weight, 
                    weight_dirt=new_weight_dirt, 
                    weight_paved=new_weight_paved
                )

        final_len = len(G)
        return initial_len == final_len # check if graph was pruned

    @staticmethod
    def iteratively_prune(G: nx.Graph, not_to_touch: set) -> None:
        "Iteratively prunes the graph in-place without touching the nodes in not_to_touch"
        while True:
            if RoadGraph._prune_graph(G, not_to_touch):
                break

    def build_tree(self, all_ids, all_stops, root, plot=True) -> tuple[nx.Graph, list]:
        "Calculates an approximation to the steiner tree and returns the tree and the edges that connect the root to all stops"
        # 1. Copy and iteratively prune 
        G_pruned = self.undirected_graph.copy()
        self.iteratively_prune(G_pruned, all_stops) # prune graph copy in-place
        
        # 2. Create Graph and add direct edges between stops
        complete_graph = nx.Graph()
        for stop1, stop2 in combinations(all_stops, 2):
            complete_graph.add_edge(
                    stop1,
                    stop2,
                    weight=nx.shortest_path_length(G_pruned, stop1, stop2, weight="weight")
                )
        T_complete = nx.minimum_spanning_tree(complete_graph, weight = "weight")
        
        # 3. Add shortest path edges between stops
        edges_to_add = set()
        for edge in T_complete.edges():
            path = nx.shortest_path(self.undirected_graph, edge[0], edge[1], weight="weight")
            for s in range(len(path) - 1):
                edge1 = (path[s], path[s + 1])
                edge2 = (path[s + 1], path[s])
                if (edge1 not in edges_to_add) and (edge2 not in edges_to_add):
                    edges_to_add.add(edge1)

        T = self.undirected_graph.edge_subgraph(list(edges_to_add))
        T = nx.minimum_spanning_tree(T, weight="weight")

        # 4. Create a subgraph of the original graph containing only the edges in T that are in between the stops
        edges_to_add_subgraph = set()
        for stop1, stop2 in combinations(all_stops, 2):
            path = nx.shortest_path(T, stop1, stop2, weight="weight")
            for s in range(len(path) - 1):
                edge1 = (path[s], path[s + 1])
                edge2 = (path[s + 1], path[s])
                if (edge1 not in edges_to_add_subgraph) and (edge2 not in edges_to_add_subgraph):
                    edges_to_add_subgraph.add(edge1)

        edges_subset = []
        for u, v in edges_to_add_subgraph:
            edges_subset.extend(self.graph.edges([u, v], keys=True))  # Get all multi-edges with their keys

        subgraph = self.graph.edge_subgraph(edges_subset)

        # 5. Set the surface attribute for each edge in the subgraph, remember that this is a multi-graph
        for u, v in subgraph.edges():
            for key in subgraph[u][v]:
                subgraph[u][v][key]["surface"] = self.undirected_graph[u][v]["surface"]

        self.iteratively_prune(T, all_stops)
        for stop in all_stops:
            assert(stop in T.nodes())

        # 6. Add a node per each id in all_ids and connect it to the stop corresponding to the id
        for i in range(len(all_ids)):
            T.add_node(all_ids[i])
            T.add_edge(all_ids[i], all_stops[i], weight=0)

        # 7. Calculate edges that connect the root to all stops
        root_edges = []
        for node_id in all_ids:
            path = nx.shortest_path(T, root, node_id, weight="weight")
            if len(path) > 1:
                path_edges = list(zip(path[:-1], path[1:]))
            else:
                path_edges = []
            root_edges.append(path_edges)
        
        if plot:
            self.plot_graph(subgraph)

        self.subgraph = subgraph

        return T, root_edges


    def plot_graph(self, subgraph: nx.MultiDiGraph, options=None) -> None:
        # Update global plot parameters
        plt.rcParams.update({
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9
        })

        # Set default options if not provided
        if options is None:
            options = {}
        figsize = options.get("figsize", (6, 6))  # Default figsize
        save_path = options.get("save_path", None)  # Default to not saving

        # Set up plot and axis
        _, ax = plt.subplots(figsize=figsize)
        
        # Plot the main graph
        ox.plot_graph(
            self.graph,
            ax=ax,
            node_color="lightgrey",
            edge_color="lightgrey",
            show=False,
            close=False
        )
        
        # Plot the edge subgraph with colors depending on the surface attribute
        edge_colors = [
            "#000000" if subgraph[u][v][key]["surface"] == "paved" else "#ff0000"  # Black for paved, red for unpaved
            for u, v, key in subgraph.edges(keys=True)
        ]
        
        # Extract node colors based on the edges connected to them
        node_colors = {}
        for u, v, key in subgraph.edges(keys=True):
            edge_color = "#000000" if subgraph[u][v][key]["surface"] == "paved" else "#ff0000"
            if u not in node_colors or node_colors[u] == "#ff0000":
                node_colors[u] = edge_color
            if v not in node_colors or node_colors[v] == "#ff0000":
                node_colors[v] = edge_color

        # Apply node colors to the plot
        ox.plot_graph(
            subgraph,
            ax=ax,
            edge_color=edge_colors,  # Color edges based on surface attribute
            node_size=2.5,
            edge_linewidth=2.5,
            node_color=[node_colors[node] for node in subgraph.nodes()],
            show=False,
            close=False
        )

        # Add legend for edge colors
        paved_patch = Line2D([0], [0], color="#000000", lw=2, label="Paved")  # Black for paved
        unpaved_patch = Line2D([0], [0], color="#ff0000", lw=2, label="Unpaved")  # Red for unpaved
        plt.legend(handles=[paved_patch, unpaved_patch], loc="upper right")

        if options:
            if "mill_node" in options and "farmer_nodes" in options:
                mill_node = options["mill_node"]
                # Draw a circle at the mill_node
                mill_node_coords = subgraph.nodes[mill_node]["x"], subgraph.nodes[mill_node]["y"]
                ax.scatter(
                    *mill_node_coords,
                    c="blue",
                    s=100,
                    label="Mill Node",
                    zorder=3
                )
                farmer_nodes = options["farmer_nodes"]
                sizes = options.get("residual", [1] * len(farmer_nodes))
                # Draw circles for farmer nodes
                for farmer_node, size in zip(farmer_nodes, sizes):
                    farmer_node_coords = subgraph.nodes[farmer_node]["x"], subgraph.nodes[farmer_node]["y"]
                    ax.scatter(
                        *farmer_node_coords,
                        c="green",
                        s=size * 100,  # Scale size for better visibility
                        label="Farmer Node",
                        zorder=3
                    )

        # Tighten the layout
        plt.tight_layout()

        # Save to PDF if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()