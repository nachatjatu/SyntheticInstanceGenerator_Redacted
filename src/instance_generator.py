"""
This Python file implements the InstanceGenerator class for simulating realistic farmer-intermediary networks
according to real-world data obtained from our partners in Indonesia. 

Code by Nachat Jatusripitak
"""

"""
IMPORT PACKAGES
"""
# data i/o 
import pickle
import os
import yaml

# data processing 
import numpy as np
import pandas as pd

# geo 
from pyproj import Transformer
import osmnx as ox

# statistical 
from scipy.stats import gaussian_kde
from scipy.special import logsumexp, gammaln
from scipy.interpolate import interp1d

# plotting 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection

# name generator package
from names_generator import generate_name


"""
DEFINE GLOBAL CONSTANTS
"""
INDO_CRS = "EPSG:23867"                         # Indonesia projected CRS
LL_CRS = "EPSG:4326"                            # WGS84 Lat/Lon
MIN_CAPACITY, MAX_CAPACITY = 2, 9               # feel free to change this
RES = 250                                       # spatial grid resolution (meters)
MAX_DIST = 63000                                # max. sampling distance
PRECOMPUTED_SIGMAS = {} # REDACTED
FALLBACK_SIGMA = 5000
DEFAULT_MILL = {}      # REDACTED
DEFAULT_KDE_BANDWIDTH_FACTOR = 0.2
KDE_DIST_BUFFER = 10000
N_GAMMA_STEPS = 2000
HIGH_INT, MED_INT, LOW_INT = None, None, None # REDACTED


"""
INSTANCEGENERATOR CLASS
"""
class InstanceGenerator:
    """
    Generate synthetic farmer-intermediary pickup instances from empirical data.

    This class builds stochastic daily pickup networks using historical farmer
    pickup records, empirical intermediary locations, and a regional road graph
    from the study area. It estimates spatial priors for intermediaries and
    farmers with kernel density estimation, models farmer-intermediary distances
    with intermediary-specific smoothed empirical distance distributions, and
    generates clustered farmer locations for each intermediary.

    Generated instances are formatted as dictionaries containing farmers,
    intermediaries, routes, quantities, and mills. Instances can optionally be
    written to YAML or visualized on top of the projected road network.

    Parameters
    ----------
    farmers_full_df_path : str, default="data/farmers.csv"
        Path to the full farmer pickup dataset. Expected to include farmer
        coordinates, intermediary IDs, pickup distances, and dates.
    farmers_14_df_path : str, default="data/farmers_2.csv"
        Path to the 14-day farmer pickup dataset used to sample daily farmer
        counts and pickup quantities.
    ints_df_path : str, default="data/ints.csv"
        Path to the intermediary dataset. Expected to include intermediary IDs
        and projected intermediary coordinates.
    graph_path : str, default="data/graph_0-14960_00_new.pickle"
        Path to a pickled OSMnx road graph for the study region.

    Attributes
    ----------
    xy_to_ll : pyproj.Transformer
        Transformer from the projected Indonesia CRS to WGS84 longitude/latitude.
    ll_to_xy : pyproj.Transformer
        Transformer from WGS84 longitude/latitude to the projected Indonesia CRS.
    farmers_full_df : pandas.DataFrame
        Full empirical farmer pickup dataset, filtered to intermediary IDs shared
        across all input datasets.
    farmers_14_df : pandas.DataFrame
        Historical 14-day farmer pickup dataset, filtered to intermediary IDs
        shared across all input datasets.
    ints_df : pandas.DataFrame
        Empirical intermediary dataset, filtered to intermediary IDs shared
        across all input datasets.
    G : networkx.MultiDiGraph
        Original OSMnx road graph loaded from ``graph_path``.
    G_proj : networkx.MultiDiGraph
        Road graph projected to ``INDO_CRS``.
    bbox_m : np.ndarray
        Projected graph bounding box as ``(min_x, min_y, max_x, max_y)``.
    grid_coords : np.ndarray
        Spatial grid coordinates in projected CRS, with shape
        ``(2, n_grid_points)``.
    int_spatial_kde : scipy.stats.gaussian_kde
        Spatial KDE fit to empirical intermediary locations.
    farmer_spatial_kde : scipy.stats.gaussian_kde
        Spatial KDE fit to empirical farmer pickup locations.
    gamma_lookups : dict
        Mapping from empirical intermediary IDs to interpolated distance-density
        functions.
    sigmas : dict
        Clustering bandwidths, in meters, keyed by empirical intermediary type.
    p_spatial : np.ndarray
        Normalized farmer spatial prior evaluated over ``grid_coords``.
    hist_quantities : dict
        Historical pickup quantities keyed by empirical intermediary ID.
    hist_n_farmers : dict
        Historical daily farmer counts keyed by empirical intermediary ID.
    ints : dict
        Generated intermediary metadata keyed by synthetic intermediary ID.
    mills : list of dict
        Mill locations included in generated instances.

    Notes
    -----
    All internal spatial sampling is performed in the projected coordinate
    system ``INDO_CRS``. Generated instance locations are exported as
    latitude/longitude pairs.
    """
    def __init__(self, 
                 farmers_full_df_path="data/farmers.csv",               # full dataset of farmer pickups
                 farmers_14_df_path="data/farmers_2.csv",               # original 14-day dataset of farmer pickups
                 ints_df_path="data/ints.csv",                          # full dataset of intermediaries
                 graph_path="data/graph_0-14960_00_new.pickle"):        # pickle file of regional road graph (osmnx)
        
        # CRS transformers
        self.xy_to_ll = Transformer.from_crs(INDO_CRS, LL_CRS, always_xy=True)
        self.ll_to_xy = Transformer.from_crs(LL_CRS, INDO_CRS, always_xy=True)

        # load empirical data 
        self.farmers_full_df = pd.read_csv(farmers_full_df_path)
        self.farmers_14_df = pd.read_csv(farmers_14_df_path)
        self.ints_df = pd.read_csv(ints_df_path)
        self.G, self.G_proj, self.bbox_m = self._init_graph(graph_path)

        # resolve int_id differences
        shared_int_ids = (
            set(self.farmers_full_df["int_id"])
            & set(self.farmers_14_df["int_id"])
            & set(self.ints_df["int_id"])
        )
        self.farmers_full_df = self.farmers_full_df[
            self.farmers_full_df["int_id"].isin(shared_int_ids)
        ]
        self.farmers_14_df = self.farmers_14_df[
            self.farmers_14_df["int_id"].isin(shared_int_ids)
        ]
        self.ints_df = self.ints_df[
            self.ints_df["int_id"].isin(shared_int_ids)
        ]
        
        # create spacial grid
        x_ax = np.arange(self.bbox_m[0], self.bbox_m[2], RES)
        y_ax = np.arange(self.bbox_m[1], self.bbox_m[3], RES)
        gx, gy = np.meshgrid(x_ax, y_ax, indexing="ij")
        self.grid_coords = np.vstack([gx.ravel(), gy.ravel()])

        # initialize KDEs
        self.int_spatial_kde = self._init_int_kde()
        self.farmer_spatial_kde = self._init_farmer_kde()
        self.gamma_lookups = self._init_gamma_kdes()

        # sigma values for clustering intensity (precomputed)
        self.sigmas = PRECOMPUTED_SIGMAS
        
        # precompute farmer spatial priors on grid
        p_spatial = self.farmer_spatial_kde.evaluate(self.grid_coords)
        self.p_spatial = p_spatial / (p_spatial.sum() + 1e-20)

        # cache historical statistics
        self.hist_quantities = (self.farmers_14_df
                                .groupby("int_id")["quantity"]
                                .apply(list)
                                .to_dict())
        
        counts_df = (self.farmers_14_df
                     .groupby(["int_id", "date"])
                     .size()
                     .reset_index(name="count"))
        self.hist_n_farmers = (counts_df
                               .groupby("int_id")["count"]
                               .apply(list)
                               .to_dict())

        # internal storage of instance parameters
        self.ints = {}
        self.mills = [DEFAULT_MILL]


    """
    --------------
    INITIALIZATION
    --------------
    This section contains helper functions that initialize the spatial aspects of the InstanceGenerator.
    In particular, it contains functions that load/project the road graph and initialize KDEs for
    generating farmers and intermediaries.
    """
    def _init_graph(self, graph_path):
        """
        Load and project the regional road graph.

        Parameters
        ----------
        graph_path : str
            Path to a pickled OSMnx graph.

        Returns
        -------
        tuple
            ``(G, G_proj, bbox_m)``, where ``G`` is the original graph,
            ``G_proj`` is the graph projected to ``INDO_CRS``, and ``bbox_m`` is
            the projected node bounding box in the form
            ``(min_x, min_y, max_x, max_y)``.
        """
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        G_proj = ox.project_graph(G, to_crs=INDO_CRS)
        nodes_proj, _ = ox.graph_to_gdfs(G_proj)
        return G, G_proj, nodes_proj.total_bounds
    
    def _init_int_kde(self):
        """
        Initialize a spatial KDE over empirical intermediary locations.

        Returns
        -------
        scipy.stats.gaussian_kde
            KDE fit to unique intermediary projected coordinates.
        """
        coords = (self.ints_df
                  .drop_duplicates(["int_id"])[["int_x", "int_y"]].T) # type: ignore
        return gaussian_kde(coords, bw_method=DEFAULT_KDE_BANDWIDTH_FACTOR)

    def _init_farmer_kde(self):
        """
        Initialize a spatial KDE over empirical farmer pickup locations.

        Returns
        -------
        scipy.stats.gaussian_kde
            KDE fit to unique farmer projected coordinates.
        """
        coords = (self.farmers_full_df
                  .drop_duplicates(["farmer_x", "farmer_y"])[["farmer_x", "farmer_y"]].T) # type: ignore
        return gaussian_kde(coords, bw_method=DEFAULT_KDE_BANDWIDTH_FACTOR)
    
    def _init_gamma_kdes(self):
        """
        Initialize intermediary-specific smoothed distance-density lookups.

        For each empirical intermediary, this method fits a smoothed distribution
        over historical farmer-intermediary distances using a Gamma-kernel mixture.
        The resulting density is stored as an interpolated lookup function for
        efficient evaluation during farmer generation.

        Returns
        -------
        dict
            Mapping from intermediary ID to an ``interp1d`` function that evaluates
            the estimated distance density.
        """
        # get historical distances by intermediary
        int_to_dists = (self.farmers_full_df
                        .drop_duplicates(["int_id", "farmer_x", "farmer_y"]) # type: ignore
                        .groupby("int_id")["distance"]
                        .apply(np.array)
                        .to_dict())
        
        lookups = {}
        x_eval = np.linspace(0, MAX_DIST + KDE_DIST_BUFFER, N_GAMMA_STEPS)

        # for each intermediary, fit a smoothed distance distribution
        for i_id, dists in int_to_dists.items():
            n = len(dists)
            h = 0.1 * np.mean(dists) + 1e-6 # bandwidth rule with 1e-6 safety buffer
            shape = dists / h
            # evaluate Gamma log-PDF for each x
            pdf_values = np.zeros_like(x_eval)
            for i in range(n):
                s, scale = shape[i], h
                with np.errstate(divide="ignore", invalid="ignore"):
                    # Gamma log-PDF: (s-1)*log(x) - x/scale - (log(gamma(s)) + s*log(scale))
                    log_pdf = (
                        (s - 1) * np.log(x_eval + 1e-10) 
                        - (x_eval / scale) 
                        - (gammaln(s + 1e-10) + s * np.log(scale))
                    )
                pdf_values += np.exp(log_pdf)
            pdf_values /= n
            # interpolate for efficiency
            lookups[i_id] = interp1d(x_eval, pdf_values, fill_value=(0,0), bounds_error=False) 
        return lookups
    
    def _init_sigmas(self):
        """
        Estimate and store clustering bandwidths for generated intermediaries.

        The method computes an adaptive maximum-likelihood clustering bandwidth
        for each intermediary currently stored in ``self.ints`` and assigns the
        result to ``self.sigmas``.

        Returns
        -------
        None
        """
        self.sigmas = {int_id: self.find_mle_sigma_adaptive(int_id) for int_id in self.ints}
    
    
    """
    ----------
    GENERATION
    ----------
    This section contains helper functions that generate intermediaries, their daily farmer pickups, and
    the aggregate daily instance. 
    """
    def gen_ints(self, n_ints, seed, set_type=None):
        """
        Generate synthetic intermediaries.

        Intermediary locations are sampled from the empirical intermediary spatial
        KDE using rejection sampling within the road-graph bounding box. Each
        generated intermediary is assigned a unique synthetic name and an empirical
        intermediary type, which determines downstream distance and quantity
        sampling behavior.

        Parameters
        ----------
        n_ints : int
            Number of intermediaries to generate.
        seed : int
            Random seed used to create reproducible intermediary-specific RNGs.
        set_type : {"high", "medium", "low", None}, optional
            If provided, restricts generated intermediaries to one representative
            empirical type. If ``None``, types are sampled from all available
            empirical intermediary types.

        Returns
        -------
        None
            Generated intermediaries are stored in ``self.ints``.
        """
        # create random seeds for reproducibility, one per intermediary
        seed_seq = np.random.SeedSequence(seed)
        int_seeds = seed_seq.spawn(n_ints)
        rngs = [np.random.default_rng(int_seed) for int_seed in int_seeds]

        # initialize possible intermediary types
        if set_type == "high":
            types = [HIGH_INT]
        elif set_type == "medium":
            types = [MED_INT]
        elif set_type == "low":
            types = [LOW_INT]
        else:
            types = list(self.gamma_lookups.keys())

        ints = {}
        names = set()
        for i in range(n_ints):
            rng = rngs[i]

            # generate unique name (no duplicates)
            while True:
                int_id = generate_name(seed=int(rng.integers(0, 2**32 - 1)))
                if int_id not in names:
                    names.add(int_id)
                    break

            int_type = rng.choice(types)
            
            # rejection sampling within bounding box
            while True:
                sample = self.int_spatial_kde.resample(1, seed=rng).flatten()
                if (self.bbox_m[0] <= sample[0] <= self.bbox_m[2] and 
                    self.bbox_m[1] <= sample[1] <= self.bbox_m[3]):
                    int_xy = sample
                    break
            
            lon, lat = self.xy_to_ll.transform(int_xy[0], int_xy[1])
            ints[int_id] = {"xy": int_xy, "ll": (lat, lon), "type": int_type}

        self.ints = ints


    def gen_farmers(self, int_xy, int_type, n_farmers, rng, sigma=500):
        """
        Generate farmer locations for a single intermediary.

        Farmer locations are sampled sequentially over the spatial grid. The base
        probability combines an intermediary-specific distance density with the
        global farmer spatial prior. After the first farmer is sampled, subsequent
        farmers are conditionally biased toward previously sampled locations using
        a Gaussian clustering kernel.

        Parameters
        ----------
        int_xy : array-like of shape (2,)
            Projected ``(x, y)`` coordinates of the intermediary.
        int_type : str
            Empirical intermediary type used to select the distance-density lookup.
        n_farmers : int
            Number of farmer locations to generate.
        rng : np.random.Generator
            Random number generator used for sampling.
        sigma : float, default=500
            Clustering bandwidth in meters. Larger values produce more spatially
            diffuse clusters.

        Returns
        -------
        np.ndarray
            Generated farmer projected coordinates with shape ``(n_farmers, 2)``.
        """
        # precompute distance from intermediary to each grid point
        dist_lookup = self.gamma_lookups[int_type]
        grid_points = self.grid_coords.T 
        dists = np.linalg.norm(grid_points - int_xy, axis=1)
        
        # compute base log probabilities for each distance
        p_dist_raw = dist_lookup(dists)
        log_p_base = np.log(p_dist_raw + 1e-20) + np.log(self.p_spatial + 1e-20)
        log_p_base -= logsumexp(log_p_base)

        locs = []
        sigma_sq_2 = 2 * (sigma ** 2)
        acc_exp_kernels = np.zeros(len(grid_points))

        for k in range(n_farmers):
            if k == 0:
                log_p_cond = log_p_base
            else:
                # bayesian update: clustering influence (add in log space)
                log_local_factor = np.log(acc_exp_kernels + 1e-20) - np.log(k)
                log_p_cond = log_p_base + log_local_factor
                log_p_cond -= logsumexp(log_p_cond)

            p_sampling = np.exp(log_p_cond)
            
            # numerical stability fallback
            if np.isnan(p_sampling).any() or p_sampling.sum() == 0:
                p_sampling = self.p_spatial

            # sample farmer locations
            idx = rng.choice(len(p_sampling), p=p_sampling/p_sampling.sum())
            sampled_xy = self.grid_coords[:, idx]
            locs.append(sampled_xy)
            
            # update kernel for next farmer in sequence
            new_dist_sq = np.sum((grid_points - sampled_xy)**2, axis=1)
            acc_exp_kernels += np.exp(-new_dist_sq / sigma_sq_2)

        return np.array(locs)
    

    def gen_instance(self, instance_id, seed, write=False, plot=False, scale_factor=1.0, sigma=None):
        """
        Generate a complete daily farmer-intermediary instance.

        For each generated intermediary in ``self.ints``, this method samples a
        daily farmer count from historical data, generates farmer locations, samples
        farmer quantities, rescales quantities to satisfy intermediary capacity
        constraints, and formats the result as an instance dictionary.

        Parameters
        ----------
        instance_id : str
            Identifier for the generated instance.
        seed : int
            Random seed used to create reproducible intermediary-specific RNGs.
        write : bool, default=False
            Whether to write the generated instance to ``data/instances/{instance_id}.yaml``.
        plot : bool, default=False
            Whether to plot the generated instance after creation.
        scale_factor : float, default=1.0
            Multiplicative factor applied to sampled farmer counts. Fractional
            counts are rounded using Bernoulli randomized rounding.
        sigma : float, optional
            Clustering bandwidth to use for all intermediaries. If ``None``,
            the method uses a precomputed bandwidth for each intermediary type,
            falling back to ``FALLBACK_SIGMA`` when unavailable.

        Returns
        -------
        dict
            Generated instance with keys ``"instance_id"``, ``"farmers"``,
            ``"intermediaries"``, and ``"mills"``.

        Raises
        ------
        ValueError
            If positive one-decimal farmer quantities cannot be adjusted to satisfy
            the maximum capacity constraint.
        AssertionError
            If the final scaled quantities do not satisfy the capacity constraints.
        """
        # create random seeds for reproducibility, one per intermediary
        int_ids = list(self.ints.keys())
        seed_seq = np.random.SeedSequence(seed)
        int_seeds = seed_seq.spawn(len(int_ids))
        rngs = {
            int_id: np.random.default_rng(int_seed)
            for int_id, int_seed in zip(int_ids, int_seeds)
        }

        # generate farmers for each intermediary
        farmers, ints = [], []
        for int_id in int_ids:
            int_data = self.ints[int_id]
            rng = rngs[int_id]

            # sample intermediary type and location
            int_type, int_xy, int_ll = int_data["type"], int_data["xy"], int_data["ll"]

            # sample number of farmers in intermediary's network
            n_farmers = rng.choice(self.hist_n_farmers[int_type])
            raw_n = n_farmers * scale_factor
            n_farmers = int(np.floor(raw_n) + (rng.random() < (raw_n % 1))) # Bernoulli using scale_factor
            
            if n_farmers > 0:
                # generate farmer locations
                sigma_i = sigma if sigma is not None else self.sigmas.get(int_type, FALLBACK_SIGMA)
                farmer_xys = self.gen_farmers(int_xy, int_type, n_farmers, rng, sigma=sigma_i)

                # generate farmer quantities
                qs = np.array([
                    rng.choice(self.hist_quantities[int_type], replace=True)
                    for _ in range(n_farmers)
                ])

                # rescale quantities to fit intermediary capacity constraints
                total_q = qs.sum()
                if total_q >= MAX_CAPACITY:
                    target_total = MAX_CAPACITY - 0.1
                elif total_q < MIN_CAPACITY:
                    target_total = MIN_CAPACITY
                else:
                    target_total = total_q
                qs_scaled = np.round(qs * (target_total / total_q), 1)
                qs_scaled = np.maximum(qs_scaled, 0.1)

                while qs_scaled.sum() >= MAX_CAPACITY:
                    idx = np.argmax(qs_scaled)
                    if qs_scaled[idx] <= 0.1:
                        raise ValueError("Cannot satisfy MAX_CAPACITY with positive 1-decimal quantities.")
                    qs_scaled[idx] = np.round(qs_scaled[idx] - 0.1, 1)

                while qs_scaled.sum() < MIN_CAPACITY:
                    idx = np.argmin(qs_scaled)
                    qs_scaled[idx] = np.round(qs_scaled[idx] + 0.1, 1)

                assert MIN_CAPACITY <= qs_scaled.sum() < MAX_CAPACITY, qs_scaled.sum()

                # format and append farmers
                routes = []
                for f in range(n_farmers):
                    f_id = f"{int_id}_f{f}"
                    f_lon, f_lat = self.xy_to_ll.transform(farmer_xys[f][0], farmer_xys[f][1])
                    
                    farmers.append({
                        "id": f_id, 
                        "location": [f_lat, f_lon],
                        "quantity": float(qs_scaled[f]),
                        "intermediary": int_id
                    })
                    routes.append(f_id)
                
                # append intermediaries
                ints.append({
                    "id": int_id, 
                    "capacity": MAX_CAPACITY, 
                    "location": list(int_ll), 
                    "routes": [routes]
                })
        
        instance = {"instance_id": instance_id,
                    "farmers": farmers, 
                    "intermediaries": ints, 
                    "mills": self.mills}

        # write and plot (if desired)
        if write:
            os.makedirs("data/instances", exist_ok=True)
            with open(f"data/instances/{instance_id}.yaml", "w") as file:
                yaml.dump(instance, file, default_flow_style=False)   
        if plot:
            self.plot_instance(instance) 

        return instance
    

    """
    ----------
    AUXILIARY
    ----------
    This section contains helper functions that plots generated instances and optimizes sigma using MLE.
    """

    def plot_instance(self, instance_data):
        """
        Plot a generated farmer-intermediary instance.

        The plot includes the farmer spatial prior, projected road network, mill
        locations, intermediary locations, farmer locations, and lines connecting
        each intermediary to its assigned farmers.

        Parameters
        ----------
        instance_data : dict
            Instance dictionary produced by ``gen_instance``. Expected to contain
            ``"farmers"``, ``"intermediaries"``, and ``"mills"``.

        Returns
        -------
        None
            Displays a Matplotlib figure.
        """
        plt.figure(figsize=(14, 11))
        
        # plot farmer location KDE
        x_coords = np.unique(self.grid_coords[0])
        y_coords = np.unique(self.grid_coords[1])
        Z = self.p_spatial.reshape(len(x_coords), len(y_coords))
        
        plt.imshow(
            Z.T, 
            origin="lower", 
            extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
            cmap="magma",
            aspect="equal"
        )
        
        # plot road map
        lines = []
        for u, v, data in self.G_proj.edges(data=True):
            if "geometry" in data:
                xs, ys = data["geometry"].xy
                lines.append(list(zip(xs, ys)))
            else:
                u_node = self.G_proj.nodes[u]
                v_node = self.G_proj.nodes[v]
                lines.append([(u_node["x"], u_node["y"]), (v_node["x"], v_node["y"])])
                
        lc = LineCollection(lines, colors="gray", linewidths=0.5, alpha=0.4, zorder=1)
        plt.gca().add_collection(lc)

        # plot mill
        for mill in instance_data["mills"]:
            x, y = self.ll_to_xy.transform(mill["location"][1], mill["location"][0])
            plt.scatter(x, y, c="white", marker="*", s=400, label="Mill", zorder=10)

        # plot clusters
        colors = cm.get_cmap("tab10", len(instance_data["intermediaries"]))
        farmer_lookup = {f["id"]: f for f in instance_data["farmers"]}

        for i, intermediary in enumerate(instance_data["intermediaries"]):
            color = colors(i)
            ix, iy = self.ll_to_xy.transform(intermediary["location"][1], intermediary["location"][0])
            
            # Intermediary marker
            plt.scatter(ix, iy, color=color, marker="s", s=120, edgecolors="k", zorder=9, label=intermediary["id"])

            # Farmer markers and lines
            for f_id in intermediary["routes"][0]:
                farmer = farmer_lookup[f_id]
                fx, fy = self.ll_to_xy.transform(farmer["location"][1], farmer["location"][0])
                
                plt.scatter(fx, fy, color=color, s=40, edgecolors="white", zorder=8)
                plt.plot([ix, fx], [iy, fy], color=color, lw=1.5, alpha=0.6, zorder=7)

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
        plt.tight_layout()
        plt.show()


    def find_mle_sigma_adaptive(self, int_type, start_sigma=2500, step=500):
        """
        Estimate a clustering bandwidth for an empirical intermediary type.

        The method evaluates the likelihood of historical daily farmer locations
        under the sequential farmer-location model and performs a simple
        one-directional hill climb over ``sigma`` values. The search stops once
        increasing ``sigma`` no longer improves the historical log-likelihood.

        Parameters
        ----------
        int_type : str
            Empirical intermediary ID/type for which to estimate the clustering
            bandwidth.
        start_sigma : float, default=2500
            Initial clustering bandwidth in meters.
        step : float, default=500
            Increment used during hill-climbing search.

        Returns
        -------
        float
            Estimated clustering bandwidth in meters.
        """
        best_sigma = start_sigma
        best_ll = -np.inf
        current_sigma = start_sigma
        
        # precompute base prior for the specific intermediary location
        int_data = self.ints_df[self.ints_df["int_id"] == int_type].iloc[0]
        int_xy = np.array([int_data["int_x"], int_data["int_y"]])
        dists = np.linalg.norm(self.grid_coords.T - int_xy, axis=1)
        
        log_p_base = np.log(self.gamma_lookups[int_type](dists) + 1e-20) + np.log(self.p_spatial + 1e-20)
        
        # pre-map historical farmers to grid indices
        daily_groups = self.farmers_full_df[self.farmers_full_df["int_id"] == int_type].groupby("date")
        historical_indices = []
        for _, group in daily_groups:
            coords = group[["farmer_x", "farmer_y"]].values
            indices = [np.argmin(np.sum((self.grid_coords.T - c) ** 2, axis=1)) for c in coords]
            historical_indices.append(indices)

        # hill-climbing optimization
        while True:
            total_ll = 0
            sigma_sq_2 = 2 * (current_sigma ** 2)
            
            for f_indices in historical_indices:
                acc_exp_kernels = np.zeros(len(self.grid_coords[0]))
                
                for k, target_idx in enumerate(f_indices):
                    if k == 0:
                        log_p_cond = log_p_base
                    else:
                        log_local_factor = np.log(acc_exp_kernels + 1e-20) - np.log(k)
                        log_p_cond = log_p_base + log_local_factor
                    
                    log_p_cond -= logsumexp(log_p_cond)
                    total_ll += log_p_cond[target_idx]
                    
                    # update kernel density for sequential evaluation
                    sampled_xy = self.grid_coords[:, target_idx]
                    dist_sq = np.sum((self.grid_coords.T - sampled_xy)**2, axis=1)
                    acc_exp_kernels += np.exp(-dist_sq / sigma_sq_2)
            
            if total_ll > best_ll:
                best_ll = total_ll
                best_sigma = current_sigma
                current_sigma += step
            else:
                break # likelihood began decreasing
                
        return best_sigma
    