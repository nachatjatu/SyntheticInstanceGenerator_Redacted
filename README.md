# :game_die: Synthetic Instance Generator and Stable Platform Solver:

Hello and welcome to our repository! Our repo contains code for generating synthetic farmer-intermediary pickup instances. This codebase extends the awesome work by @serarca [link here](https://github.com/serarca/FactoredPlatformSolver), who implemented the stable matching & pricing solver on such instances. See original paper [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5499541).

Our motivating setting is a centralized platform in which smallholder farmers supply fruit for intermediaries to collect and deliver to a designated processing mill. The platform matches farmers to intermediaries and makes payments to both parties subject to stability constraints. 

This repository includes:
1. Empirical data-driven synthetic instance generator
2. Road-network-based routing and matching construction
3. Stable pricing and platform profit optimization solver
4. Experiments studying how platform outcomes scale with market size, farmer density, and transporation costs
5. Notebooks for exploratory data analysis, synthetic generator validation, and visualizations

We hope this repo is a useful resource for you! 

Sincerely,

Nachat :smile:

---

## 🌴 Repository Structure

```text
SyntheticInstanceGenerator/
├── data/
│   ├── farmers.csv
│   ├── farmers_2.csv
│   ├── ints.csv
│   ├── mills.csv
│   ├── graph_0-14960_00_new.pickle
│   ├── instances/
│   ├── results_exp_4_new/
│   └── results_exp_6_new/
├── experiments/
│   ├── experiment_4_new.py
│   ├── experiment_6_new.py
│   ├── scale_costs.py
│   ├── scale_n_farmers.py
│   ├── scale_n_ints.py
│   └── cluster/
│       ├── submit_experiment_4_new.sbatch
│       ├── submit_experiment_6_new.sbatch
│       ├── submit_scale_costs.sbatch
│       ├── submit_scale_n_farmers.sbatch
│       └── submit_scale_n_ints.sbatch
├── figures/
├── notebooks/
│   ├── eda.ipynb
│   ├── ig_dev.ipynb
│   ├── figure_matching.ipynb
│   ├── figure_prices.ipynb
│   ├── figure_scale_cost.ipynb
│   ├── figure_scale_n_farmers.ipynb
│   ├── figure_scale_n_ints.ipynb
│   └── figure_welfare.ipynb
├── src/
│   ├── farmers_intermediaries.py
│   ├── instance_generator.py
│   ├── pricing.py
│   ├── LP_solvers.py
│   ├── dynamic_solvers.py
│   └── road_graphs.py
├── requirements.txt
└── README.md
```

---

## ⚡ Installation

First, create and activate a virtual environment using the following commands:

```bash
python -m venv .venv
source .venv/bin/activate
```

Second, install required dependencies using ``requirements.txt``:

```bash
pip install -r requirements.txt
```

:warning: This solver requires Gurobi; make sure you have a valid Gurobi license before running optimization experiments!

---

## 📊 Data

The main empirical input files are stored in `data/`:

- `farmers.csv`: full farmer pickup data.
- `farmers_2.csv`: abbreviated (14-day) farmer pickup data used for sampling farmer counts and quantities.
- `ints.csv`: intermediary location and metadata.
- `mills.csv`: mill location data.
- `graph_0-14960_00_new.pickle`: road network graph used for routing.

Generated instances and experimental outputs are stored in directories such as:

```text
data/instances/
results/scale_costs/
results/scale_n_farmers/
results/scale_n_ints/
```

Older or existing experiment outputs may also appear under:

```text
data/results_exp_4_new/
data/results_exp_6_new/
```

---

## ⚙️ Core Modules

### `src/instance_generator.py`

Generates synthetic farmer–intermediary pickup instances from empirical data.

The generator samples:

1. Intermediary locations from an empirical spatial distribution.
2. Intermediary types from empirical intermediary categories.
3. Farmer counts from historical daily pickup counts.
4. Farmer locations from a distance-conditioned spatial model.
5. Farmer quantities from historical pickup quantities.
6. Rescaled quantities satisfying intermediary capacity constraints.

The output is an instance dictionary containing farmers, intermediaries, routes, and mills.

**Farmer location sampling**

Sampling realistic farmer networks for each intermediary from empirical data is not trivial. These networks respect local geography and are often spatially correlated. There is also significant heterogeneity across intermediaries: some operate highly clustered short-range networks whereas others operate sparse long-range networks. 

Our approach to sampling farmer locations attempts to address these details. We generate locations using a sequential spatial point process on a discretized grid. The first farmer is sampled from a baseline distribution proportional to the product of a global empirical farmer location prior and an intermediary-type-specific farmer-intermediary distance density. Subsequent farmers are sampled from this same baseline distribution reweighted by a Gaussian kernel density centered on previously sampled farmers, inducing local clustering within each intermediary's farmer network.

Formally, let the intermediary be located at $z \in \mathbb{R}^2$ with type $t$. Let the spatial grid be

$$
G = \{x_1, \ldots, x_M\}, \quad x_j \in \mathbb{R}^2.
$$

Let $g_t(r)$ be the empirical distance-density lookup for farmer-intermediary distances and let $p_0(x_j)$ be the global empirical farmer spatial prior over grid cells (using Gaussian KDE).

The baseline probability over grid cells is thus

$$
\pi_0(x_j | z, t) \propto g_t(\Vert x_j - z\Vert) p_0(x_j)
$$

where $\Vert\cdot \Vert$ is the Euclidean distance in the projected coordinate system. Intuitively, this distribution makes a farmer location more likely if (1) it is in an area dense with farmers and (2) its distance from the intermediary is plausible for that intermediary type.

The first farmer is sampled from this baseline:

$$
X_1 \sim \pi_0(\cdot |z, t).
$$

For subsequent farmers, we add a clustering factor based on the previously sampled locations. Suppose farmers $X_1, \ldots, X_{k-1}$ have already been sampled. Then, for a candidate grid point $x_j \in \mathbb{R}^2$, define

$$
C_{k-1}(x_j) = \frac{1}{k-1} \sum_{h=1}^{k-1} \exp{\left(-\frac{\Vert x_j - X_h\Vert^2}{2\sigma^2}\right)}.
$$

This is a Gaussian kernel density centered at previously sampled farmers. The next farmer is thus sampled according to 

$$
\pi_k (x_j | X_1, \ldots, X_{k-1}, z, t) \propto \pi_0(x_j | z, t) C_{k-1} (x_j).
$$

In summary, the full process is $X_1 \sim \pi_0$ and $X_k | X_1,\ldots, X_{k-1} \sim \pi_k$ for $k \geq 2$.

_Choosing clustering hyperparameter_ $\sigma$

In the procedure above, the hyperparameter $\sigma$ controls the clustering strength: for small $\sigma$, farmers cluster tightly around previously sampled farmers; for large $\sigma$, clustering is diffuse and approaches the baseline spatial-distance prior. 

We choose $\sigma$ using maximum likelihood estimation on our empirical data. The baseline un-normalized log-probability of grid cell $x_j$ is

$$
\log P(X_0 = x_j) = l_0(x_j) = \log g_t(\Vert x_j - z_t\Vert) + \log p_0(x_j).
$$

The conditional log probability of the $k$ th farmer is

$$
\log P(X_k = x_j | X_1, \ldots, X_{k-1}) = l_0(x_j) + \log C_{k-1})(x_j) - \log Z_{k}
$$

where $Z_k$ is a normalizing constant. 

For each historical day $d$, we evaluate the sequential log-likelihood

$$
\mathcal{L}_d(\sigma) = \sum_{k=1}^{n_d} \log P(X_k = x_j | X_1, \ldots, X_{k-1}).
$$

Then for an intermediary of type $t$, the total log-likelihood across _all days_ is

$$
\mathcal{L}(\sigma) = \sum_d\sum_{k=1}^{n_d} \log P(X_k = x_j | X_1, \ldots, X_{k-1}).
$$

We search for $\sigma$ using a simple one-directional hill-climbing search and cache it for efficiency.


### `src/farmers_intermediaries.py`

Defines the core platform data structures, including:

- `Farmer`
- `Intermediary`
- `Mill`
- `Instance`
- `Route`
- `Matching`

These objects represent the generated pickup market, road-network distances, feasible pickup routes, and matchings.

### `src/pricing.py`

Implements the main platform optimization logic. The optimizer solves for stable prices, intermediary participation, platform profit, welfare allocations, and matching costs.

The solver first optimizes platform profit, then computes welfare-extreme tie-breaks among profit-optimal solutions.

### `src/LP_solvers.py`

Contains Gurobi-based routing and vehicle-routing formulations.

### `src/dynamic_solvers.py`

Contains dynamic-programming-based solvers used to accelerate route generation and pricing subproblems.

### `src/road_graphs.py`

Provides utilities for loading and interacting with road network graphs.

---

## ⚡ Quick Start

Run one seed of the cost-scaling experiment:

```bash
python experiments/scale_costs.py 0
```

Run one seed of the farmer-scaling experiment:

```bash
python experiments/scale_n_farmers.py 0
```

Run one seed of the intermediary-scaling experiment:

```bash
python experiments/scale_n_ints.py 0
```

Then open the corresponding analysis notebooks:

```text
notebooks/figure_scale_cost.ipynb
notebooks/figure_scale_n_farmers.ipynb
notebooks/figure_scale_n_ints.ipynb
```

---

## 🧪 Experiments

### Scaling Transportation Costs

The `scale_costs.py` experiment fixes one generated instance per seed and sweeps transportation cost multipliers.

```bash
python experiments/scale_costs.py 0
```

Each run takes a seed `n_id` and solves the same generated market under different transportation cost multipliers.

This experiment is paired within seed: the farmers, intermediaries, locations, and quantities are fixed, while transportation costs are varied.

Typical outputs are saved to:

```text
results/scale_costs/{n_id}.json
```

### Scaling the Number of Farmers

The `scale_n_farmers.py` experiment studies how outcomes change as generated farmer counts are scaled up or down.

```bash
python experiments/scale_n_farmers.py 0
```

Each run takes a seed `n_id` and sweeps over a set of farmer-count scale factors.

This is a common-random-seed scaling experiment. The seed is fixed within each sweep, but changing the scale factor changes the generated farmer network.

Typical outputs are saved to:

```text
results/scale_n_farmers/{n_id}.json
```

### Scaling the Number of Intermediaries

The `scale_n_ints.py` experiment studies how outcomes change as the number of intermediaries increases.

```bash
python experiments/scale_n_ints.py 0
```

Each run takes a seed `n_id` and sweeps over different numbers of intermediaries.

This is a market-scaling experiment: increasing the number of intermediaries also changes the generated farmer-intermediary market.

Typical outputs are saved to:

```text
results/scale_n_ints/{n_id}.json
```

---

## 🏢 Running on a Cluster

Slurm submission scripts are stored in:

```text
experiments/cluster/
```

For example:

```bash
sbatch experiments/cluster/submit_scale_costs.sbatch
sbatch experiments/cluster/submit_scale_n_farmers.sbatch
sbatch experiments/cluster/submit_scale_n_ints.sbatch
```

Each array task corresponds to one random seed.

---

## 🔬 Analysis Notebooks

The main notebooks are:

- `notebooks/eda.ipynb`: exploratory analysis of empirical pickup data.
- `notebooks/ig_dev.ipynb`: validation of the instance generator against empirical marginal distributions.
- `notebooks/figure_scale_cost.ipynb`: analysis and figures for the cost-scaling experiment.
- `notebooks/figure_scale_n_farmers.ipynb`: analysis and figures for the farmer-scaling experiment.
- `notebooks/figure_scale_n_ints.ipynb`: analysis and figures for the intermediary-scaling experiment.
- `notebooks/figure_matching.ipynb`: matching-related figures.
- `notebooks/figure_prices.ipynb`: price-related figures.
- `notebooks/figure_welfare.ipynb`: welfare-related figures.

---

## 🏭 Output Format

Experiment scripts write one JSON file per seed. Each JSON file contains a list of result dictionaries.

Common fields include:

- `n_id`: random seed / instance ID.
- `n_ints`: number of intermediaries.
- `scale_factor`: farmer-count scale factor, for farmer-scaling experiments.
- `multiplier`: transportation cost multiplier, for cost-scaling experiments.
- `epsilon`: stability/slack parameter by intermediary.
- `cost`: heterogeneous intermediary costs.
- `farmer_quantities`: generated farmer quantities.
- `total_quantity`: total generated fruit quantity.
- `total_fruit_value`: total generated fruit value.
- `summary_vanilla`: solver output, including profit, welfare, matching cost, runtime, oracle calls, and final solutions.

---

## 📈 Generated Figures

Generated figures are saved in:

```text
figures/
```

Current figures include:

- platform profit
- farmer welfare
- intermediary welfare
- intermediary profit
- farmer payments
- matching heatmaps
- price boxplots
- scaling experiment figures

---

## 🗒️ Remarks

- The number-of-intermediaries experiment changes the _overall _generated market size; it does not hold farmers fixed while adding intermediaries.
- The farmer-scaling experiment changes the generated farmer network through the instance generator.
- Welfare metrics depend on the selected tie-break among profit-optimal solutions.
- Confidence intervals in the scaling notebooks are pointwise confidence intervals for the mean paired change. They are not prediction intervals and do not show the full heterogeneity across seeds.
