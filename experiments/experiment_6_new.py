import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
from pprint import pprint

sys.path.insert(0, "src")

from farmers_intermediaries import Instance
from road_graphs import RoadGraph
from pricing import Optimizer
from instance_generator import InstanceGenerator


# =========================
# CLI ARGUMENTS
# =========================
if len(sys.argv) != 2:
    raise ValueError("Please provide n_id as a single argument.")

n_id = int(sys.argv[1])

# =========================
# CONFIG
# =========================
TEXTWIDTH = 80
SIM_SIZE = 10
N_INTS = 12
EPSILONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

GRAPH_PATH = "data/graph_0-14960_00_new.pickle"
FARMERS_PATH = "data/farmers.csv"
FARMERS_2_PATH = "data/farmers_2.csv"
INTS_PATH = "data/ints.csv"

# =========================
# CORE BUILDERS
# =========================


def run_single_simulation(platform, graph, sampled_epsilon):
    platform.set_graph(RoadGraph(graph))

    epsilon = {int.id: sampled_epsilon for int in platform.intermediaries}
    het_costs = {intermediary.id: (platform.dist_to_mill[intermediary.id] * 4) for intermediary in platform.intermediaries}

    parameters = {
        "epsilon": epsilon,
        "solver": "gurobi",
        "het_costs": het_costs,
    }

    opt = Optimizer(platform, parameters)
    base_matchings = opt.base_matchings

    summary_vanilla = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": False,
        "domination": False,})
    
    opt = Optimizer(platform, parameters, base_matchings=base_matchings)
    
    summary_structured = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": True,
        "domination": False,})
    
    opt = Optimizer(platform, parameters, base_matchings=base_matchings)
    
    summary_domination = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": False,
        "domination": True,})

    farmer_quantities = {f.id: f.quantity for f in platform.farmers}

    print(" Profits ".center(TEXTWIDTH, "-"))
    print(f"Vanilla: {summary_vanilla.max_int_welf_sol.profit}")
    print(f"Structured: {summary_structured.max_int_welf_sol.profit}")
    print(f"Domination: {summary_domination.max_int_welf_sol.profit}")
    print(f"Vanilla Profit %: {summary_vanilla.max_int_welf_sol.profit / (np.sum(list(farmer_quantities.values())) * platform.fruit_price) * 100}")
    print()

    return {
        "epsilon": epsilon,
        "cost": het_costs,
        "farmer_quantities": farmer_quantities,
        "summary_vanilla": summary_vanilla.to_dict(),
        "summary_structured": summary_structured.to_dict(),
        "summary_domination": summary_domination.to_dict(),
        "farmer_dirt_to_mill": {f.id: f.dirt_to_mill for f in platform.farmers},
        "farmer_paved_to_mill": {f.id: f.paved_to_mill for f in platform.farmers},
    }



# =========================
# JSON SERIALIZATION
# =========================
def convert(obj):
    if isinstance(obj, (np.float64, np.int64)):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def make_int_seed(seed_seq: np.random.SeedSequence) -> int:
    """Convert a SeedSequence into a plain Python int for functions that expect int seeds."""
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


# =========================
# MAIN EXPERIMENT LOOP
# =========================
def main():
    # load static data
    farmers_df = pd.read_csv(FARMERS_PATH)
    farmers_2_df = pd.read_csv(FARMERS_2_PATH)
    ints_df = pd.read_csv(INTS_PATH)
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)

    # initialize generator and platform
    instance_generator = InstanceGenerator(farmers_df, farmers_2_df, ints_df, GRAPH_PATH)
    sampled_epsilon = EPSILONS[n_id % len(EPSILONS)] # cycle through epsilon

    results = []

    root_seed = n_id
    root_ss = np.random.SeedSequence(root_seed)

    sim_seed_sequences = root_ss.spawn(SIM_SIZE)
    
    for sim_n, sim_ss in enumerate(sim_seed_sequences):
        instance_id = f'{n_id}_{sim_n}'

        msg = f" Experiment {sim_n}: n_ints = {N_INTS}, instance_seed = {n_id}, epsilon = {sampled_epsilon} "
        print(msg.center(TEXTWIDTH, "="))
        print()

        gen_ints_ss, gen_instance_ss = sim_ss.spawn(2)

        gen_ints_seed = make_int_seed(gen_ints_ss)
        gen_instance_seed = make_int_seed(gen_instance_ss)

        msg = (
            f" Experiment {sim_n}: n_ints = {N_INTS}, "
            f"root_seed = {root_seed}, "
            f"gen_ints_seed = {gen_ints_seed}, "
            f"gen_instance_seed = {gen_instance_seed}"
        )
        print(msg.center(TEXTWIDTH, "="))
        print()
        
        instance_generator.gen_ints(N_INTS, gen_ints_seed)

        instance_dict = instance_generator.gen_instance(
            instance_id,
            write=False,
            plot=False,
            seed=gen_instance_seed
        )

        platform = Instance.from_dict(instance_dict)
        platform.set_graph(RoadGraph(graph))

        msg = f" Instance Details ".center(TEXTWIDTH, "-")
        print(msg)
        print("Intermediaries:")
        pprint(platform.intermediaries)
        print()
        print("Farmers:")
        pprint(platform.farmers)
        print()


        # run one stochastic solve
        sim_result = run_single_simulation(platform, graph, sampled_epsilon)

        # add metadata
        sim_result.update({
            "instance_id": instance_id,
            "root_seed": root_seed,
            "sim_n": sim_n,
            "gen_ints_seed": gen_ints_seed,
            "gen_instance_seed": gen_instance_seed,
        })

        results.append(sim_result)

    # save results
    results_path = Path(f"results/exp_6_new/{n_id}.json")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, default=convert)

    print(f"Results saved to {results_path}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()