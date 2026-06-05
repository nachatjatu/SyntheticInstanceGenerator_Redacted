import sys
import json
import numpy as np
from pathlib import Path
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
N_INTS = 12
SCALE_FACTORS = [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
HET_COST_PER_METER = 4
EPSILON = 2

# =========================
# JSON SERIALIZATION
# =========================
def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.str_):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

# =========================
# CORE BUILDERS
# =========================

def run_single_simulation(platform, graph):

    platform.set_graph(RoadGraph(graph))

    epsilon = {intermediary.id: EPSILON for intermediary in platform.intermediaries}
    het_costs = {intermediary.id: (platform.dist_to_mill[intermediary.id] * HET_COST_PER_METER) for intermediary in platform.intermediaries}

    parameters = {
        "epsilon": epsilon,
        "solver": "gurobi",
        "het_costs": het_costs,
    }

    opt = Optimizer(platform, parameters)

    summary_vanilla = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": False,
        "domination": False,})

    farmer_quantities = {f.id: f.quantity for f in platform.farmers}

    total_quantity = float(np.sum(list(farmer_quantities.values())))
    total_fruit_value = total_quantity * platform.fruit_price
    profit = summary_vanilla.max_int_welf_sol.profit
    profit_pct = profit / total_fruit_value * 100 if total_fruit_value > 0 else np.nan


    print(" Profits ".center(TEXTWIDTH, "-"))
    print(f"Vanilla: {summary_vanilla.max_int_welf_sol.profit}")
    print(f"Total Fruit Value: {total_fruit_value}")
    print(f"Vanilla Profit %: {profit_pct}")
    print()

    return {
        "epsilon": epsilon,
        "cost": het_costs,
        "farmer_quantities": farmer_quantities,
        "summary_vanilla": summary_vanilla.to_dict(),
        "farmer_dirt_to_mill": {f.id: f.dirt_to_mill for f in platform.farmers},
        "farmer_paved_to_mill": {f.id: f.paved_to_mill for f in platform.farmers},
        "total_quantity": total_quantity,
        "total_fruit_value": total_fruit_value,
        "vanilla_profit": profit,
        "vanilla_profit_pct": profit_pct,
    }


# =========================
# MAIN EXPERIMENT LOOP
# =========================
def main():
    # initialize generator and platform
    instance_generator = InstanceGenerator()

    results = []

    for scale_factor in SCALE_FACTORS:
        instance_id = f'{n_id}_{scale_factor}'

        msg = f" Experiment {n_id}: n_ints = {N_INTS}, instance_seed = {n_id}, scale_factor = {scale_factor} "
        print(msg.center(TEXTWIDTH, "="))
        print()

        instance_generator.gen_ints(N_INTS, n_id, set_type="medium")

        instance_dict = instance_generator.gen_instance(
            instance_id,
            write=False,
            plot=False,
            seed=n_id,
            scale_factor=scale_factor
        )

        platform = Instance.from_dict(instance_dict)

        msg = f" Instance Details ".center(TEXTWIDTH, "-")
        print(msg)
        print("Intermediaries:")
        pprint(platform.intermediaries)
        print()
        print("Farmers:")
        pprint(platform.farmers)
        print()


        # run one stochastic solve
        sim_result = run_single_simulation(platform, instance_generator.G)

        # add metadata
        sim_result.update({
            "instance_id": instance_id,
            "n_id": n_id,
            "n_ints": N_INTS,
            "scale_factor": scale_factor
        })

        results.append(sim_result)

    # save results
    results_path = Path(f"results/scale_n_farmers/{n_id}.json")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, default=convert)

    print(f"Results saved to {results_path}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()