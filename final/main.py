from .evol_strat import evolution_strategy
from .himmelblau import Himmelblau
import random

def main():
    rng = random.Random(5000)
    final_es_pop = evolution_strategy(
        Himmelblau(),
        15,  # mu
        100,  # lambda
        2,  # mem_size
        (-10.0, 10.0),  # mem_range
        0.1,  # sigma
        0.1,  # tau
        1000,  # max_gens
        rng
    )

    for member in final_es_pop:
        fitness = Himmelblau().fitness(member[:2])
        print(f"ES Member: {member}, Fitness: {fitness}")

if __name__ == "__main__":
    main()