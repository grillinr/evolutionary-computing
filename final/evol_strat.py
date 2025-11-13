import random
import math
from typing import List, Tuple

def init_population(pop_size: int, mem_size: int, mem_range: Tuple[float, float], sigma: float, rng: random.Random) -> List[List[float]]:
    population = []
    for _ in range(pop_size):
        member = []
        for _ in range(mem_size):
            gene = rng.uniform(mem_range[0], mem_range[1])
            member.append(gene)
        member.append(sigma)
        population.append(member)
    return population

def evolution_strategy(fitness_fn, mu: int, lambda_: int, mem_size: int, mem_range: Tuple[float, float], sigma: float, tau: float, max_gens: int, rng: random.Random) -> List[List[float]]:
    population = init_population(mu, mem_size, mem_range, sigma, rng)
    cumulative_evals = 0

    for generation_number in range(1, max_gens + 1):
        fitnesses = [fitness_fn.fitness(member[:mem_size]) for member in population]
        cumulative_evals += mu

        offspring = []
        for _ in range(lambda_):
            # Select parent using tournament
            candidates = rng.sample(range(mu), 2)
            parent_idx = max(candidates, key=lambda i: fitnesses[i])
            parent = population[parent_idx]

            child = []
            genes = parent[:mem_size]
            sigma_val = parent[mem_size]
            for gene in genes:
                mutation = rng.gauss(0.0, sigma_val)
                mutated_gene = gene + mutation
                child.append(mutated_gene)
            # Mutate sigma
            sigma_mutation = rng.gauss(0.0, 1.0)
            new_sigma = sigma_val * math.exp(tau * sigma_mutation)
            child.append(new_sigma)
            offspring.append(child)

        offspring_fitnesses = [fitness_fn.fitness(member[:mem_size]) for member in offspring]
        cumulative_evals += lambda_

        max_fitness = max(fitnesses)
        average = sum(fitnesses) / mu
        diversity = 0.0
        for i in range(mu):
            for j in range(i + 1, mu):
                dist = math.sqrt(sum((population[i][k] - population[j][k])**2 for k in range(mem_size)))
                if dist > diversity:
                    diversity = dist
        print(f"Himmelblau ES {mu} {lambda_} {tau} 0.0 {generation_number} {cumulative_evals} {max_fitness} {average} {diversity}")

        if average > 0.99:
            break

        # Select best mu from offspring
        indexed = [(f, i) for i, f in enumerate(offspring_fitnesses)]
        indexed.sort(key=lambda x: x[0], reverse=True)
        selected_indices = [i for _, i in indexed[:mu]]

        new_population = [offspring[idx] for idx in selected_indices]
        population = new_population

    return population