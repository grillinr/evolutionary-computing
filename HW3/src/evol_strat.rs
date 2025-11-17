use crate::fitness::Fitness;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

fn init_population(
    pop_size: usize,
    mem_size: usize,
    mem_range: (f64, f64),
    sigma: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<f64>> {
    let mut population = Vec::new();
    for _ in 0..pop_size {
        let mut member: Vec<f64> = Vec::new();
        for _ in 0..mem_size {
            let gene: f64 = rng.random_range(mem_range.0..mem_range.1);
            member.push(gene);
        }
        member.push(sigma);
        population.push(member);
    }
    population
}

// Ignore warning
#[allow(clippy::too_many_arguments)]
pub fn evolution_strategy<F: Fitness>(
    fitness_fn: &F,
    mu: usize,
    lambda: usize,
    mem_size: usize,
    mem_range: (f64, f64),
    sigma: f64,
    tau: f64,
    max_gens: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<f64>> {
    // Initialize population
    let mut population = init_population(mu, mem_size, mem_range, sigma, rng);
    let mut cumulative_evals = 0;

    for generation_number in 1..=max_gens {
        // Evaluate fitness of current population
        let fitnesses: Vec<f64> = population
            .iter()
            .map(|member| fitness_fn.fitness(&member[0..mem_size]))
            .collect();
        cumulative_evals += mu;

        // Create lambda offspring
        let mut offspring = Vec::new();
        for _ in 0..lambda {
            // Select a parent using tournament selection
            let parent_idx = (0..mu)
                .choose_multiple(rng, 2)
                .into_iter()
                .max_by(|&i, &j| fitnesses[i].partial_cmp(&fitnesses[j]).unwrap())
                .unwrap();
            let parent = &population[parent_idx];

            // Mutate the parent to create an offspring
            let mut child = Vec::new();
            let genes = &parent[0..mem_size];
            let sigma_val = parent[mem_size];
            for &gene in genes {
                let mutation: f64 =
                    rng.sample::<f64, _>(rand_distr::Normal::new(0.0, sigma_val).unwrap());
                let mutated_gene = gene + mutation;
                child.push(mutated_gene);
            }
            // Mutate sigma
            let sigma_mutation: f64 =
                rng.sample::<f64, _>(rand_distr::Normal::new(0.0, 1.0).unwrap());
            let new_sigma = sigma_val * (tau * sigma_mutation).exp();
            child.push(new_sigma);
            offspring.push(child);
        }

        // Evaluate fitness of offspring
        let offspring_fitnesses: Vec<f64> = offspring
            .iter()
            .map(|member| fitness_fn.fitness(&member[0..mem_size]))
            .collect();
        cumulative_evals += lambda;

        let max_fitness = fitnesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f64 = fitnesses.iter().sum();
        let average = sum / mu as f64;
        let mut diversity = 0.0;
        for i in 0..mu {
            for j in (i + 1)..mu {
                let dist = (0..mem_size)
                    .map(|k| (population[i][k] - population[j][k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if dist > diversity {
                    diversity = dist;
                }
            }
        }
        println!(
            "Dejong Rosenbrock ES {mu} {lambda} {tau} 0.0 {generation_number} {cumulative_evals} {max_fitness} {average} {diversity}"
        );

        // Early stopping if average fitness exceeds threshold
        if average > 0.99 {
            break;
        }

        // Select the best mu from lambda offspring
        let mut indexed: Vec<(f64, usize)> = offspring_fitnesses
            .iter()
            .enumerate()
            .map(|(i, &f)| (f, i))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // descending
        let selected_indices: Vec<usize> = indexed.into_iter().take(mu).map(|(_, i)| i).collect();

        let mut new_population = Vec::new();
        for &idx in &selected_indices {
            new_population.push(offspring[idx].clone());
        }

        population = new_population;
    }

    population
}
