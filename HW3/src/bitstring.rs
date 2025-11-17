use crate::fitness::Fitness;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

pub struct GAParameters {
    pub pop_size: usize,
    pub mem_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub max_iters: usize,
    pub convergence_threshold: f64,
}

// Creates a population of random bitstrings with specified size and member length
fn init_population(params: &GAParameters, rng: &mut ChaCha8Rng) -> Vec<String> {
    let mut population = Vec::new();
    for _ in 0..params.pop_size {
        let mut member: String = String::new();
        for _ in 0..params.mem_size {
            let bit = if rng.random() { '1' } else { '0' };
            member.push(bit);
        }
        population.push(member);
    }
    population
}

// Given a bitstring, flips each bit with a probability equal to mutation_rate
fn mutate(bitstring: &str, mutation_rate: f64, rng: &mut ChaCha8Rng) -> String {
    let mut mutated = String::new();
    for c in bitstring.chars() {
        let random: f64 = rng.random();
        let bit = if random < mutation_rate {
            if c == '1' { '0' } else { '1' }
        } else {
            c
        };
        mutated.push(bit);
    }
    mutated
}

// Perform single point crossover on parents. Becuase we are storing these as strings, we
// can use string formmatting to do this in a straightforward manner.
fn crossover(
    parent1: &str,
    parent2: &str,
    crossover_rate: f64,
    rng: &mut ChaCha8Rng,
) -> (String, String) {
    // Basic error checking for parent lengths
    if parent1.len() != parent2.len() {
        panic!("Parents must be of the same length");
    }

    // Generate a random number, if under crossover rate, perform crossover
    let random: f64 = rng.random();
    if random >= crossover_rate {
        return (parent1.to_string(), parent2.to_string());
    }

    // Pick a random crossover point within parent1
    let crossover_point = rng.random_range(1..parent1.len());

    // Slice strings at crossover point and rejoin with format
    let offspring1 = format!(
        "{}{}",
        &parent1[..crossover_point],
        &parent2[crossover_point..]
    );

    // Slice strings at crossover point and rejoin with format
    let offspring2 = format!(
        "{}{}",
        &parent2[..crossover_point],
        &parent1[crossover_point..]
    );
    (offspring1, offspring2)
}

// Tournament selection
fn tournament_selection(
    population: &Vec<String>,
    num_dims: usize,
    fitness_fn: &impl Fitness,
    tournament_size: usize,
    rng: &mut ChaCha8Rng,
) -> String {
    let mut best_individual = String::new();
    let mut best_fitness = f64::MIN;

    // Randomly select tournament_size individuals and pick the best one
    for _ in 0..tournament_size {
        let random_index = rng.random_range(0..population.len());
        let individual = &population[random_index];
        let fitness = fitness_fn.fitness_bitstring(individual, num_dims);

        if fitness > best_fitness {
            best_fitness = fitness;
            best_individual = individual.clone();
        }
    }

    best_individual
}

// Tournament selection for two parents
fn parent_selection(
    population: &Vec<String>,
    num_dims: usize,
    fitness_fn: &impl Fitness,
    rng: &mut ChaCha8Rng,
) -> (String, String) {
    let tournament_size = 3; // Common tournament size, can be adjusted

    let parent1 = tournament_selection(population, num_dims, fitness_fn, tournament_size, rng);
    let parent2 = tournament_selection(population, num_dims, fitness_fn, tournament_size, rng);

    (parent1, parent2)
}

// Calculate population statistics
fn calculate_stats(
    population: &[String],
    fitness_fn: &impl Fitness,
    num_dims: usize,
) -> (f64, f64, f64, f64) {
    let fitnesses: Vec<f64> = population
        .iter()
        .map(|m| fitness_fn.fitness_bitstring(m, num_dims))
        .collect();

    let max_fitness = fitnesses.iter().cloned().fold(0.0, f64::max);
    let avg_fitness = fitnesses.iter().sum::<f64>() / population.len() as f64;

    // Calculate percentage of identical individuals
    let mut unique_count = 0;
    for (i, member1) in population.iter().enumerate() {
        let mut is_unique = true;
        for (j, member2) in population.iter().enumerate() {
            if i != j && member1 == member2 {
                is_unique = false;
                break;
            }
        }
        if is_unique {
            unique_count += 1;
        }
    }
    let pct_identical = (population.len() - unique_count) as f64 / population.len() as f64;

    // Calculate diversity as max euclidean distance in decoded space
    let mut diversity = 0.0;
    for i in 0..population.len() {
        for j in (i + 1)..population.len() {
            let decoded1 = fitness_fn.decode_bitstring(&population[i], num_dims);
            let decoded2 = fitness_fn.decode_bitstring(&population[j], num_dims);
            let mut dist_sq = 0.0;
            for k in 0..num_dims {
                dist_sq += (decoded1[k] - decoded2[k]).powi(2);
            }
            let dist = dist_sq.sqrt();
            if dist > diversity {
                diversity = dist;
            }
        }
    }

    (max_fitness, avg_fitness, pct_identical, diversity)
}

// Check for convergence (pct identical individuals exceeds threshold or avg fitness exceeds threshold)
fn check_convergence(
    population: &[String],
    fitness_fn: &impl Fitness,
    num_dims: usize,
    threshold: f64,
) -> bool {
    let (_, avg_fitness, pct_identical, _) = calculate_stats(population, fitness_fn, num_dims);
    if avg_fitness >= threshold {
        return true;
    }
    pct_identical >= threshold
}

pub fn sga(
    fitness_fn: &impl Fitness,
    params: &GAParameters,
    rng: &mut ChaCha8Rng,
) -> Vec<String> {
    // Initialize population
    let mut population = init_population(params, rng);
    let mut cumulative_evals = 0;

    // Print algorithm parameters
    println!(
        "Running Dejong Rosenbrock GA with Pop={} MemberSize={} Mutation={} Crossover={}",
        params.pop_size, params.mem_size, params.mutation_rate, params.crossover_rate
    );

    for gen_number in 0..params.max_iters {
        // Calculate and print statistics
        let (max_fitness, avg_fitness, _, diversity) =
            calculate_stats(&population, fitness_fn, params.mem_size / 2);
        cumulative_evals += params.pop_size;
        println!(
            "Dejong Rosenbrock GA {} {} {} {} {} {} {} {} {}",
            params.pop_size, params.pop_size, params.mutation_rate, params.crossover_rate, gen_number, cumulative_evals, max_fitness, avg_fitness, diversity
        );

        // Check for convergence
        if check_convergence(&population, fitness_fn, params.mem_size / 2, params.convergence_threshold) {
            println!("Converged at generation {gen_number}");
            return population;
        }

        // Create new generation
        let mut new_population = Vec::new();

        // Generate offspring pairs until we have a full new population
        while new_population.len() < params.pop_size {
            // Select parents
            let (parent1, parent2) = parent_selection(&population, params.mem_size / 2, fitness_fn, rng);

            // Crossover
            let (mut child1, mut child2) = crossover(&parent1, &parent2, params.crossover_rate, rng);

            // Mutation
            child1 = mutate(&child1, params.mutation_rate, rng);
            child2 = mutate(&child2, params.mutation_rate, rng);

            // Add children to new population
            new_population.push(child1);
            if new_population.len() < params.pop_size {
                new_population.push(child2);
            }
        }

        // Ensure we have exactly pop_size individuals (handle odd pop_size case)
        while new_population.len() > params.pop_size {
            new_population.pop();
        }

        // Full replacement: new population replaces old population
        population = new_population;
    }
    println!("Max iterations reached");
    population
}
