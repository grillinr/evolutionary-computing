use crate::fitness::Fitness;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

// Creates a population of random bitstrings with specified size and member length
fn init_population(pop_size: usize, mem_size: usize, rng: &mut ChaCha8Rng) -> Vec<String> {
    let mut population = Vec::new();
    for _ in 0..pop_size {
        let mut member: String = String::new();
        for _ in 0..mem_size {
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

// Roulette wheel selection (fitness proportional)
fn parent_selection(
    population: &Vec<String>,
    fitness_fn: &impl Fitness,
    rng: &mut ChaCha8Rng,
) -> (String, String) {
    // Determine the "wheel size"
    let total_fitness: f64 = population
        .iter()
        .map(|member| fitness_fn.fitness(member))
        .sum();

    // Helper function to select one parent
    fn select_parent(population: &Vec<String>, fitness_fn: &impl Fitness, mut pick: f64) -> String {
        let mut parent = String::new();

        // Iterate through the wheel until we find the slot that matches the pick
        for member in population {
            // Get the fitness of the current member
            let fitness = fitness_fn.fitness(member);

            // If the pick is less than or equal to the fitness parent is found
            if pick <= fitness {
                parent = member.clone();
                break;
            }

            // If parent not found, keep moving down the wheel
            pick -= fitness;
        }
        parent
    }

    // Select first parent on the wheel
    let pick1 = rng.random_range(0.0..total_fitness);
    let parent1 = select_parent(population, fitness_fn, pick1);

    // Select second parent
    let pick2 = rng.random_range(0.0..total_fitness);
    let parent2 = select_parent(population, fitness_fn, pick2);

    (parent1, parent2)
}

// Calculate population statistics
fn calculate_stats(population: &[String], fitness_fn: &impl Fitness) -> (f64, f64, f64) {
    let fitnesses: Vec<f64> = population.iter().map(|m| fitness_fn.fitness(m)).collect();

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

    (max_fitness, avg_fitness, pct_identical)
}

// Check for convergence (pct identical individuals exceeds threshold or avg fitness exceeds threshold)
fn check_convergence(population: &[String], fitness_fn: &impl Fitness, threshold: f64) -> bool {
    let (_, avg_fitness, pct_identical) = calculate_stats(population, fitness_fn);
    if avg_fitness >= threshold {
        return true;
    }
    pct_identical >= threshold
}

// Ignore warning
#[allow(clippy::too_many_arguments)]
pub fn sga(
    fitness_fn: &impl Fitness,
    pop_size: usize,
    mem_size: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    max_iters: usize,
    convergence_threshold: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<String> {
    // Initialize population
    let mut population = init_population(pop_size, mem_size, rng);

    // Print algorithm parameters
    println!("{pop_size} {mem_size} {mutation_rate} {crossover_rate}");

    for gen_number in 0..max_iters {
        // Calculate and print statistics
        let (max_fitness, avg_fitness, pct_identical) = calculate_stats(&population, fitness_fn);
        println!("{gen_number} {max_fitness:.2} {avg_fitness:.2} {pct_identical:.2}");

        // Check for convergence
        if check_convergence(&population, fitness_fn, convergence_threshold) {
            println!("Converged at generation {gen_number}");
            return population;
        }

        // Create new generation
        let mut new_population = Vec::new();

        // Generate offspring pairs until we have a full new population
        while new_population.len() < pop_size {
            // Select parents
            let (parent1, parent2) = parent_selection(&population, fitness_fn, rng);

            // Crossover
            let (mut child1, mut child2) = crossover(&parent1, &parent2, crossover_rate, rng);

            // Mutation
            child1 = mutate(&child1, mutation_rate, rng);
            child2 = mutate(&child2, mutation_rate, rng);

            // Add children to new population
            new_population.push(child1);
            if new_population.len() < pop_size {
                new_population.push(child2);
            }
        }

        // Full replacement: new population replaces old population
        population = new_population;
    }
    println!("Max iterations reached");
    population
}
