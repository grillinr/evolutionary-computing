use crate::bitstring::GAParameters;
use crate::evol_strat::ESParameters;
use crate::fitness::Fitness;
use crate::parameter_tuning::{TuningResult, TuningConfig};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub max_fitness: f64,
    pub generations: usize,
    pub converged: bool,
    pub timeout_reached: bool,
    pub execution_time: f64,
}

pub struct TimeoutRunner;

impl TimeoutRunner {
    pub fn run_sga_with_timeout<F: Fitness + Send + Sync + 'static>(
        fitness_fn: F,
        params: GAParameters,
        config: &TuningConfig,
        run_id: usize,
    ) -> TuningResult {
        let fitness_fn = Arc::new(fitness_fn);
        let timeout_duration = Duration::from_secs(config.timeout_seconds);
        let params_clone = params.clone();
        let params_for_result = params.clone();
        let num_dimensions = config.num_dimensions;
        
        let handle = thread::spawn(move || {
            let start_time = Instant::now();
            let mut rng = ChaCha8Rng::seed_from_u64(5000 + run_id as u64);
            
            // Run SGA with timeout checking
            let mut current_gen = 0;
            let mut max_fitness = 0.0;
            let mut converged = false;
            
            // Initialize population
            let mut population = Self::init_population(&params_clone, &mut rng);
            
            while current_gen < params.max_iters {
                // Check timeout
                if start_time.elapsed() >= timeout_duration {
                    break;
                }
                
                // Calculate fitnesses
                let fitnesses: Vec<f64> = population
                    .iter()
                    .map(|m| fitness_fn.fitness_bitstring(m, num_dimensions))
                    .collect();
                
                max_fitness = fitnesses.iter().cloned().fold(0.0, f64::max);
                let avg_fitness = fitnesses.iter().sum::<f64>() / population.len() as f64;
                
                // Check convergence
                if avg_fitness >= params.convergence_threshold {
                    converged = true;
                    break;
                }
                
                // Create new generation (simplified version)
                population = Self::create_next_generation(&population, &params_clone, &*fitness_fn, num_dimensions, &mut rng);
                current_gen += 1;
            }
            
            let execution_time = start_time.elapsed().as_secs_f64();
            let timeout_reached = start_time.elapsed() >= timeout_duration;
            
            ExecutionStats {
                max_fitness,
                generations: current_gen,
                converged,
                timeout_reached,
                execution_time,
            }
        });
        
        // Wait for completion or timeout
        let execution_stats = match handle.join() {
            Ok(stats) => stats,
            Err(_) => ExecutionStats {
                max_fitness: 0.0,
                generations: 0,
                converged: false,
                timeout_reached: true,
                execution_time: config.timeout_seconds as f64,
            },
        };
        
        let score = if execution_stats.execution_time > 0.0 {
            execution_stats.max_fitness / execution_stats.execution_time
        } else {
            0.0
        };
        
        TuningResult {
            algorithm: "SGA".to_string(),
            parameters: crate::parameter_tuning::ParameterGrid::params_to_map_ga(&params_for_result),
            run_id,
            max_fitness: execution_stats.max_fitness,
            execution_time: execution_stats.execution_time,
            score,
            converged: execution_stats.converged,
            generations: execution_stats.generations,
            timeout_reached: execution_stats.timeout_reached,
        }
    }
    
    pub fn run_es_with_timeout<F: Fitness + Send + Sync + 'static>(
        fitness_fn: F,
        params: ESParameters,
        config: &TuningConfig,
        run_id: usize,
    ) -> TuningResult {
        let fitness_fn = Arc::new(fitness_fn);
        let timeout_duration = Duration::from_secs(config.timeout_seconds);
        
        let params_clone = params.clone();
        let params_for_result = params.clone();
        let handle = thread::spawn(move || {
            let start_time = Instant::now();
            let mut rng = ChaCha8Rng::seed_from_u64(5000 + run_id as u64);
            
            // Run ES with timeout checking
            let mut current_gen = 0;
            let mut max_fitness = 0.0;
            let mut converged = false;
            
            // Initialize population
            let mut population = Self::init_es_population(&params_clone, &mut rng);
            
            while current_gen < params.max_gens {
                // Check timeout
                if start_time.elapsed() >= timeout_duration {
                    break;
                }
                
                // Evaluate fitness
                let fitnesses: Vec<f64> = population
                    .iter()
                    .map(|member| fitness_fn.fitness(&member[0..params.mem_size]))
                    .collect();
                
                max_fitness = fitnesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let avg_fitness = fitnesses.iter().sum::<f64>() / params.mu as f64;
                
                // Check convergence
                if avg_fitness > 0.99 {
                    converged = true;
                    break;
                }
                
                // Create offspring (simplified version)
                population = Self::create_es_offspring(&population, &params_clone, &*fitness_fn, &mut rng);
                current_gen += 1;
            }
            
            let execution_time = start_time.elapsed().as_secs_f64();
            let timeout_reached = start_time.elapsed() >= timeout_duration;
            
            ExecutionStats {
                max_fitness,
                generations: current_gen,
                converged,
                timeout_reached,
                execution_time,
            }
        });
        
        // Wait for completion or timeout
        let execution_stats = match handle.join() {
            Ok(stats) => stats,
            Err(_) => ExecutionStats {
                max_fitness: 0.0,
                generations: 0,
                converged: false,
                timeout_reached: true,
                execution_time: config.timeout_seconds as f64,
            },
        };
        
        let score = if execution_stats.execution_time > 0.0 {
            execution_stats.max_fitness / execution_stats.execution_time
        } else {
            0.0
        };
        
        TuningResult {
            algorithm: "ES".to_string(),
            parameters: crate::parameter_tuning::ParameterGrid::params_to_map_es(&params_for_result),
            run_id,
            max_fitness: execution_stats.max_fitness,
            execution_time: execution_stats.execution_time,
            score,
            converged: execution_stats.converged,
            generations: execution_stats.generations,
            timeout_reached: execution_stats.timeout_reached,
        }
    }
    
    // Helper functions for simplified algorithm execution
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
    
    fn create_next_generation(
        population: &[String],
        params: &GAParameters,
        fitness_fn: &impl Fitness,
        num_dims: usize,
        rng: &mut ChaCha8Rng,
    ) -> Vec<String> {
        let mut new_population = Vec::new();
        
        while new_population.len() < params.pop_size {
            // Tournament selection
            let parent1 = Self::tournament_selection(population, fitness_fn, num_dims, rng);
            let parent2 = Self::tournament_selection(population, fitness_fn, num_dims, rng);
            
            // Crossover
            let (mut child1, mut child2) = Self::crossover(&parent1, &parent2, params.crossover_rate, rng);
            
            // Mutation
            child1 = Self::mutate(&child1, params.mutation_rate, rng);
            child2 = Self::mutate(&child2, params.mutation_rate, rng);
            
            new_population.push(child1);
            if new_population.len() < params.pop_size {
                new_population.push(child2);
            }
        }
        
        new_population
    }
    
    fn tournament_selection(
        population: &[String],
        fitness_fn: &impl Fitness,
        num_dims: usize,
        rng: &mut ChaCha8Rng,
    ) -> String {
        let tournament_size = 3;
        let mut best_individual = String::new();
        let mut best_fitness = f64::MIN;
        
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
    
    fn crossover(parent1: &str, parent2: &str, crossover_rate: f64, rng: &mut ChaCha8Rng) -> (String, String) {
        let random: f64 = rng.random();
        if random >= crossover_rate {
            return (parent1.to_string(), parent2.to_string());
        }
        
        let crossover_point = rng.random_range(1..parent1.len());
        let offspring1 = format!("{}{}", &parent1[..crossover_point], &parent2[crossover_point..]);
        let offspring2 = format!("{}{}", &parent2[..crossover_point], &parent1[crossover_point..]);
        
        (offspring1, offspring2)
    }
    
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
    
    fn init_es_population(params: &ESParameters, rng: &mut ChaCha8Rng) -> Vec<Vec<f64>> {
        let mut population = Vec::new();
        for _ in 0..params.mu {
            let mut member: Vec<f64> = Vec::new();
            for _ in 0..params.mem_size {
                let gene: f64 = rng.random_range(params.mem_range.0..params.mem_range.1);
                member.push(gene);
            }
            member.push(params.sigma);
            population.push(member);
        }
        population
    }
    
    fn create_es_offspring(
        population: &[Vec<f64>],
        params: &ESParameters,
        fitness_fn: &impl Fitness,
        rng: &mut ChaCha8Rng,
    ) -> Vec<Vec<f64>> {
        // Evaluate current population
        let fitnesses: Vec<f64> = population
            .iter()
            .map(|member| fitness_fn.fitness(&member[0..params.mem_size]))
            .collect();
        
        // Create lambda offspring
        let mut offspring = Vec::new();
        for _ in 0..params.lambda {
            // Tournament selection
            let parent_idx = (0..params.mu)
                .choose_multiple(rng, 2)
                .into_iter()
                .max_by(|&i, &j| fitnesses[i].partial_cmp(&fitnesses[j]).unwrap())
                .unwrap();
            let parent = &population[parent_idx];
            
            // Mutate
            let mut child = Vec::new();
            let genes = &parent[0..params.mem_size];
            let sigma_val = parent[params.mem_size];
            
            for &gene in genes {
                let mutation: f64 = rng.sample::<f64, _>(rand_distr::Normal::new(0.0, sigma_val).unwrap());
                let mutated_gene = gene + mutation;
                child.push(mutated_gene);
            }
            
            // Mutate sigma
            let sigma_mutation: f64 = rng.sample::<f64, _>(rand_distr::Normal::new(0.0, 1.0).unwrap());
            let new_sigma = sigma_val * (params.tau * sigma_mutation).exp();
            child.push(new_sigma);
            
            offspring.push(child);
        }
        
        // Select best mu from lambda offspring
        let offspring_fitnesses: Vec<f64> = offspring
            .iter()
            .map(|member| fitness_fn.fitness(&member[0..params.mem_size]))
            .collect();
        
        let mut indexed: Vec<(f64, usize)> = offspring_fitnesses
            .iter()
            .enumerate()
            .map(|(i, &f)| (f, i))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        
        let selected_indices: Vec<usize> = indexed.into_iter().take(params.mu).map(|(_, i)| i).collect();
        let mut new_population = Vec::new();
        for &idx in &selected_indices {
            new_population.push(offspring[idx].clone());
        }
        
        new_population
    }
}