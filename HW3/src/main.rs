mod bitstring;
mod evol_strat;
mod fitness;
mod rosenbrock;
mod parameter_tuning;
mod timeout_runner;
mod results_analyzer;

use crate::bitstring::{GAParameters, sga};
use crate::evol_strat::{ESParameters, evolution_strategy};
use crate::fitness::Fitness;
use crate::rosenbrock::Rosenbrock;
use crate::parameter_tuning::{ParameterGrid, TuningConfig};
use crate::timeout_runner::TimeoutRunner;
use crate::results_analyzer::ResultsAnalyzer;
use std::env;
use std::time::Instant;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() > 1 && args[1] == "tune" {
        run_parameter_tuning();
    } else {
        run_default();
    }
}

fn run_default() {
    const NUM_DIMS: usize = 10;
    // Seed the random number generator for reproducibility
    let mut rng = ChaCha8Rng::seed_from_u64(5000);

    // Run ES for comparison
    let es_params = ESParameters {
        mu: 100,                                   // population size (mu)
        lambda: 100,                               // offspring size (lambda)
        mem_size: NUM_DIMS,                        // number of dimensions
        mem_range: (-5.12, 5.11),                  // range for initial population
        sigma: 1.0,                                // initial step size (sigma)
        tau: 1.0 / (2.0 * NUM_DIMS as f64).sqrt(), // learning rate (tau)
        max_gens: 1000,                            // max generations
    };
    let final_es_pop = evolution_strategy(&Rosenbrock, &es_params, &mut rng);

    // Print ES results
    println!("\n=== ES Results ===");
    for member in final_es_pop.iter().take(3) {
        let fitness = Rosenbrock.fitness(&member[0..NUM_DIMS]);
        let x = &member[0..NUM_DIMS];
        print!("ES Member: [");
        for val in x.iter().take(3) {
            print!("{val:.4}, ");
        }
        println!("...] Fitness: {fitness}");
    }

    // Reset RNG for GA
    let mut rng = ChaCha8Rng::seed_from_u64(5000);
    let ga_params = GAParameters {
        pop_size: 100,               //population size (mu = lambda)
        mem_size: 16 * NUM_DIMS,     // member size (in bits)
        mutation_rate: 0.01,         // mutation rate
        crossover_rate: 0.75,        // crossover rate
        max_iters: 1000,             // max evaluations
        convergence_threshold: 0.95, // convergence threshold
    };
    let final_ea_pop = sga(&Rosenbrock, &ga_params, &mut rng);

    // Print final populations and their fitnesses
    for member in final_ea_pop {
        let fitness = Rosenbrock.fitness_bitstring(&member, NUM_DIMS);
        let x = Rosenbrock.decode_bitstring(&member, NUM_DIMS);
        print!("GA Member: [");
        for val in x {
            print!("{val:.4}, ");
        }
        println!("] Fitness: {fitness}");
    }

    // Test a few random individuals to see typical values
    println!("\nTesting random individuals:");
    for _ in 0..5 {
        let mut random_bits = String::new();
        for _ in 0..(20 * NUM_DIMS) {
            random_bits.push(if rng.random() { '1' } else { '0' });
        }
        let x = Rosenbrock.decode_bitstring(&random_bits, NUM_DIMS);
        let fitness = Rosenbrock.fitness_bitstring(&random_bits, NUM_DIMS);
        print!("Random: [");
        for val in x.iter().take(3) {
            print!("{val:.4}, ");
        }
        println!("...] Fitness: {fitness:e}");
    }
}

fn run_parameter_tuning() {
    println!("Starting parameter tuning...");
    let start_time = Instant::now();
    
    let config = TuningConfig::default();
    let mut all_results = Vec::new();
    
    // Generate parameter grids
    let sga_grid = ParameterGrid::generate_sga_grid();
    let es_grid = ParameterGrid::generate_es_grid();
    
    println!("Generated {} SGA parameter combinations", sga_grid.len());
    println!("Generated {} ES parameter combinations", es_grid.len());
    println!("Each will be tested {} times with {} second timeout", config.num_runs, config.timeout_seconds);
    
    // Test SGA parameters
    println!("\n=== Testing SGA Parameters ===");
    for (i, params) in sga_grid.iter().enumerate() {
        println!("Testing SGA combination {}/{}: pop_size={}, mutation_rate={:.3}", 
            i + 1, sga_grid.len(), params.pop_size, params.mutation_rate);
        
        for run in 0..config.num_runs {
            let result = TimeoutRunner::run_sga_with_timeout(
                Rosenbrock,
                params.clone(),
                &config,
                run,
            );
            all_results.push(result);
        }
    }
    
    // Test ES parameters
    println!("\n=== Testing ES Parameters ===");
    for (i, params) in es_grid.iter().enumerate() {
        println!("Testing ES combination {}/{}: lambda={}, sigma={:.3}", 
            i + 1, es_grid.len(), params.lambda, params.sigma);
        
        for run in 0..config.num_runs {
            let result = TimeoutRunner::run_es_with_timeout(
                Rosenbrock,
                params.clone(),
                &config,
                run,
            );
            all_results.push(result);
        }
    }
    
    // Save results to files
    println!("\n=== Saving Results ===");
    if let Err(e) = ResultsAnalyzer::save_results_to_csv(&all_results, "tuning_results.csv") {
        eprintln!("Error saving results: {}", e);
    } else {
        println!("Results saved to tuning_results.csv");
    }
    
    // Analyze results
    let sga_analysis = ResultsAnalyzer::analyze_sga_results(&all_results);
    let es_analysis = ResultsAnalyzer::analyze_es_results(&all_results);
    
    // Print summary
    ResultsAnalyzer::print_summary(&sga_analysis, &es_analysis);
    
    let total_time = start_time.elapsed();
    println!("\nTotal tuning time: {:.2} minutes", total_time.as_secs_f64() / 60.0);
    println!("Total runs completed: {}", all_results.len());
}
