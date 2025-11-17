use crate::bitstring::GAParameters;
use crate::evol_strat::ESParameters;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningResult {
    pub algorithm: String,
    pub parameters: HashMap<String, f64>,
    pub run_id: usize,
    pub max_fitness: f64,
    pub execution_time: f64,
    pub score: f64,
    pub converged: bool,
    pub generations: usize,
    pub timeout_reached: bool,
}

pub struct ParameterGrid;

impl ParameterGrid {
    pub fn generate_sga_grid() -> Vec<GAParameters> {
        let mutation_rates = vec![0.001, 0.05, 0.1, 0.15, 0.2];
        let population_sizes = vec![50, 162, 275, 387, 500];

        let mut grid = Vec::new();

        for &pop_size in &population_sizes {
            for &mutation_rate in &mutation_rates {
                grid.push(GAParameters {
                    pop_size,
                    mem_size: 16 * 10, // 16 bits per dimension * 10 dimensions
                    mutation_rate,
                    crossover_rate: 0.75,
                    max_iters: 1000,
                    convergence_threshold: 0.95,
                });
            }
        }

        grid
    }

    pub fn generate_es_grid() -> Vec<ESParameters> {
        let lambda_values = vec![50, 162, 275, 387, 500];
        let sigma_values = vec![0.1, 0.575, 1.05, 1.525, 2.0];

        let mut grid = Vec::new();

        for &lambda in &lambda_values {
            for &sigma in &sigma_values {
                grid.push(ESParameters {
                    mu: lambda / 2,
                    lambda,
                    mem_size: 10, // 10 dimensions
                    mem_range: (-5.12, 5.11),
                    sigma,
                    tau: 1.0 / (2.0 * 10.0_f64).sqrt(),
                    max_gens: 1000,
                });
            }
        }

        grid
    }

    pub fn params_to_map_ga(params: &GAParameters) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("pop_size".to_string(), params.pop_size as f64);
        map.insert("mem_size".to_string(), params.mem_size as f64);
        map.insert("mutation_rate".to_string(), params.mutation_rate);
        map.insert("crossover_rate".to_string(), params.crossover_rate);
        map.insert("max_iters".to_string(), params.max_iters as f64);
        map.insert(
            "convergence_threshold".to_string(),
            params.convergence_threshold,
        );
        map
    }

    pub fn params_to_map_es(params: &ESParameters) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("mu".to_string(), params.mu as f64);
        map.insert("lambda".to_string(), params.lambda as f64);
        map.insert("mem_size".to_string(), params.mem_size as f64);
        map.insert("mem_range_min".to_string(), params.mem_range.0);
        map.insert("mem_range_max".to_string(), params.mem_range.1);
        map.insert("sigma".to_string(), params.sigma);
        map.insert("tau".to_string(), params.tau);
        map.insert("max_gens".to_string(), params.max_gens as f64);
        map
    }
}

pub struct TuningConfig {
    pub num_runs: usize,
    pub timeout_seconds: u64,
    pub num_dimensions: usize,
    pub bits_per_dimension: usize,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            num_runs: 5,
            timeout_seconds: 60,
            num_dimensions: 10,
            bits_per_dimension: 16,
        }
    }
}

