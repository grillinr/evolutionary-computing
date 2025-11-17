use crate::parameter_tuning::TuningResult;
use crate::bitstring::GAParameters;
use crate::evol_strat::ESParameters;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub struct ResultsAnalyzer;

impl ResultsAnalyzer {
    pub fn save_results_to_csv(results: &[TuningResult], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let path = Path::new(filename);
        let mut file = File::create(path)?;
        
        // Write header
        let header = "algorithm,run_id,max_fitness,execution_time,score,converged,generations,timeout_reached";
        writeln!(file, "{header}")?;
        
        // Write parameter headers (get all unique parameter names)
        let mut all_param_names = Vec::new();
        if !results.is_empty() {
            let first_result = &results[0];
            for key in first_result.parameters.keys() {
                all_param_names.push(key.clone());
            }
            all_param_names.sort();
        }
        
        // Write parameter headers
        for param_name in &all_param_names {
            write!(file, ",{param_name}")?;
        }
        writeln!(file)?;
        
        // Write data rows
        for result in results {
            write!(file, "{},{},{:.6},{:.6},{:.6},{},{},{}",
                result.algorithm,
                result.run_id,
                result.max_fitness,
                result.execution_time,
                result.score,
                result.converged,
                result.generations,
                result.timeout_reached
            )?;
            
            // Write parameter values
            for param_name in &all_param_names {
                let value = result.parameters.get(param_name).unwrap_or(&0.0);
                write!(file, ",{value}")?;
            }
            writeln!(file)?;
        }
        
        Ok(())
    }
    
    pub fn analyze_sga_results(results: &[TuningResult]) -> SGAAnalysis {
        let sga_results: Vec<&TuningResult> = results
            .iter()
            .filter(|r| r.algorithm == "SGA")
            .collect();
        
        let mut best_score = 0.0;
        let mut best_params: Option<GAParameters> = None;
        let mut score_sum = 0.0;
        let mut convergence_count = 0;
        let mut timeout_count = 0;
        
        // Group by parameters for analysis
        let mut param_groups: HashMap<String, Vec<&TuningResult>> = HashMap::new();
        
        for result in &sga_results {
            let param_key = Self::ga_params_to_key(&result.parameters);
            param_groups.entry(param_key).or_default().push(result);
            
            score_sum += result.score;
            if result.converged {
                convergence_count += 1;
            }
            if result.timeout_reached {
                timeout_count += 1;
            }
            
            if result.score > best_score {
                best_score = result.score;
                best_params = Self::key_to_ga_params(&result.parameters);
            }
        }
        
        // Find best average performing parameter set
        let mut best_avg_score = 0.0;
        let mut best_avg_params: Option<GAParameters> = None;
        
        for group_results in param_groups.values() {
            let avg_score: f64 = group_results.iter().map(|r| r.score).sum::<f64>() / group_results.len() as f64;
            if avg_score > best_avg_score {
                best_avg_score = avg_score;
                best_avg_params = Self::key_to_ga_params(&group_results[0].parameters);
            }
        }
        
        SGAAnalysis {
            total_runs: sga_results.len(),
            best_single_run_score: best_score,
            best_single_run_params: best_params,
            best_avg_score,
            best_avg_params,
            avg_score: score_sum / sga_results.len() as f64,
            convergence_rate: convergence_count as f64 / sga_results.len() as f64,
            timeout_rate: timeout_count as f64 / sga_results.len() as f64,
            param_groups: param_groups.len(),
        }
    }
    
    pub fn analyze_es_results(results: &[TuningResult]) -> ESAnalysis {
        let es_results: Vec<&TuningResult> = results
            .iter()
            .filter(|r| r.algorithm == "ES")
            .collect();
        
        let mut best_score = 0.0;
        let mut best_params: Option<ESParameters> = None;
        let mut score_sum = 0.0;
        let mut convergence_count = 0;
        let mut timeout_count = 0;
        
        // Group by parameters for analysis
        let mut param_groups: HashMap<String, Vec<&TuningResult>> = HashMap::new();
        
        for result in &es_results {
            let param_key = Self::es_params_to_key(&result.parameters);
            param_groups.entry(param_key).or_default().push(result);
            
            score_sum += result.score;
            if result.converged {
                convergence_count += 1;
            }
            if result.timeout_reached {
                timeout_count += 1;
            }
            
            if result.score > best_score {
                best_score = result.score;
                best_params = Self::key_to_es_params(&result.parameters);
            }
        }
        
        // Find best average performing parameter set
        let mut best_avg_score = 0.0;
        let mut best_avg_params: Option<ESParameters> = None;
        
        for group_results in param_groups.values() {
            let avg_score: f64 = group_results.iter().map(|r| r.score).sum::<f64>() / group_results.len() as f64;
            if avg_score > best_avg_score {
                best_avg_score = avg_score;
                best_avg_params = Self::key_to_es_params(&group_results[0].parameters);
            }
        }
        
        ESAnalysis {
            total_runs: es_results.len(),
            best_single_run_score: best_score,
            best_single_run_params: best_params,
            best_avg_score,
            best_avg_params,
            avg_score: score_sum / es_results.len() as f64,
            convergence_rate: convergence_count as f64 / es_results.len() as f64,
            timeout_rate: timeout_count as f64 / es_results.len() as f64,
            param_groups: param_groups.len(),
        }
    }
    
    pub fn print_summary(sga_analysis: &SGAAnalysis, es_analysis: &ESAnalysis) {
        println!("\n{}", "=".repeat(60));
        println!("PARAMETER TUNING SUMMARY");
        println!("{}", "=".repeat(60));
        
        println!("\n--- SGA Results ---");
        println!("Total runs: {}", sga_analysis.total_runs);
        println!("Average score: {:.6}", sga_analysis.avg_score);
        println!("Convergence rate: {:.2}%", sga_analysis.convergence_rate * 100.0);
        println!("Timeout rate: {:.2}%", sga_analysis.timeout_rate * 100.0);
        println!("Parameter combinations tested: {}", sga_analysis.param_groups);
        
        if let Some(ref params) = sga_analysis.best_single_run_params {
            println!("\nBest single run parameters:");
            println!("  Population size: {}", params.pop_size);
            println!("  Mutation rate: {:.3}", params.mutation_rate);
            println!("  Score: {:.6}", sga_analysis.best_single_run_score);
        }
        
        if let Some(ref params) = sga_analysis.best_avg_params {
            println!("\nBest average parameters:");
            println!("  Population size: {}", params.pop_size);
            println!("  Mutation rate: {:.3}", params.mutation_rate);
            println!("  Average score: {:.6}", sga_analysis.best_avg_score);
        }
        
        println!("\n--- ES Results ---");
        println!("Total runs: {}", es_analysis.total_runs);
        println!("Average score: {:.6}", es_analysis.avg_score);
        println!("Convergence rate: {:.2}%", es_analysis.convergence_rate * 100.0);
        println!("Timeout rate: {:.2}%", es_analysis.timeout_rate * 100.0);
        println!("Parameter combinations tested: {}", es_analysis.param_groups);
        
        if let Some(ref params) = es_analysis.best_single_run_params {
            println!("\nBest single run parameters:");
            println!("  Mu: {}, Lambda: {}", params.mu, params.lambda);
            println!("  Sigma: {:.3}", params.sigma);
            println!("  Score: {:.6}", es_analysis.best_single_run_score);
        }
        
        if let Some(ref params) = es_analysis.best_avg_params {
            println!("\nBest average parameters:");
            println!("  Mu: {}, Lambda: {}", params.mu, params.lambda);
            println!("  Sigma: {:.3}", params.sigma);
            println!("  Average score: {:.6}", es_analysis.best_avg_score);
        }
        
        // Compare algorithms
        println!("\n--- Algorithm Comparison ---");
        if sga_analysis.best_avg_score > es_analysis.best_avg_score {
            println!("SGA performs better on average");
            println!("SGA avg score: {:.6} vs ES avg score: {:.6}", 
                sga_analysis.best_avg_score, es_analysis.best_avg_score);
        } else {
            println!("ES performs better on average");
            println!("ES avg score: {:.6} vs SGA avg score: {:.6}", 
                es_analysis.best_avg_score, sga_analysis.best_avg_score);
        }
    }
    
    // Helper functions for parameter key conversion
    fn ga_params_to_key(params: &HashMap<String, f64>) -> String {
        format!("{}_{:.3}", 
            params.get("pop_size").unwrap_or(&0.0),
            params.get("mutation_rate").unwrap_or(&0.0)
        )
    }
    
    fn es_params_to_key(params: &HashMap<String, f64>) -> String {
        format!("{}_{:.3}", 
            params.get("lambda").unwrap_or(&0.0),
            params.get("sigma").unwrap_or(&0.0)
        )
    }
    
    fn key_to_ga_params(params: &HashMap<String, f64>) -> Option<GAParameters> {
        Some(GAParameters {
            pop_size: *params.get("pop_size")? as usize,
            mem_size: *params.get("mem_size")? as usize,
            mutation_rate: *params.get("mutation_rate")?,
            crossover_rate: *params.get("crossover_rate")?,
            max_iters: *params.get("max_iters")? as usize,
            convergence_threshold: *params.get("convergence_threshold")?,
        })
    }
    
    fn key_to_es_params(params: &HashMap<String, f64>) -> Option<ESParameters> {
        Some(ESParameters {
            mu: *params.get("mu")? as usize,
            lambda: *params.get("lambda")? as usize,
            mem_size: *params.get("mem_size")? as usize,
            mem_range: (
                *params.get("mem_range_min")?,
                *params.get("mem_range_max")?
            ),
            sigma: *params.get("sigma")?,
            tau: *params.get("tau")?,
            max_gens: *params.get("max_gens")? as usize,
        })
    }
}

#[derive(Debug)]
pub struct SGAAnalysis {
    pub total_runs: usize,
    pub best_single_run_score: f64,
    pub best_single_run_params: Option<GAParameters>,
    pub best_avg_score: f64,
    pub best_avg_params: Option<GAParameters>,
    pub avg_score: f64,
    pub convergence_rate: f64,
    pub timeout_rate: f64,
    pub param_groups: usize,
}

#[derive(Debug)]
pub struct ESAnalysis {
    pub total_runs: usize,
    pub best_single_run_score: f64,
    pub best_single_run_params: Option<ESParameters>,
    pub best_avg_score: f64,
    pub best_avg_params: Option<ESParameters>,
    pub avg_score: f64,
    pub convergence_rate: f64,
    pub timeout_rate: f64,
    pub param_groups: usize,
}