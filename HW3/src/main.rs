mod bitstring;
mod evol_strat;
mod fitness;
mod rosenbrock;

use crate::bitstring::sga;
use crate::evol_strat::evolution_strategy;
use crate::fitness::Fitness;
use crate::rosenbrock::Rosenbrock;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

fn main() {
    const NUM_DIMS: usize = 10;
    // Seed the random number generator for reproducibility
    let mut rng = ChaCha8Rng::seed_from_u64(5000);
    
    // Run ES for comparison
    let final_es_pop = evolution_strategy(
        &Rosenbrock,
        100,           // population size (mu)
        100,           // offspring size (lambda)
        NUM_DIMS,      // number of dimensions
        (-5.12, 5.11), // range for initial population
        1.0,           // initial step size (sigma)
        1.0 / (2.0 * NUM_DIMS as f64).sqrt(), // learning rate (tau)
        1000,          // max generations
        &mut rng,
    );
    
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
    let final_ea_pop = sga(
        &Rosenbrock,
        100,           //population size (mu = lambda)
        20 * NUM_DIMS, // member size (in bits)
        0.01,          // mutation rate
        0.75,          // crossover rate
        1000,          // max evaluations
        0.95,          // convergence threshold
        &mut rng,
    );

    // Print final populations and their fitnesses
    // ES currently disabled to focus on GA
    /*
    for member in final_es_pop {
        let fitness = Rosenbrock.fitness(&member[0..NUM_DIMS]);
        let x = &member[0..NUM_DIMS];
        println!("ES Member: {x:?}, Fitness: {fitness}");
    }
    */

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
