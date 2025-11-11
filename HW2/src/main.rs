mod bitstring;
mod evol_strat;
mod fitness;
mod himmelblau;

use crate::bitstring::sga;
use crate::evol_strat::evolution_strategy;
use crate::fitness::Fitness;
use crate::himmelblau::Himmelblau;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

fn main() {
    // Seed the random number generator for reproducibility
    let mut rng = ChaCha8Rng::seed_from_u64(5000);
    let final_es_pop = evolution_strategy(
        &Himmelblau,
        15,            // mu: parent population size
        100,           // lambda: offspring population size
        2,             // Member size
        (-10.0, 10.0), // Gene range
        0.1,           // Initial mutation standard deviation
        0.1,           // Tau for sigma adaptation
        1000,          // Max generations
        &mut rng,
    );

    // Reset RNG for fair comparison
    let mut rng = ChaCha8Rng::seed_from_u64(5000);
    let final_ea_pop = sga(
        &Himmelblau,
        100,          //population size (mu = lambda)
        20,           // member size (in bits)
        0.00125,      // mutation rate
        0.25,         // crossover rate
        1e6 as usize, // max evaluations
        0.925,        // convergence threshold
        &mut rng,
    );

    // Print final populations and their fitnesses
    for member in final_es_pop {
        let fitness = Himmelblau.fitness(&member[0..2]);
        println!("ES Member: {member:?}, Fitness: {fitness}");
    }

    for member in final_ea_pop {
        let fitness = Himmelblau.fitness_bitstring(&member);
        let (x, y) = Himmelblau.decode_bitstring(&member);
        println!("EA Member: ({x:.4}, {y:.4}), Fitness: {fitness}");
    }
}
