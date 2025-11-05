mod bitstring;
mod fitness;
mod maxones;
mod rosenbrock;

use crate::bitstring::sga;
use crate::maxones::MaxOnes;
use crate::rosenbrock::Rosenbrock;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

fn main() {
    // Seed the random number generator for reproducibility
    let mut rng = ChaCha8Rng::seed_from_u64(5132);

    // Test the algorithms
    print!("max_ones ");
    sga(&MaxOnes, 100, 32, 0.01, 0.5, 1e6 as usize, 0.85, &mut rng);

    // Reseed so running order does not matter
    rng = ChaCha8Rng::seed_from_u64(5132);

    print!("rosenbrock ");
    let rosenbrock = Rosenbrock;
    let final_rosenbrock_pop = sga(
        &rosenbrock,
        500,
        48,
        0.00125,
        0.25,
        1e6 as usize,
        0.925,
        &mut rng,
    );

    // Compute and print average x and y values of final population
    let avg_x = final_rosenbrock_pop
        .iter()
        .map(|member| rosenbrock.decode(member).0)
        .sum::<f64>()
        / final_rosenbrock_pop.len() as f64;
    let avg_y = final_rosenbrock_pop
        .iter()
        .map(|member| rosenbrock.decode(member).0)
        .sum::<f64>()
        / final_rosenbrock_pop.len() as f64;
    println!("Average member: ({avg_x:.4}, {avg_y:.4})");
}
