use crate::fitness::Fitness;

pub struct MaxOnes;

impl Fitness for MaxOnes {
    fn fitness(&self, _member: &[f64]) -> f64 {
        // Not used for bitstring
        0.0
    }

    fn fitness_bitstring(&self, bitstring: &str) -> f64 {
        let ones: f64 = bitstring.chars().filter(|&c| c == '1').count() as f64;
        ones / bitstring.len() as f64
    }

    fn decode_bitstring(&self, _bitstring: &str) -> (f64, f64) {
        (0.0, 0.0)
    }
}