use crate::fitness::Fitness;

// MaxOnes fitness implementation
pub struct MaxOnes;

impl Fitness for MaxOnes {
    fn fitness(&self, bitstring: &str) -> f64 {
        bitstring.chars().filter(|c| *c == '1').count() as f64 / bitstring.len() as f64
    }
}
