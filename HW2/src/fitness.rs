// Define a trait for fitness functions for reusability
pub trait Fitness {
    fn fitness(&self, member: &[f64]) -> f64;
    fn fitness_bitstring(&self, bitstring: &str) -> f64;
    fn decode_bitstring(&self, bitstring: &str) -> (f64, f64);
}
