// Define a trait for fitness functions for reusability
pub trait Fitness {
    fn fitness(&self, bitstring: &str) -> f64;
}
