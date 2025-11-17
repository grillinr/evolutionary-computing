use std::vec;

fn main() {
    // Test the Rosenbrock function directly
    let optimal = vec![1.0; 10];
    let random = vec![0.5, -1.2, 2.3, 0.1, -0.8, 1.5, -2.1, 0.9, 1.1, -0.3];
    
    println!("Testing Rosenbrock function:");
    
    // Test optimal point (should be 0)
    let rosenbrock_optimal = rosenbrock(&optimal);
    let fitness_optimal = 1.0 / (1.0 + rosenbrock_optimal);
    println!("Optimal point: rosenbrock={}, fitness={}", rosenbrock_optimal, fitness_optimal);
    
    // Test random point
    let rosenbrock_random = rosenbrock(&random);
    let fitness_random = 1.0 / (1.0 + rosenbrock_random);
    println!("Random point: rosenbrock={}, fitness={}", rosenbrock_random, fitness_random);
    
    // Test near-optimal point
    let near_optimal = vec![1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99, 1.03, 0.97];
    let rosenbrock_near = rosenbrock(&near_optimal);
    let fitness_near = 1.0 / (1.0 + rosenbrock_near);
    println!("Near optimal: rosenbrock={}, fitness={}", rosenbrock_near, fitness_near);
}

fn rosenbrock(x: &[f64]) -> f64 {
    let mut value = 0.0;
    for i in 0..(x.len() - 1) {
        value += (1.0 - x[i]).powi(2) + 100.0 * (x[i + 1] - x[i].powi(2)).powi(2);
    }
    value
}