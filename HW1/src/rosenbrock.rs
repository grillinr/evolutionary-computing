use crate::fitness::Fitness;

// Rosenbrock fitness implementation
pub struct Rosenbrock;

impl Fitness for Rosenbrock {
    fn fitness(&self, bitstring: &str) -> f64 {
        // Decode bitstring to (x, y) in range [-2, 2]
        let (x, y) = self.decode(bitstring);
        // Use the standard Rosenbrock function formula
        let a = 1.0;
        let b = 100.0;
        let rosenbrock_value = (a - x).powi(2) + b * (y - x.powi(2)).powi(2);

        // Convert to fitness (higher is better, maximum at global optimum x=1, y=1)
        1.0 / (1.0 + rosenbrock_value)
    }
}

impl Rosenbrock {
    //Converts bitstring back to float values in range [-2, 2]
    pub fn decode(&self, bitstring: &str) -> (f64, f64) {
        if bitstring.len() % 2 != 0 {
            panic!("Bitstring length must be even for Rosenbrock decoding");
        }
        let x_bits = &bitstring[..bitstring.len() / 2];
        let y_bits = &bitstring[bitstring.len() / 2..];

        // Convert binary strings (base 2) to integers, then to floats
        let mut x = i64::from_str_radix(x_bits, 2).unwrap() as f64;
        let mut y = i64::from_str_radix(y_bits, 2).unwrap() as f64;

        let max_val = 2_f64.powi(x_bits.len() as i32) - 1.0;
        x = (x / max_val) * 4.0 - 2.0; // Scale to [-2, 2]
        y = (y / max_val) * 4.0 - 2.0; // Scale to [-2, 2]

        (x, y)
    }
}
