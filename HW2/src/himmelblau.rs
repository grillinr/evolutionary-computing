use crate::fitness::Fitness;

pub struct Himmelblau;

impl Fitness for Himmelblau {
    fn fitness(&self, member: &[f64]) -> f64 {
        let x = member[0];
        let y = member[1];
        // Use the standard Himmelblau function formula
        let himmelblau_value = (x.powi(2) + y - 11.0).powi(2) + (x + y.powi(2) - 7.0).powi(2);
        // Convert to fitness in the range (0, 1], higher is better, maximum at global optima
        1.0 / (1.0 + himmelblau_value)
    }

    fn fitness_bitstring(&self, bitstring: &str) -> f64 {
        let (x, y) = self.decode_bitstring(bitstring);
        // Use the standard Himmelblau function formula
        let himmelblau_value = (x.powi(2) + y - 11.0).powi(2) + (x + y.powi(2) - 7.0).powi(2);
        // Convert to fitness in the range (0, 1], higher is better, maximum at global optima
        1.0 / (1.0 + himmelblau_value)
    }

    fn decode_bitstring(&self, bitstring: &str) -> (f64, f64) {
        if bitstring.len() % 2 != 0 {
            panic!("Bitstring length must be even for Himmelblau decoding");
        }
        let x_bits = &bitstring[..bitstring.len() / 2];
        let y_bits = &bitstring[bitstring.len() / 2..];

        // Convert binary strings (base 2) to integers, then to floats
        let mut x = i64::from_str_radix(x_bits, 2).unwrap() as f64;
        let mut y = i64::from_str_radix(y_bits, 2).unwrap() as f64;

        let max_val = 2_f64.powi(x_bits.len() as i32) - 1.0;
        x = (x / max_val) * 20.0 - 10.0; // Scale to [-10, 10]
        y = (y / max_val) * 20.0 - 10.0; // Scale to [-10, 10]

        (x, y)
    }
}
