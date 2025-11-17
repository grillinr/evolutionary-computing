use crate::fitness::Fitness;

pub struct Rosenbrock;

impl Fitness for Rosenbrock {
    fn fitness(&self, member: &[f64]) -> f64 {
        let mut rosenbrock_value = 0.0;
        // Use the generalized Rosenbrock function formula
        for i in 0..(member.len() - 1) {
            rosenbrock_value +=
                (1.0 - member[i]).powi(2) + 100.0 * (member[i + 1] - member[i].powi(2)).powi(2);
        }
        // Convert to fitness in the range (0, 1], higher is better, maximum at global optima
        1.0 / (1.0 + rosenbrock_value)
    }

    fn fitness_bitstring(&self, bitstring: &str, num_dims: usize) -> f64 {
        let x: Vec<f64> = self.decode_bitstring(bitstring, num_dims);
        let mut rosenbrock_value = 0.0;
        // Use the generalized Rosenbrock function formula
        for i in 0..(num_dims - 1) {
            rosenbrock_value +=
                (1.0 - x[i]).powi(2) + 100.0 * (x[i + 1] - x[i].powi(2)).powi(2);
        }
        // Convert to fitness in the range (0, 1], higher is better, maximum at global optima
        1.0 / (1.0 + rosenbrock_value)
    }

    fn decode_bitstring(&self, bitstring: &str, num_dims: usize) -> Vec<f64> {
        if num_dims == 0 {
            panic!("Number of dimensions must be greater than 0");
        }
        if bitstring.len() % num_dims != 0 {
            panic!("Bitstring length must be divisible by number of dimensions");
        }

        let mut x: Vec<f64> = Vec::new();
        for i in 0..num_dims {
            let segment = &bitstring
                [(bitstring.len() / num_dims) * i..(bitstring.len() / num_dims) * (i + 1)];

            // Convert binary strings (base 2) to integers, then to floats
            let mut value = i64::from_str_radix(segment, 2).unwrap() as f64;
            let max_val = 2_f64.powi(segment.len() as i32) - 1.0;
            value = (value / max_val) * 10.24 - 2.0; // Scale to [-5.12, 5.11]
            x.push(value);
        }
        x
    }
}
