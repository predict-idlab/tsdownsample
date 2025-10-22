use std::ops::{Add, Sub};

use num_traits::{NumCast, ToPrimitive};
use rand::distr::uniform::Error as UniformError;
use rand::distr::Uniform;
use rand::{rng, Rng};

// random array that samples between min and max of T
pub fn get_random_array<T>(n: usize, min_value: T, max_value: T) -> Vec<T>
where
    T: Copy + rand::distr::uniform::SampleUniform + ToPrimitive + NumCast,
{
    let rng = rng();
    match Uniform::new_inclusive(min_value, max_value) {
        Ok(uni) => rng.sample_iter(uni).take(n).collect(),
        Err(UniformError::NonFinite) => {
            let min = min_value
                .to_f64()
                .expect("failed to convert lower bound to f64");
            let max = max_value
                .to_f64()
                .expect("failed to convert upper bound to f64");
            let uni = Uniform::new_inclusive(min, max).unwrap();
            rng.sample_iter(uni)
                .take(n)
                .map(|v| NumCast::from(v).expect("failed to convert sample"))
                .collect()
        }
        Err(err) => panic!("invalid range for random array: {err:?}"),
    }
}

// worst case array that alternates between increasing max and decreasing min values
pub fn get_worst_case_array<T>(n: usize, step: T) -> Vec<T>
where
    T: Copy + Default + Sub<Output = T> + Add<Output = T>,
{
    let mut arr: Vec<T> = Vec::with_capacity(n);
    let mut min_value: T = Default::default();
    let mut max_value: T = Default::default();
    for i in 0..n {
        if i % 2 == 0 {
            arr.push(min_value);
            min_value = min_value - step;
        } else {
            arr.push(max_value);
            max_value = max_value + step;
        }
    }
    arr
}
