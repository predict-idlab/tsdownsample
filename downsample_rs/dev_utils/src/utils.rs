use std::ops::{Add, Sub};

use rand::distributions::Uniform;
use rand::{thread_rng, Rng};

// random array that samples between min and max of T
pub fn get_random_array<T>(n: usize, min_value: T, max_value: T) -> Vec<T>
where
    T: Copy + rand::distributions::uniform::SampleUniform,
{
    let rng = thread_rng();
    let uni = Uniform::new_inclusive(min_value, max_value);
    rng.sample_iter(uni).take(n).collect()
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
