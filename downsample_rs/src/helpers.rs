use num_traits::AsPrimitive;

use crate::types::Num;

// ------------ AVERAGE

// TODO: future work -> this can be optimized by using SIMD instructions (similar to the argminmax crate)
// TODO: this implementation can overfow (but numpy does the same)

// This trait implements the average function for all types that this crate
// supports. It is used in the lttb algorithm.
// We intend to use the same implementation for all types as is used in the
// numpy (Python) library (- which uses add reduce):
//  - f64 & f32: use the data type to calculate the average
//  - f16: cast to f32 and calculate the average
//  - signed & unsigned integers: cast to f64 and calculate the average
// Note: the only difference with the numpy implementation is that this
// implementation always returns a f64, while numpy returns f32 for f32 and f16
// (however the calculation is done in f32 - only the result is casted to f64).
// See more details: https://github.com/numpy/numpy/blob/8cec82012694571156e8d7696307c848a7603b4e/numpy/core/_methods.py#L164

pub trait Average {
    fn average(&self) -> f64;
}

impl<T> Average for [T]
where
    T: Num + AsPrimitive<f64>,
{
    fn average(&self) -> f64 {
        self.iter().fold(0f64, |acc, &x| acc + x.as_()) as f64 / self.len() as f64
    }
}
