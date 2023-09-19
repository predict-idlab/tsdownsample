use num_traits::AsPrimitive;
use polars::prelude::{ChunkAgg, ChunkAnyValue, ChunkedArray, Float64Type, TakeRandom};

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

impl<T> Average for Vec<T>
where
    T: Num + AsPrimitive<f64>,
{
    fn average(&self) -> f64 {
        self.iter().fold(0f64, |acc, &x| acc + x.as_()) as f64 / self.len() as f64
    }
}

pub trait LttbParam {
    type IterType: Iterator<Item = f64>;
    type SliceType: LttbParam;

    fn slice(&self, start_idx: usize, end_idx: usize) -> Self::SliceType;

    fn get(&self, index: usize) -> f64;

    fn into_iter(&self) -> Self::IterType;

    fn len(&self) -> usize;

    fn average(&self) -> f64;
}

impl<T> LttbParam for [T]
where
    T: Num + AsPrimitive<f64>,
{
    type IterType = std::vec::IntoIter<f64>;
    type SliceType = std::vec::Vec<f64>;

    fn slice(&self, start_idx: usize, end_idx: usize) -> Self::SliceType {
        self[start_idx..end_idx].iter().map(|v| v.as_()).collect()
    }

    fn get(&self, index: usize) -> f64 {
        self[index].as_()
    }

    fn into_iter(&self) -> Self::IterType {
        IntoIterator::into_iter(self.iter().map(|v| v.as_()).collect::<Vec<f64>>())
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn average(&self) -> f64 {
        Average::average(self)
    }
}

impl<T> LttbParam for Vec<T>
where
    T: Num + AsPrimitive<f64>,
{
    type IterType = std::vec::IntoIter<f64>;
    type SliceType = std::vec::Vec<f64>;

    fn slice(&self, start_idx: usize, end_idx: usize) -> Self::SliceType {
        self[start_idx..end_idx].iter().map(|v| v.as_()).collect()
    }

    fn get(&self, index: usize) -> f64 {
        self[index].as_()
    }

    fn into_iter(&self) -> Self::IterType {
        IntoIterator::into_iter(self.iter().map(|v| v.as_()).collect::<Vec<f64>>())
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn average(&self) -> f64 {
        Average::average(self)
    }
}

impl LttbParam for ChunkedArray<Float64Type> {
    type IterType = std::vec::IntoIter<f64>;
    type SliceType = ChunkedArray<Float64Type>;

    fn slice(&self, start_idx: usize, end_idx: usize) -> Self::SliceType {
        self.slice(start_idx as i64, end_idx - start_idx)
    }

    fn get(&self, index: usize) -> f64 {
        match self
            .get_any_value(index)
            .unwrap()
            .cast(&polars::prelude::DataType::Float64)
            .unwrap()
        {
            polars::prelude::AnyValue::Float64(x) => x,
            _ => panic!(""), // this can never be reached, as it should have panicked when casting
        }
    }

    fn into_iter(&self) -> Self::IterType {
        // TODO: fix this so we don't do any needless copying
        self.into_no_null_iter().collect::<Vec<f64>>().into_iter()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn average(&self) -> f64 {
        self.mean().unwrap_or(0.0)
    }
}
