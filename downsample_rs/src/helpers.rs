#[cfg(feature = "half")]
use half::f16;
use ndarray::ArrayView1;

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
    fn average(self) -> f64;
}

impl Average for ArrayView1<'_, f64> {
    fn average(self) -> f64 {
        self.mean().unwrap()
    }
}

impl Average for ArrayView1<'_, f32> {
    fn average(self) -> f64 {
        self.mean().unwrap() as f64
    }
}

#[cfg(feature = "half")]
impl Average for ArrayView1<'_, f16> {
    fn average(self) -> f64 {
        self.fold(0f32, |acc, &x| acc + x.to_f32()) as f64 / self.len() as f64
    }
}

macro_rules! impl_average {
    ($($t:ty)*) => ($(
        impl Average for ArrayView1<'_, $t> {
            #[inline(always)]
            fn average(self) -> f64 {
                self.fold(0f64, |acc, &x| acc + x as f64) / self.len() as f64
            }
        }
    )*)
}

// Implement for all signed and unsigned integers
impl_average!(i8 i16 i32 i64 u8 u16 u32 u64);
