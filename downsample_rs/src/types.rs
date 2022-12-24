use std::ops::{Add, Div, Mul, Sub};

#[cfg(feature = "half")]
use half::f16;

pub trait FromUsize {
    fn from_usize(value: usize) -> Self;
}

// pub trait FromF64 {
//     fn from_f64(value: f64) -> Self;
// }

pub trait ToF64 {
    fn to_f64(&self) -> f64;
}

macro_rules! impl_from_and_to_traits {
    ($($t:ty),*) => {
        $(
            impl FromUsize for $t {
                #[inline]
                fn from_usize(value: usize) -> Self {
                    value as Self
                }
            }
            // impl FromF64 for $t {
            //     #[inline]
            //     fn from_f64(value: f64) -> Self {
            //         value as Self
            //     }
            // }
            impl ToF64 for $t {
                #[inline]
                fn to_f64(&self) -> f64 {
                    *self as f64
                }
            }
        )*
    };
}

impl_from_and_to_traits!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, usize);

use num_traits::cast::FromPrimitive;

#[cfg(feature = "half")]
impl FromUsize for f16 {
    #[inline]
    fn from_usize(value: usize) -> Self {
        <f16>::from_u64(value as u64).unwrap()
    }
}
// #[cfg(feature = "half")]
// impl FromF64 for f16 {
//     #[inline]
//     fn from_f64(value: f64) -> Self {
//         <f16>::from_f64(value)
//     }
// }
#[cfg(feature = "half")]
impl ToF64 for f16 {
    #[inline]
    fn to_f64(&self) -> f64 {
        <f16>::to_f64(*self)
    }
}

pub trait Num:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self> // + ToF64
// + FromF64
{
}
// pub trait NumFull: Num + FromUsize + FromF64 + ToF64 {}

// implement the traits
impl<T> Num for T where
    T: Copy + PartialOrd + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> // + ToF64
                                                                                                 // + FromF64
{
}
