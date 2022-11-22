#[cfg(feature = "half")]
use half::f16;

pub trait ToF64 {
    fn to_f64(&self) -> f64;
}

macro_rules! impl_to_f64 {
    ($($t:ty),*) => {
        $(
            impl ToF64 for $t {
                #[inline]
                fn to_f64(&self) -> f64 {
                    *self as f64
                }
            }
        )*
    };
}

impl_to_f64!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, usize);

// impl ToF64 for bool {
//     #[inline]
//     fn to_f64(&self) -> f64 {
//         if *self {
//             1.0
//         } else {
//             0.0
//         }
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
    Copy + Default + std::ops::Add<Output = Self> + std::ops::Div<Output = Self> + ToF64
{
}

// implement the trait
impl<T> Num for T where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Div<Output = T> + ToF64
{
}
