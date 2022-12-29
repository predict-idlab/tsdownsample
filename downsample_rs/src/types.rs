use std::ops::{Add, Div, Mul, Sub};

pub trait Num:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
}

// Implement the trait for all types that satisfy the trait bounds
impl<T> Num for T where
    T: Copy + PartialOrd + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
{
}
