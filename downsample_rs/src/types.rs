use std::ops::{Add, Div, Mul, Sub};

use argminmax::ArgMinMax;

use crate::helpers::Average;

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

pub trait Indexable<T> {
    fn get(&self, index: usize) -> T;
}

pub trait HasLength {
    fn len(&self) -> usize;
}

pub trait Sliceable {
    type SliceType;

    fn slice(&self, start_idx: usize, end_idx: usize) -> Self::SliceType;
}

pub trait IntoIter {
    type IterType: Iterator;

    fn into_iter(&self) -> Self::IterType;
}

pub trait M4Param: HasLength + ArgMinMax + Sliceable {}

impl<T, I: M4Param> M4Param for T where T: HasLength + ArgMinMax + Sliceable<SliceType = I> {}

pub trait LttbParam<T>: HasLength + Indexable<T> + IntoIter + Sliceable + Average {}

impl<T, I, S: LttbParam<I>> LttbParam<I> for T where
    T: HasLength + Indexable<I> + IntoIter + Sliceable<SliceType = S> + Average
{
}

pub trait BinSearchParam<T: Copy + PartialOrd>: Indexable<T> {}

impl<T: ?Sized, I: Copy + PartialOrd> BinSearchParam<I> for T where T: Indexable<I> {}

pub trait EqBinIdxIteratorParam<T>: Indexable<T> + HasLength {}

impl<T: ?Sized, I: Clone> EqBinIdxIteratorParam<I> for T where T: Indexable<I> + HasLength {}
