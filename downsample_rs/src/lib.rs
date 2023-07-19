// It is necessary to import this at the root of the crate
// See: https://github.com/la10736/rstest/tree/master/rstest_reuse#use-rstest_resuse-at-the-top-of-your-crate
#[cfg(test)]
use rstest_reuse;

pub mod minmax;
pub use minmax::*;
pub mod lttb;
pub use lttb::*;
pub mod minmaxlttb;
pub use minmaxlttb::*;
pub mod m4;
pub use m4::*;
pub(crate) mod helpers;
pub(crate) mod searchsorted;
pub(crate) mod types;
