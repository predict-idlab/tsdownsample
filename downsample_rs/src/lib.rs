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

use once_cell::sync::Lazy;
use rayon::{ThreadPool, ThreadPoolBuilder};

pub static POOL: Lazy<ThreadPool> = Lazy::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(
            std::env::var("TSDOWNSAMPLE_MAX_THREADS")
                .map(|s| s.parse::<usize>().expect("integer"))
                .unwrap_or_else(|_| {
                    std::thread::available_parallelism()
                        .unwrap_or(std::num::NonZeroUsize::new(1).unwrap())
                        .get()
                }),
        )
        .build()
        .expect("could not spawn threads")
});
