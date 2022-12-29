# Store some global configuration for tests

import numpy as np

_core_supported_dtypes = [
    np.float32,
    np.float64,
    np.int16,
    np.int32,
    np.int64,
    np.uint16,
    np.uint32,
    np.uint64,
    np.datetime64,
    np.timedelta64,
]

supported_dtypes_x = _core_supported_dtypes
supported_dtypes_y = _core_supported_dtypes + [np.float16, np.int8, np.uint8, np.bool_]

_core_rust_primitive_types = ["f32", "f64", "i16", "i32", "i64", "u16", "u32", "u64"]

rust_primitive_types_x = _core_rust_primitive_types
rust_primitive_types_y = _core_rust_primitive_types + ["f16", "i8", "u8"]
