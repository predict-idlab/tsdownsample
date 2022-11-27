import tsdownsample._rust._tsdownsample_rs as tsds_rs

DTYPES = ["f16", "f32", "f64", "i16", "i32", "i64", "u16", "u32", "u64"]


def _test_rust_mod_correctly_build(mod, sub_mods, has_x_impl: bool):
    # Without x
    for sub_mod in sub_mods:
        assert hasattr(mod, sub_mod)
        m = getattr(mod, sub_mod)
        for ty in DTYPES:
            assert hasattr(m, f"downsample_{ty}")
    # With x
    if not has_x_impl:
        return
    for sub_mod in sub_mods:
        assert hasattr(mod, sub_mod)
        m = getattr(mod, sub_mod)
        for tx in DTYPES:
            for ty in DTYPES:
                assert hasattr(m, f"downsample_{tx}_{ty}")


def test_minmax_rust_mod_correctly_build():
    mod = tsds_rs.minmax
    sub_mods = ["simd", "simd_parallel", "scalar", "scalar_parallel"]
    _test_rust_mod_correctly_build(mod, sub_mods, has_x_impl=False)


def test_m4_rust_mod_correctly_build():
    mod = tsds_rs.m4
    sub_mods = ["simd", "simd_parallel", "scalar", "scalar_parallel"]
    _test_rust_mod_correctly_build(mod, sub_mods, has_x_impl=False)


def test_lttb_rust_mod_correctly_build():
    mod = tsds_rs.lttb
    sub_mods = ["scalar"]
    _test_rust_mod_correctly_build(mod, sub_mods, has_x_impl=True)


def test_minmaxlttb_rust_mod_correctly_build():
    mod = tsds_rs.minmaxlttb
    sub_mods = ["simd", "simd_parallel", "scalar", "scalar_parallel"]
    _test_rust_mod_correctly_build(mod, sub_mods, has_x_impl=True)
