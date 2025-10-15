import tsdownsample._rust._tsdownsample_rs as tsds_rs
from test_config import (
    rust_primitive_types_x,
    rust_primitive_types_y,
    rust_primitive_types_y_nan,
)


def _test_rust_mod_correctly_build(mod, sub_mods, has_x_impl: bool):
    # Without x
    for sub_mod in sub_mods:
        assert hasattr(mod, sub_mod)
        m = getattr(mod, sub_mod)
        for ty in rust_primitive_types_y:
            assert hasattr(m, f"downsample_{ty}")
    # With x
    if not has_x_impl:
        return
    for sub_mod in sub_mods:
        assert hasattr(mod, sub_mod)
        m = getattr(mod, sub_mod)
        for tx in rust_primitive_types_x:
            for ty in rust_primitive_types_y:
                assert hasattr(m, f"downsample_{tx}_{ty}")


def _test_rust_nan_mod_correctly_build(mod, sub_mods, has_x_impl: bool):
    # without x
    for sub_mod in sub_mods:
        assert hasattr(mod, sub_mod)
        m = getattr(mod, sub_mod)
        for ty in rust_primitive_types_y_nan:
            assert hasattr(m, f"downsample_nan_{ty}")

    # with x
    if not has_x_impl:
        return
    for sub_mod in sub_mods:
        assert hasattr(mod, sub_mod)
        m = getattr(mod, sub_mod)
        for tx in rust_primitive_types_x:
            for ty in rust_primitive_types_y_nan:
                assert hasattr(m, f"downsample_{tx}_{ty}")


def test_minmax_rust_mod_correctly_build():
    mod = tsds_rs.minmax
    sub_mods = ["sequential", "parallel"]
    _test_rust_mod_correctly_build(mod, sub_mods, has_x_impl=True)
    _test_rust_nan_mod_correctly_build(mod, sub_mods, has_x_impl=True)


def test_m4_rust_mod_correctly_build():
    mod = tsds_rs.m4
    sub_mods = ["sequential", "parallel"]
    _test_rust_mod_correctly_build(mod, sub_mods, has_x_impl=True)
    _test_rust_nan_mod_correctly_build(mod, sub_mods, has_x_impl=True)


def test_lttb_rust_mod_correctly_build():
    mod = tsds_rs.lttb
    sub_mods = ["sequential"]
    _test_rust_mod_correctly_build(mod, sub_mods, has_x_impl=True)


def test_minmaxlttb_rust_mod_correctly_build():
    mod = tsds_rs.minmaxlttb
    sub_mods = ["sequential", "parallel"]
    _test_rust_mod_correctly_build(mod, sub_mods, has_x_impl=True)
    _test_rust_nan_mod_correctly_build(mod, sub_mods, has_x_impl=True)
