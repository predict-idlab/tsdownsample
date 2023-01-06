use super::super::types::Num;
use ndarray::{s, Array1, ArrayView1};
use num_traits::{AsPrimitive, FromPrimitive, Zero};
use std::cmp;

#[inline(always)]
fn f64_to_i64unsigned(v: f64) -> i64 {
    // Transmute to i64 and mask out the sign bit
    let v: i64 = unsafe { std::mem::transmute::<f64, i64>(v) };
    v & 0x7FFF_FFFF_FFFF_FFFF
}

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn lttb_with_x<
    Tx: Num + AsPrimitive<f64>,
    Ty: Num + AsPrimitive<f64> + FromPrimitive + Zero,
>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize> {
    assert_eq!(x.len(), y.len());
    if n_out >= x.len() || n_out == 0 {
        return Array1::from((0..x.len()).collect::<Vec<usize>>());
    }
    assert!(n_out >= 3); // avoid division by 0

    // Bucket size. Leave room for start and end data points.
    let every: f64 = (x.len() - 2) as f64 / (n_out - 2) as f64;
    // Initially a is the first point in the triangle.
    let mut a: usize = 0;

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out);

    // Always add the first point
    sampled_indices[0] = 0;

    for i in 0..n_out - 2 {
        // Calculate point average for next bucket (containing c).
        let avg_range_start = (every * (i + 1) as f64) as usize + 1;
        let avg_range_end = cmp::min((every * (i + 2) as f64) as usize + 1, x.len());

        // for i in avg_range_start..avg_range_end {
        //     avg_x += x[i].as_();
        //     avg_y += y[i].as_();
        // }
        // avg_x /= (avg_range_end - avg_range_start) as f64;
        // avg_y /= (avg_range_end - avg_range_start) as f64;
        // let avg_y: f64 = y.slice(s![avg_range_start..avg_range_end]).sum().as_()
        //     / (avg_range_end - avg_range_start) as f64;
        let avg_y: f64 = y
            .slice(s![avg_range_start..avg_range_end])
            .mean()
            .unwrap()
            .as_();
        // TODO: avg_y could be approximated argminmax instead of mean?

        // let avg_x: f64 = x.slice(s![avg_range_start..avg_range_end]).sum().as_() / (avg_range_end - avg_range_start) as f64;
        // TODO: below is faster than above, but not as accurate
        let avg_x: f64 = (x[avg_range_end - 1].as_() + x[avg_range_start].as_()) / 2.0;

        // Get the range for this bucket
        let range_offs = (every * i as f64) as usize + 1;
        let range_to = avg_range_start; // = start of the next bucket

        // Point a
        let point_ax = x[a].as_();
        let point_ay = y[a].as_();

        let mut max_area = -1i64;
        let d1 = point_ax - avg_x;
        let d2 = avg_y - point_ay;
        let offset: f64 = d1 * point_ay + d2 * point_ax;
        for i in range_offs..range_to {
            // Calculate triangle area over three buckets
            // -> area = d1 * (y_ - point_ay) - (point_ax - x_) * d2;
            let area = d1 * y[i].as_() + d2 * x[i].as_() - offset;
            let abs_area = f64_to_i64unsigned(area);
            if abs_area > max_area {
                max_area = abs_area;
                a = i;
            }
        }
        sampled_indices[i + 1] = a;
    }

    // Always add the last point
    sampled_indices[n_out - 1] = y.len() - 1;

    sampled_indices
}

// ----------- WITHOUT X

pub fn lttb_without_x<Ty: Num + AsPrimitive<f64> + FromPrimitive + Zero>(
    // TODO: why is this slower than the one with x?
    y: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize> {
    if n_out >= y.len() || n_out == 0 {
        return Array1::from((0..y.len()).collect::<Vec<usize>>());
    }
    assert!(n_out >= 3); // avoid division by 0

    // Bucket size. Leave room for start and end data points.
    let every: f64 = (y.len() - 2) as f64 / (n_out - 2) as f64;
    // Initially a is the first point in the triangle.
    let mut a: usize = 0;

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out);

    // Always add the first point
    sampled_indices[0] = 0;

    for i in 0..n_out - 2 {
        // Calculate point average for next bucket (containing c).
        let avg_range_start = (every * (i + 1) as f64) as usize + 1;
        let avg_range_end = cmp::min((every * (i + 2) as f64) as usize + 1, y.len());

        // TODO: handle this with a trait?
        //  => f32 and f64 can be handled with the same code
        //  => all other dtypes can be handled with a cast to f64 in a fold
        let avg_y: f64 = y
            .slice(s![avg_range_start..avg_range_end])
            .mean()
            .unwrap()
            .as_();

        // let avg_y: f64 = y.slice(s![avg_range_start..avg_range_end]).sum().as_()
        //     / (avg_range_end - avg_range_start) as f64;
        // Do not use slice here, it is slower
        // let y_ptr = y.as_ptr();
        // let y_slice = unsafe { ArrayView1::from_shape_ptr(
        //     avg_range_end - avg_range_start,
        //     y_ptr.add(avg_range_start),
        // ) };
        // let avg_y: f64 = y_slice.sum().as_() / (avg_range_end - avg_range_start) as f64;
        // let avg_y: f64 = (y[avg_range_end - 1].as_() + y[avg_range_start].as_()) / 2.0;
        // let avg_y: f64 = y
        //     .slice(s![avg_range_start..avg_range_end])
        //     .iter()
        //     .fold(0.0, |acc, y| acc + y.as_()) // TODO: this might overflow
        //     / (avg_range_end - avg_range_start) as f64;
        let avg_x: f64 = (avg_range_start + avg_range_end - 1) as f64 / 2.0;

        // Get the range for this bucket
        let range_offs = (every * i as f64) as usize + 1;
        let range_to = avg_range_start; // = start of the next bucket

        // Point a
        let point_ay = y[a].as_();
        let point_ax = a as f64;

        let d1 = point_ax - avg_x;
        let d2 = avg_y - point_ay;
        let point_ax = point_ax - range_offs as f64;

        let mut max_area = -1i64;
        let mut ax_x = point_ax; // point_ax - x[i]
        let offset: f64 = d1 * point_ay;
        for i in range_offs..range_to {
            // Calculate triangle area over three buckets
            // -> area: f64 = d1 * y[i].as_() - ax_x * d2;
            let area: f64 = d1 * y[i].as_() - ax_x * d2 - offset;
            let abs_area: i64 = f64_to_i64unsigned(area);
            if abs_area > max_area {
                a = i;
                max_area = abs_area;
            }
            ax_x -= 1.0;
        }

        sampled_indices[i + 1] = a;
    }

    // Always add the last point
    sampled_indices[n_out - 1] = y.len() - 1;

    sampled_indices
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    extern crate dev_utils;

    use dev_utils::utils;

    use super::{lttb_with_x, lttb_without_x};
    use ndarray::{array, Array1};

    #[test]
    fn test_lttb_with_x() {
        let x = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = lttb_with_x(x.view(), y.view(), 4);
        assert_eq!(sampled_indices, array![0, 1, 5, 9]);
    }

    #[test]
    fn test_lttb_without_x() {
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = lttb_without_x(y.view(), 4);
        assert_eq!(sampled_indices, array![0, 1, 5, 9]);
    }

    #[test]
    fn test_random_same_output() {
        for _ in 0..100 {
            let n = 5_000;
            let x: Array1<i32> = Array1::from((0..n).map(|i| i as i32).collect::<Vec<i32>>());
            let y = utils::get_random_array(n, f32::MIN, f32::MAX);
            let sampled_indices1 = lttb_with_x(x.view(), y.view(), 200);
            let sampled_indices2 = lttb_without_x(y.view(), 200);
            assert_eq!(sampled_indices1, sampled_indices2);
        }
    }
}
