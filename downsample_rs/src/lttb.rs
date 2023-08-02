use super::helpers::Average;
use super::types::Num;
use num_traits::AsPrimitive;
use std::cmp;

#[inline(always)]
fn f64_to_i64unsigned(v: f64) -> i64 {
    // Transmute to i64 and mask out the sign bit
    let v: i64 = unsafe { std::mem::transmute::<f64, i64>(v) };
    v & 0x7FFF_FFFF_FFFF_FFFF
}

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn lttb_with_x<Tx: Num + AsPrimitive<f64>, Ty: Num + AsPrimitive<f64>>(
    x: &[Tx],
    y: &[Ty],
    n_out: usize,
) -> Vec<usize> {
    assert_eq!(x.len(), y.len());
    if n_out >= x.len() {
        return (0..x.len()).collect::<Vec<usize>>();
    }
    assert!(n_out >= 3); // avoid division by 0

    // Bucket size. Leave room for start and end data points.
    let every: f64 = (x.len() - 2) as f64 / (n_out - 2) as f64;
    // Initially a is the first point in the triangle.
    let mut a: usize = 0;

    let mut sampled_indices: Vec<usize> = vec![usize::default(); n_out];

    // Always add the first point
    sampled_indices[0] = 0;

    for i in 0..n_out - 2 {
        // Calculate point average for next bucket (containing c).
        let avg_range_start = (every * (i + 1) as f64) as usize + 1;
        let avg_range_end = cmp::min((every * (i + 2) as f64) as usize + 1, x.len());

        let y_slice = &y[avg_range_start..avg_range_end];
        let avg_y: f64 = y_slice.average();
        // TODO: avg_y could be approximated argminmax instead of mean?
        // TODO: below is faster than above, but not as accurate
        // let avg_x: f64 = (x_slice[avg_range_end - 1].as_() + x_slice[avg_range_start].as_()) / 2.0;
        let avg_x: f64 = unsafe {
            (x.get_unchecked(avg_range_end - 1).as_() + x.get_unchecked(avg_range_start).as_())
                / 2.0
        };

        // Get the range for this bucket
        let range_offs = (every * i as f64) as usize + 1;
        let range_to = avg_range_start; // = start of the next bucket

        // Point a
        let point_ax = unsafe { x.get_unchecked(a).as_() };
        let point_ay = unsafe { y.get_unchecked(a).as_() };

        let d1 = point_ax - avg_x;
        let d2 = avg_y - point_ay;
        let offset: f64 = d1 * point_ay + d2 * point_ax;

        let x_slice = &x[range_offs..range_to];
        let y_slice = &y[range_offs..range_to];
        (_, a) = y_slice.iter().zip(x_slice.iter()).enumerate().fold(
            (-1i64, a),
            |(max_area, a), (i, (y_, x_))| {
                // Calculate triangle area over three buckets
                // -> area = d1 * (y_ - point_ay) - (point_ax - x_) * d2;
                // let area = d1 * y[i].as_() + d2 * x[i].as_() - offset;
                // let area = d1 * y_slice[i].as_() + d2 * x_slice[i].as_() - offset;
                let area = d1 * y_.as_() + d2 * x_.as_() - offset;
                let area = f64_to_i64unsigned(area); // this is faster than abs
                if area > max_area {
                    (area, i)
                } else {
                    (max_area, a)
                }
            },
        );
        a += range_offs;

        sampled_indices[i + 1] = a;
    }

    // Always add the last point
    sampled_indices[n_out - 1] = y.len() - 1;

    sampled_indices
}

// ----------- WITHOUT X

pub fn lttb_without_x<Ty: Num + AsPrimitive<f64>>(y: &[Ty], n_out: usize) -> Vec<usize> {
    if n_out >= y.len() {
        return (0..y.len()).collect::<Vec<usize>>();
    }
    assert!(n_out >= 3); // avoid division by 0

    // Bucket size. Leave room for start and end data points.
    let every: f64 = (y.len() - 2) as f64 / (n_out - 2) as f64;
    // Initially a is the first point in the triangle.
    let mut a: usize = 0;

    let mut sampled_indices: Vec<usize> = vec![usize::default(); n_out];

    // Always add the first point
    sampled_indices[0] = 0;

    for i in 0..n_out - 2 {
        // Calculate point average for next bucket (containing c).
        let avg_range_start = (every * (i + 1) as f64) as usize + 1;
        let avg_range_end = cmp::min((every * (i + 2) as f64) as usize + 1, y.len());

        let y_slice = &y[avg_range_start..avg_range_end];
        let avg_y: f64 = y_slice.average();
        let avg_x: f64 = (avg_range_start + avg_range_end - 1) as f64 / 2.0;

        // Get the range for this bucket
        let range_offs = (every * i as f64) as usize + 1;
        let range_to = avg_range_start; // = start of the next bucket

        // Point a
        let point_ay = unsafe { y.get_unchecked(a).as_() };
        let point_ax = a as f64;

        let d1 = point_ax - avg_x;
        let d2 = avg_y - point_ay;
        let point_ax = point_ax - range_offs as f64;

        // let mut max_area = -1i64;
        let mut ax_x = point_ax; // point_ax - x[i]
        let offset: f64 = d1 * point_ay;

        // TODO: for some reason is this faster than the loop below -> check if this is true for other devices
        let y_slice = &y[range_offs..range_to];
        (_, a) = y_slice
            .iter()
            .enumerate()
            .fold((-1i64, a), |(max_area, a), (i, y)| {
                // Calculate triangle area over three buckets
                // -> area: f64 = d1 * y[i].as_() - ax_x * d2;
                let area: f64 = d1 * y.as_() - ax_x * d2 - offset;
                let area: i64 = f64_to_i64unsigned(area);
                ax_x -= 1.0;
                if area > max_area {
                    (area, i + range_offs)
                } else {
                    (max_area, a)
                }
            });

        // let y_slice = unsafe { std::slice::from_raw_parts(y_ptr.add(range_offs), range_to - range_offs) };
        // (_, a) = y_slice
        //     .iter()
        //     .enumerate()
        //     .fold((-1i64, a), |(max_area, a), (i, y_)| {
        //         // Calculate triangle area over three buckets
        //         // -> area: f64 = d1 * y[i].as_() - ax_x * d2;
        //         let area: f64 = d1 * y_.as_() - ax_x * d2 - offset;
        //         let area: i64 = f64_to_i64unsigned(area);
        //         ax_x -= 1.0;
        //         if area > max_area {
        //             (area, i)
        //         } else {
        //             (max_area, a)
        //         }
        //     });
        // a += range_offs;

        sampled_indices[i + 1] = a;
    }

    // Always add the last point
    sampled_indices[n_out - 1] = y.len() - 1;

    sampled_indices
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use dev_utils::utils;

    use super::{lttb_with_x, lttb_without_x};

    #[test]
    fn test_lttb_with_x() {
        let x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = lttb_with_x(&x, &y, 4);
        assert_eq!(sampled_indices, vec![0, 1, 5, 9]);
    }

    #[test]
    fn test_lttb_without_x() {
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = lttb_without_x(&y, 4);
        assert_eq!(sampled_indices, vec![0, 1, 5, 9]);
    }

    #[test]
    fn test_random_same_output() {
        for _ in 0..100 {
            const N: usize = 5_000;
            let x: [i32; N] = core::array::from_fn(|i| i as i32);
            let y = utils::get_random_array(N, f32::MIN, f32::MAX);
            let sampled_indices1 = lttb_with_x(&x, y.as_slice(), 200);
            let sampled_indices2 = lttb_without_x(y.as_slice(), 200);
            assert_eq!(sampled_indices1, sampled_indices2);
        }
    }
}
