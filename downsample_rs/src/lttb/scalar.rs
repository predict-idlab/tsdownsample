use super::super::types::Num;
use ndarray::{Array1, ArrayView1};
use num_traits::AsPrimitive;
use std::cmp;

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn lttb_with_x<Tx: Num + AsPrimitive<f64>, Ty: Num + AsPrimitive<f64>>(
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
    let every = (x.len() - 2) as f64 / (n_out - 2) as f64;
    // Initially a is the first point in the triangle.
    let mut a = 0;

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out);

    // Always add the first point
    sampled_indices[0] = 0;

    for i in 0..n_out - 2 {
        // Calculate point average for next bucket (containing c).
        let mut avg_x: f64 = 0.0;
        let mut avg_y: f64 = 0.0;

        let avg_range_start = (every * (i + 1) as f64) as usize + 1;
        let avg_range_end = cmp::min((every * (i + 2) as f64) as usize + 1, x.len());

        for i in avg_range_start..avg_range_end {
            avg_x += x[i].as_();
            avg_y += y[i].as_();
        }
        // Slicing seems to be a lot slower
        // let avg_x: Tx = x.slice(s![avg_range_start..avg_range_end]).sum();
        // let avg_y: Ty = y.slice(s![avg_range_start..avg_range_end]).sum();
        let avg_x: f64 = avg_x / (avg_range_end - avg_range_start) as f64;
        let avg_y: f64 = avg_y / (avg_range_end - avg_range_start) as f64;

        // Get the range for this bucket
        let range_offs = (every * i as f64) as usize + 1;
        let range_to = (every * (i + 1) as f64) as usize + 1;

        // Point a
        let point_ax = x[a].as_();
        let point_ay = y[a].as_();

        let mut max_area = -1.0;
        for i in range_offs..range_to {
            // Calculate triangle area over three buckets
            let area = ((point_ax - avg_x) * (y[i].as_() - point_ay)
                - (point_ax - x[i].as_()) * (avg_y - point_ay))
                .abs();
            if area > max_area {
                max_area = area;
                a = i;
            }
        }
        // Vectorized implementation
        // let point_ax: Tx = x[a];
        // let point_ay: Ty = y[a];
        // let ar_x: Vec<Tx> = x.slice(s![range_offs..range_to]).into_iter().map(|v| point_ax - *v).collect();
        // let ar_y: Vec<Ty> = y.slice(s![range_offs..range_to]).into_iter().map(|v| *v - point_ay).collect();
        // let max_idx: usize = (ar_x.iter().zip(ar_y.iter()).map(|(x, y)| (x.to_f64().unwrap() * avg_y - y.to_f64().unwrap() * avg_x).abs()).enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0) + range_offs;
        // a = max_idx;
        sampled_indices[i + 1] = a;
    }

    // Always add the last point
    sampled_indices[n_out - 1] = y.len() - 1;

    sampled_indices
}

// ----------- WITHOUT X

pub fn lttb_without_x<Ty: Num + AsPrimitive<f64>>(
    // TODO: why is this slower than the one with x?
    y: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize> {
    if n_out >= y.len() || n_out == 0 {
        return Array1::from((0..y.len()).collect::<Vec<usize>>());
    }
    assert!(n_out >= 3); // avoid division by 0

    // Bucket size. Leave room for start and end data points.
    let every = (y.len() - 2) as f64 / (n_out - 2) as f64;
    // Initially a is the first point in the triangle.
    let mut a = 0;

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out);

    // Always add the first point
    sampled_indices[0] = 0;

    for i in 0..n_out - 2 {
        // Calculate point average for next bucket (containing c).
        let mut avg_y: f64 = 0.0;

        let avg_range_start = (every * (i + 1) as f64) as usize + 1;
        let avg_range_end = cmp::min((every * (i + 2) as f64) as usize + 1, y.len());

        for i in avg_range_start..avg_range_end {
            avg_y += y[i].as_();
        }
        // Slicing seems to be a lot slower
        // let avg_x: Tx = x.slice(s![avg_range_start..avg_range_end]).sum();
        let avg_y: f64 = avg_y / (avg_range_end - avg_range_start) as f64;
        let avg_x: f64 = (avg_range_start + avg_range_end - 1) as f64 / 2.0;

        // Get the range for this bucket
        let range_offs = (every * i as f64) as usize + 1;
        let range_to = (every * (i + 1) as f64) as usize + 1;

        // Point a
        let point_ay = y[a].as_();
        let point_ax = a as f64;

        let mut max_area = -1.0;
        for i in range_offs..range_to {
            // Calculate triangle area over three buckets
            let area = ((point_ax - avg_x) * (y[i].as_() - point_ay)
                - (point_ax - i as f64) * (avg_y - point_ay))
                .abs();
            if area > max_area {
                max_area = area;
                a = i;
            }
        }
        // Vectorized implementation
        // let point_ax: Tx = x[a];
        // let point_ay: Ty = y[a];
        // let ar_x: Vec<Tx> = x.slice(s![range_offs..range_to]).into_iter().map(|v| point_ax - *v).collect();
        // let ar_y: Vec<Ty> = y.slice(s![range_offs..range_to]).into_iter().map(|v| *v - point_ay).collect();
        // let max_idx: usize = (ar_x.iter().zip(ar_y.iter()).map(|(x, y)| (x.to_f64().unwrap() * avg_y - y.to_f64().unwrap() * avg_x).abs()).enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0) + range_offs;
        // a = max_idx;
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
