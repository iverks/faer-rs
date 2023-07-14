use dyn_stack::{DynStack, GlobalMemBuffer, StackReq};
use faer_core::{
    mat, mul::inner_prod::inner_prod_with_conj, zipped, ComplexField, Conj, Mat, MatMut, MatRef,
};
use faer_evd::{hessenberg_real_evd::EvdParams, SymmetricEvdParams};
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

/// Implicitly restarted arnoldi iteration
/// matrix is m x m
/// vector is m long
/// n is 1 less than size of krylov subspace, should be >=  1
pub fn implicitly_restarted_arnoldi<T: ComplexField>(
    matrix: MatMut<'_, T>,
    vector: MatMut<'_, T>,
    n: i32,
) -> Result<(), ()> {
    // 0. Start: Choose initial vector v^0_1
    // 1. Build an initial Arnoldi iteration of k steps: (V^0_k , H^0_k )
    // 2. For s = 0, 1,... Do
    //      3. Test for convergence
    //      4. Extend V^s_k to k + p vectors, taking p more Arnoldi steps: (V^s_{k+p}, H^s_{k+p})
    //      5. Choose shifts µi, i = 1,...,p
    //      6. H_{k+p} = Q^T H^s_{k+p} Q, with Q the orthogonal matrix obtained
    //         through the implicit QR algorithm with µi, i = 1,...,p shifts
    //      7. Define V^(s+1)_k = [ V^s_{k+p} Q ]dot[ I_k 0 ], and
    //      8. H^(s+1)_k = [ Ik 0 ] Hk+p [ Ik 0 ]
    // 9. Enddo
    todo!();
    Ok(())
}

/// m is basis size
/// s_max is max number of steps
pub fn arnoldi<T: ComplexField>(matrix: MatMut<'_, T>, vector: MatMut<'_, T>, m: i32, s_max: i32) {
    // 0. Choose initial unit vector v^0
    // 1. For s = 0, 1,... Do
    for s in 0..s_max {
        // 2. v1 = v^s, V^s_1 = {v1}
        // 3. For j = 1,...,m Do
        //     4. hij = (Avj , vi), i = 1,...,j,
        //     5. wj = Avj − Pj_i=1 hijvi
        //     6. hj+1,j = kwjk2, if hj+1,j = 0 stop.
        //     7. vj+1 = wj/hj+1,j
        //     8. Enddo
        // 9. Compute the wanted eigenpairs (µ(s)_i , y(s)_i ) of H(s) m = (hi,j )
        //  and the Ritz vectors x(s)_i = V (s) m y(s)_i , where V (s) m = {v1,...,vm}
        // 10. v(s+1) = Pcix(s)_i , for some c_i, and the wanted x(s)_i
        // 11. Enddo
        todo!();
    }
}

pub fn lanczos(a: MatRef<'_, f64>) {
    // Define q, alpha and beta
    let n_eigs = a.ncols();
    let iters = 3;
    let mut q = Mat::<f64>::zeros(n_eigs, iters + 1);
    let mut alpha = Mat::<f64>::zeros(iters, 1);
    let mut beta = Mat::<f64>::zeros(iters, 1);

    // Generate random initial vector
    let b = Mat::with_dims(n_eigs, 1, |_, _| {
        thread_rng().sample::<f64, _>(StandardNormal)
    });

    // Get norm of b
    let bnorm = inner_prod_with_conj(b.as_ref(), Conj::Yes, b.as_ref(), Conj::No)
        .real()
        .sqrt();

    // Assign b / bnorm to first column of q
    zipped!(q.as_mut().col(0), b.as_ref())
        .for_each(|mut q_el, b_el| q_el.write(b_el.read() / bnorm));

    println!("b {:?}", q.as_ref().col(0));

    // Lanczos iteration loop
    for m in 0..iters {
        // v = a dot q[m], shape is (n_eigs, 1)
        let mut v = a * q.as_ref().col(m);

        // alpha[m] = q[m] dot v
        let qm_dot_v =
            inner_prod_with_conj(q.as_ref().col(m), Conj::Yes, v.as_ref(), Conj::No).real();
        alpha.as_mut().write(m, 0, qm_dot_v);
        if m == 0 {
            // v = v - alpha[m] * q.col(m)
            let iter = zipped!(v.as_mut(), q.as_ref().col(m));
            iter.for_each(|mut v_el, q_el| {
                v_el.write(v_el.read() - alpha.as_ref().read(m, 0) * q_el.read())
            });
        } else {
            // v = v - beta[m - 1] * &q.column(m - 1) - alpha[m] * &q.column(m);
            zipped!(v.as_mut(), q.as_ref().col(m - 1), q.as_ref().col(m)).for_each(
                |mut v_el, qm1_el, qm_el| {
                    v_el.write(
                        v_el.read()
                            - beta.as_ref().read(m - 1, 0) * qm1_el.read() // beta[m-1] * q[m-1]
                            - alpha.as_ref().read(m, 0) * qm_el.read(), // alpha[m] * q[m]
                    )
                },
            );
        }
        // let vnorm = v.iter().map(|x| x.abs()).sum();
        let vnorm = inner_prod_with_conj(v.as_ref(), Conj::Yes, v.as_ref(), Conj::No)
            .real()
            .sqrt();
        // beta[m] = vnorm;
        println!("{vnorm}");
        beta.as_mut().write(m, 0, vnorm);
        // q.column_mut(m + 1).assign(&(&v / vnorm));
        zipped!(q.as_mut().col(m + 1), v.as_ref())
            .for_each(|mut qm_el, v_el| qm_el.write(v_el.read() / vnorm));
    }

    // Get eigenpairs from Hessenberg matrix
    let hessenberg = q.as_ref().transpose() * a * q.as_ref();

    let mut mem = GlobalMemBuffer::new(StackReq::any_of([faer_evd::compute_evd_req::<f64>(
        5,
        faer_evd::ComputeVectors::No,
        faer_core::Parallelism::None,
        EvdParams::default(),
    )
    .unwrap()]));
    let mut stack = DynStack::new(&mut mem);

    let mut s = Mat::zeros(iters + 1, 1);
    faer_evd::compute_hermitian_evd(
        hessenberg.as_ref(),
        s.as_mut(),
        None,
        0.01,
        0.001,
        faer_core::Parallelism::None,
        stack,
        SymmetricEvdParams::default(),
    );

    println!("{s:?}");
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_ira() {}

    #[test]
    fn test_lanczos() {
        // Define the input array
        let a = mat![
            [0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 3.0]
        ];

        // println!("{}", a * 1.0);

        lanczos(a.as_ref());
    }
}
