
def compute_m0_n1_amplitude_optimized(A, N, ansatz, truncation, h_args, t_args, opt_paths):
    """Compute the operator(name='b', rank=1, m=0, n=1) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A, N), dtype=complex)

    # unpack the optimized paths
    optimized_connected_paths, optimized_linked_paths, optimized_unlinked_paths = opt_paths

    # add each of the terms
    add_m0_n1_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, optimized_connected_paths)
    add_m0_n1_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, optimized_linked_paths)
    add_m0_n1_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, optimized_unlinked_paths)
    return R

