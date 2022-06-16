
def compute_m0_n1_amplitude_optimized(A, N, ansatz, truncation, h_args, t_args, opt_path_lists):
    """Compute the operator(name='b', rank=1, m=0, n=1) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A, N), dtype=complex)

    # unpack the optimized paths
    opt_connected_path_list, opt_linked_path_list, opt_unlinked_path_list = opt_path_lists[(0, 1)]

    # add each of the terms
    add_m0_n1_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_connected_path_list)
    add_m0_n1_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_linked_path_list)
    add_m0_n1_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_unlinked_path_list)
    return R

