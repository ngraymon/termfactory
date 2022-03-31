
def compute_m0_n1_amplitude(A, N, ansatz, truncation, h_args, t_args):
    """Compute the operator(name='b', rank=1, m=0, n=1) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A, N), dtype=complex)

    # add each of the terms
    add_m0_n1_fully_connected_terms(R, ansatz, truncation, h_args, t_args)
    add_m0_n1_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
    add_m0_n1_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
    return R

