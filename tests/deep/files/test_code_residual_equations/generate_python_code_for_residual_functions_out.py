
def calculate_order_0_residual(A, N, truncation, h_args, w_args):
    """Calculate the 0 order residual as a function of the W operators."""
    h_ab, h_abI, h_abi, h_abIj, h_abIJ, h_abij = h_args
    w_i, w_ij, *unusedargs = w_args

    R = np.zeros((A, A), dtype=complex)

    assert truncation.singles, \
        f"Cannot calculate order 0 residual for {truncation.cc_truncation_order}"

    R += 1.0 * h_ab

    R += 1.0 * np.einsum('acm,cbm->ab', h_abI, w_i)

    return R


def calculate_order_1_residual(A, N, truncation, h_args, w_args):
    """Calculate the 1 order residual as a function of the W operators."""
    h_ab, h_abI, h_abi, h_abIj, h_abIJ, h_abij = h_args
    w_i, w_ij, w_ijk, *unusedargs = w_args

    R = np.zeros((A, A, N), dtype=complex)

    assert truncation.singles, \
        f"Cannot calculate order 1 residual for {truncation.cc_truncation_order}"

    R += 1.0 * np.einsum('ac,cbi->abi', h_ab, w_i)

    if w_ij is not None:
        R += 1.0 * np.einsum('acm,cbmi->abi', h_abI, w_ij)

    R += 1.0 * h_abi

    return R