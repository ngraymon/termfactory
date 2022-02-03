
def calculate_order_2_residual(A, N, truncation, h_args, w_args):
    """Calculate the 2 order residual as a function of the W operators."""
    h_ab, h_abI, h_abi, h_abIj, h_abIJ, h_abij = h_args
    w_i, w_ij, w_ijk, w_ijkl, *unusedargs = w_args

    R = np.zeros((A, A, N, N), dtype=complex)

    assert truncation.doubles, \
        f"Cannot calculate order 2 residual for {truncation.cc_truncation_order}"

    if w_ij is not None:
        R += (1/2) * np.einsum('ac,cbij->abij', h_ab, w_ij)
        R += 1.0 * np.einsum('acmi,cbmj->abij', h_abIj, w_ij)

    if w_ijk is not None:
        R += (3/6) * np.einsum('acm,cbmij->abij', h_abI, w_ijk)

    if truncation.quadratic:
        if w_ijkl is not None:
            R += (6/24) * np.einsum('acmn,cbmnij->abij', h_abIJ, w_ijkl)
        else:
            R += (1/2) * h_abij

    R += 1.0 * np.einsum('aci,cbj->abij', h_abi, w_i)

    return R