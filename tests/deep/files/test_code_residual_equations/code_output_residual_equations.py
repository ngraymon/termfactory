# system imports

# third party imports
import numpy as np

# local imports
from .symmetrize import symmetrize_tensor


def calculate_order_0_residual(A, N, truncation, h_args, w_args):
    """Calculate the 0 order residual as a function of the W operators."""
    h_ab, h_abI, h_abi, h_abIj, h_abIJ, h_abij = h_args
    w_i, w_ij, *unusedargs = w_args

    R = np.zeros((A, A), dtype=complex)

    assert truncation.singles, \
        f"Cannot calculate order 0 residual for {truncation.cc_truncation_order}"

    R += 1.0 * h_ab

    R += 1.0 * np.einsum('acm,cbm->ab', h_abI, w_i)

    if truncation.quadratic:
        if w_ij is not None:
            R += (1/2) * np.einsum('acmn,cbmn->ab', h_abIJ, w_ij)

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

    if truncation.quadratic:
        if w_ijk is not None:
            R += (3/6) * np.einsum('acmn,cbmni->abi', h_abIJ, w_ijk)

    R += 1.0 * h_abi

    R += 1.0 * np.einsum('acmi,cbm->abi', h_abIj, w_i)

    return R


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
