# system imports
from math import factorial

# third party imports
import numpy as np
import opt_einsum as oe

# local imports
from .symmetrize import symmetrize_tensor
from ..log_conf import log

# ------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- DEFAULT FUNCTIONS --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #

# --------------------------------------------- INDIVIDUAL TERMS --------------------------------------------- #


# -------------- operator(name='', rank=0, m=0, n=0) TERMS -------------- #
def compute_m0_n0_HZ_LHS(Z, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """ Calculate the operator(name='', rank=0, m=0, n=0) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def compute_m0_n0_eT_HZ_LHS(Z, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """ Calculate the operator(name='', rank=0, m=0, n=0) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_linear:
                Z += np.einsum('i, ci -> c', t_conj[(0, 1)], dz_args[(1, 0)])
            if truncation.z_at_least_quadratic:
                Z += (1 / 2) * np.einsum('i, j, cij -> c', t_conj[(0, 1)], t_conj[(0, 1)], dz_args[(2, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                Z += np.einsum('i, i, c -> c', t_conj[(0, 1)], dT[(1, 0)], z_args[(0, 0)])
                if truncation.z_at_least_linear:
                    Z += np.einsum('i, j, j, ci -> c', t_conj[(0, 1)], t_conj[(0, 1)], dT[(1, 0)], z_args[(1, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                Z += (1 / 2) * np.einsum('i, j, ij, c -> c', t_conj[(0, 1)], t_conj[(0, 1)], dT[(2, 0)], z_args[(0, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  1 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='b', rank=1, m=0, n=1) TERMS -------------- #
def compute_m0_n1_HZ_LHS(Z, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """ Calculate the operator(name='b', rank=1, m=0, n=1) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            Z += np.einsum('z, c -> cz', dT[(1, 0)], z_args[(0, 0)])

    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def compute_m0_n1_eT_HZ_LHS(Z, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """ Calculate the operator(name='b', rank=1, m=0, n=1) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_quadratic:
                Z += np.einsum('i, ciz -> cz', t_conj[(0, 1)], dz_args[(2, 0)])
            if truncation.z_at_least_cubic:
                Z += (1 / 2) * np.einsum('i, j, cijz -> cz', t_conj[(0, 1)], t_conj[(0, 1)], dz_args[(3, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_linear:
                    Z += (
                        np.einsum('i, z, ci -> cz', t_conj[(0, 1)], dT[(1, 0)], z_args[(1, 0)]) +
                        np.einsum('i, i, cz -> cz', t_conj[(0, 1)], dT[(1, 0)], z_args[(1, 0)])
                    )
                if truncation.z_at_least_quadratic:
                    Z += np.einsum('i, j, j, ciz -> cz', t_conj[(0, 1)], t_conj[(0, 1)], dT[(1, 0)], z_args[(2, 0)])
                    Z += (1 / 2) * np.einsum('i, j, z, cij -> cz', t_conj[(0, 1)], t_conj[(0, 1)], dT[(1, 0)], z_args[(2, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                Z += np.einsum('i, iz, c -> cz', t_conj[(0, 1)], dT[(2, 0)], z_args[(0, 0)])
                if truncation.z_at_least_linear:
                    Z += np.einsum('i, j, jz, ci -> cz', t_conj[(0, 1)], t_conj[(0, 1)], dT[(2, 0)], z_args[(1, 0)])
                    Z += (1 / 2) * np.einsum('i, j, ij, cz -> cz', t_conj[(0, 1)], t_conj[(0, 1)], dT[(2, 0)], z_args[(1, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


# --------------------------------------------- RESIDUAL FUNCTIONS --------------------------------------------- #
def compute_m0_n0_amplitude(A, N, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """Compute the operator(name='', rank=0, m=0, n=0) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    dz = np.zeros(shape=(A), dtype=complex)

    # compute the terms
    compute_m0_n0_HZ_LHS(dz, ansatz, truncation, t_conj, dT, z_args, dz_args)
    compute_m0_n0_eT_HZ_LHS(dz, ansatz, truncation, t_conj, dT, z_args, dz_args)
    return dz


def compute_m0_n1_amplitude(A, N, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """Compute the operator(name='b', rank=1, m=0, n=1) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    dz = np.zeros(shape=(A, N), dtype=complex)

    # compute the terms
    compute_m0_n1_HZ_LHS(dz, ansatz, truncation, t_conj, dT, z_args, dz_args)
    compute_m0_n1_eT_HZ_LHS(dz, ansatz, truncation, t_conj, dT, z_args, dz_args)
    return dz

# ------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------- OPTIMIZED FUNCTIONS -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #

# --------------------------------------------- INDIVIDUAL TERMS --------------------------------------------- #


# -------------- operator(name='', rank=0, m=0, n=0) TERMS -------------- #
def compute_m0_n0_HZ_LHS_optimized(Z, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_einsum):
    """ Optimized calculation of the operator(name='', rank=0, m=0, n=0) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def compute_m0_n0_eT_HZ_LHS_optimized(Z, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_einsum):
    """ Optimized calculation of the operator(name='', rank=0, m=0, n=0) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_linear:
                Z += np.einsum('i, ci -> c', t_conj[(0, 1)], dz_args[(1, 0)])
            if truncation.z_at_least_quadratic:
                Z += (1 / 2) * np.einsum('i, j, cij -> c', t_conj[(0, 1)], t_conj[(0, 1)], dz_args[(2, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                Z += np.einsum('i, i, c -> c', t_conj[(0, 1)], dT[(1, 0)], z_args[(0, 0)])
                if truncation.z_at_least_linear:
                    Z += np.einsum('i, j, j, ci -> c', t_conj[(0, 1)], t_conj[(0, 1)], dT[(1, 0)], z_args[(1, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                Z += (1 / 2) * np.einsum('i, j, ij, c -> c', t_conj[(0, 1)], t_conj[(0, 1)], dT[(2, 0)], z_args[(0, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  1 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='b', rank=1, m=0, n=1) TERMS -------------- #
def compute_m0_n1_HZ_LHS_optimized(Z, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_einsum):
    """ Optimized calculation of the operator(name='b', rank=1, m=0, n=1) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            Z += np.einsum('z, c -> cz', dT[(1, 0)], z_args[(0, 0)])

    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def compute_m0_n1_eT_HZ_LHS_optimized(Z, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_einsum):
    """ Optimized calculation of the operator(name='b', rank=1, m=0, n=1) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_quadratic:
                Z += np.einsum('i, ciz -> cz', t_conj[(0, 1)], dz_args[(2, 0)])
            if truncation.z_at_least_cubic:
                Z += (1 / 2) * np.einsum('i, j, cijz -> cz', t_conj[(0, 1)], t_conj[(0, 1)], dz_args[(3, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_linear:
                    Z += (
                        np.einsum('i, z, ci -> cz', t_conj[(0, 1)], dT[(1, 0)], z_args[(1, 0)]) +
                        np.einsum('i, i, cz -> cz', t_conj[(0, 1)], dT[(1, 0)], z_args[(1, 0)])
                    )
                if truncation.z_at_least_quadratic:
                    Z += np.einsum('i, j, j, ciz -> cz', t_conj[(0, 1)], t_conj[(0, 1)], dT[(1, 0)], z_args[(2, 0)])
                    Z += (1 / 2) * np.einsum('i, j, z, cij -> cz', t_conj[(0, 1)], t_conj[(0, 1)], dT[(1, 0)], z_args[(2, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                Z += np.einsum('i, iz, c -> cz', t_conj[(0, 1)], dT[(2, 0)], z_args[(0, 0)])
                if truncation.z_at_least_linear:
                    Z += np.einsum('i, j, jz, ci -> cz', t_conj[(0, 1)], t_conj[(0, 1)], dT[(2, 0)], z_args[(1, 0)])
                    Z += (1 / 2) * np.einsum('i, j, ij, cz -> cz', t_conj[(0, 1)], t_conj[(0, 1)], dT[(2, 0)], z_args[(1, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


# --------------------------------------------- RESIDUAL FUNCTIONS --------------------------------------------- #
def compute_m0_n0_amplitude_optimized(A, N, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_paths):
    """Compute the operator(name='', rank=0, m=0, n=0) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    dz = np.zeros(shape=(A), dtype=complex)

    # unpack the optimized paths
    optimized_HZ_paths, optimized_eT_HZ_paths = opt_paths

    # compute the terms
    compute_m0_n0_HZ_LHS_optimized(dz, ansatz, truncation, t_conj, dT, z_args, dz_args, optimized_HZ_paths)
    compute_m0_n0_eT_HZ_LHS_optimized(dz, ansatz, truncation, t_conj, dT, z_args, dz_args, optimized_eT_HZ_paths)
    return dz


def compute_m0_n1_amplitude_optimized(A, N, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_paths):
    """Compute the operator(name='b', rank=1, m=0, n=1) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    dz = np.zeros(shape=(A, N), dtype=complex)

    # unpack the optimized paths
    optimized_HZ_paths, optimized_eT_HZ_paths = opt_paths

    # compute the terms
    compute_m0_n1_HZ_LHS_optimized(dz, ansatz, truncation, t_conj, dT, z_args, dz_args, optimized_HZ_paths)
    compute_m0_n1_eT_HZ_LHS_optimized(dz, ansatz, truncation, t_conj, dT, z_args, dz_args, optimized_eT_HZ_paths)
    return dz


# --------------------------------------------- OPTIMIZED PATHS FUNCTION --------------------------------------------- #