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
def add_m0_n0_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args):
    """ Calculate the operator(name='', rank=0, m=0, n=0) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_linear:
                R += np.einsum('aci, ci -> a', h_args[(0, 1)], z_args[(1, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.z_at_least_quadratic:
                R += (1 / 2) * np.einsum('acij, cij -> a', h_args[(0, 2)], z_args[(2, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def add_m0_n0_eT_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args):
    """ Calculate the operator(name='', rank=0, m=0, n=0) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_linear:
                R += np.einsum('i, ac, ci -> a', t_args[(0, 1)], h_args[(0, 0)], z_args[(1, 0)])
            if truncation.z_at_least_quadratic:
                R += (1 / 2) * np.einsum('i, j, ac, cij -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 0)], z_args[(2, 0)])
            if truncation.z_at_least_cubic:
                R += (1 / 6) * np.einsum('i, j, k, ac, cijk -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 0)], z_args[(3, 0)])
            if truncation.z_at_least_quartic:
                R += (1 / 24) * np.einsum('i, j, k, l, ac, cijkl -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 0)], z_args[(4, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                R += np.einsum('i, aci, c -> a', t_args[(0, 1)], h_args[(1, 0)], z_args[(0, 0)])
                if truncation.z_at_least_linear:
                    R += (
                        np.einsum('i, acij, cj -> a', t_args[(0, 1)], h_args[(1, 1)], z_args[(1, 0)]) +
                        np.einsum('i, j, acj, ci -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(1, 0)])
                    )
                if truncation.z_at_least_quadratic:
                    R += (
                        np.einsum('i, acj, cij -> a', t_args[(0, 1)], h_args[(0, 1)], z_args[(2, 0)]) +
                        np.einsum('i, j, acjk, cik -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(2, 0)])
                    )
                    R += (1 / 2) * np.einsum('i, j, k, ack, cij -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                if truncation.z_at_least_cubic:
                    R += (1 / 2) * (
                        np.einsum('i, j, ack, cijk -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 1)], z_args[(3, 0)]) +
                        np.einsum('i, j, k, ackl, cijl -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(3, 0)])
                    )
                    R += (1 / 6) * np.einsum('i, j, k, l, acl, cijk -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 6) * (
                        np.einsum('i, j, k, acl, cijkl -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 1)], z_args[(4, 0)]) +
                        np.einsum('i, j, k, l, aclm, cijkm -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(4, 0)])
                    )

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                R += (1 / 2) * np.einsum('i, j, acij, c -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(0, 0)])
                if truncation.z_at_least_linear:
                    R += (1 / 2) * np.einsum('i, j, k, acjk, ci -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(1, 0)])
                if truncation.z_at_least_quadratic:
                    R += (1 / 4) * np.einsum('i, j, k, l, ackl, cij -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(2, 0)])
                if truncation.z_at_least_cubic:
                    R += (1 / 2) * np.einsum('i, acjk, cijk -> a', t_args[(0, 1)], h_args[(0, 2)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 4) * np.einsum('i, j, ackl, cijkl -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 2)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  1 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='b', rank=1, m=0, n=1) TERMS -------------- #
def add_m0_n1_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args):
    """ Calculate the operator(name='b', rank=1, m=0, n=1) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            R += np.einsum('acz, c -> az', h_args[(1, 0)], z_args[(0, 0)])
            if truncation.z_at_least_linear:
                R += np.einsum('aciz, ci -> az', h_args[(1, 1)], z_args[(1, 0)])
            if truncation.z_at_least_quadratic:
                R += np.einsum('aci, ciz -> az', h_args[(0, 1)], z_args[(2, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.z_at_least_cubic:
                R += (1 / 2) * np.einsum('acij, cijz -> az', h_args[(0, 2)], z_args[(3, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def add_m0_n1_eT_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args):
    """ Calculate the operator(name='b', rank=1, m=0, n=1) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_quadratic:
                R += np.einsum('i, ac, ciz -> az', t_args[(0, 1)], h_args[(0, 0)], z_args[(2, 0)])
            if truncation.z_at_least_cubic:
                R += (1 / 2) * np.einsum('i, j, ac, cijz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 0)], z_args[(3, 0)])
            if truncation.z_at_least_quartic:
                R += (1 / 6) * np.einsum('i, j, k, ac, cijkz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 0)], z_args[(4, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_linear:
                    R += (
                        np.einsum('i, acz, ci -> az', t_args[(0, 1)], h_args[(1, 0)], z_args[(1, 0)]) +
                        np.einsum('i, aci, cz -> az', t_args[(0, 1)], h_args[(1, 0)], z_args[(1, 0)])
                    )
                if truncation.z_at_least_quadratic:
                    R += (
                        np.einsum('i, acij, cjz -> az', t_args[(0, 1)], h_args[(1, 1)], z_args[(2, 0)]) +
                        np.einsum('i, acjz, cij -> az', t_args[(0, 1)], h_args[(1, 1)], z_args[(2, 0)]) +
                        np.einsum('i, j, acj, ciz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                    )
                    R += (1 / 2) * np.einsum('i, j, acz, cij -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                if truncation.z_at_least_cubic:
                    R += (
                        np.einsum('i, acj, cijz -> az', t_args[(0, 1)], h_args[(0, 1)], z_args[(3, 0)]) +
                        np.einsum('i, j, acjk, cikz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(3, 0)])
                    )
                    R += (1 / 2) * (
                        np.einsum('i, j, ackz, cijk -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(3, 0)]) +
                        np.einsum('i, j, k, ack, cijz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                    )
                    R += (1 / 6) * np.einsum('i, j, k, acz, cijk -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 2) * (
                        np.einsum('i, j, ack, cijkz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 1)], z_args[(4, 0)]) +
                        np.einsum('i, j, k, ackl, cijlz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(4, 0)])
                    )
                    R += (1 / 24) * np.einsum('i, j, k, l, acz, cijkl -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
                    R += (1 / 6) * (
                        np.einsum('i, j, k, aclz, cijkl -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(4, 0)]) +
                        np.einsum('i, j, k, l, acl, cijkz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
                    )

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                R += np.einsum('i, aciz, c -> az', t_args[(0, 1)], h_args[(2, 0)], z_args[(0, 0)])
                if truncation.z_at_least_linear:
                    R += np.einsum('i, j, acjz, ci -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(1, 0)])
                    R += (1 / 2) * np.einsum('i, j, acij, cz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(1, 0)])
                if truncation.z_at_least_quadratic:
                    R += (1 / 2) * (
                        np.einsum('i, j, k, acjk, ciz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(2, 0)]) +
                        np.einsum('i, j, k, ackz, cij -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(2, 0)])
                    )
                if truncation.z_at_least_cubic:
                    R += (1 / 4) * np.einsum('i, j, k, l, ackl, cijz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(3, 0)])
                    R += (1 / 6) * np.einsum('i, j, k, l, aclz, cijk -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 2) * np.einsum('i, acjk, cijkz -> az', t_args[(0, 1)], h_args[(0, 2)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  2 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='bb', rank=2, m=0, n=2) TERMS -------------- #
def add_m0_n2_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args):
    """ Calculate the operator(name='bb', rank=2, m=0, n=2) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_linear:
                R += np.einsum('acz, cy -> azy', h_args[(1, 0)], z_args[(1, 0)])
            if truncation.z_at_least_quadratic:
                R += np.einsum('aciz, ciy -> azy', h_args[(1, 1)], z_args[(2, 0)])
            if truncation.z_at_least_cubic:
                R += (1 / 2) * np.einsum('aci, cizy -> azy', h_args[(0, 1)], z_args[(3, 0)])

        if truncation.h_at_least_quadratic:
            R += (1 / 2) * np.einsum('aczy, c -> azy', h_args[(2, 0)], z_args[(0, 0)])
            if truncation.z_at_least_quartic:
                R += (1 / 4) * np.einsum('acij, cijzy -> azy', h_args[(0, 2)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def add_m0_n2_eT_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args):
    """ Calculate the operator(name='bb', rank=2, m=0, n=2) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_cubic:
                R += (1 / 2) * np.einsum('i, ac, cizy -> azy', t_args[(0, 1)], h_args[(0, 0)], z_args[(3, 0)])
            if truncation.z_at_least_quartic:
                R += (1 / 4) * np.einsum('i, j, ac, cijzy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 0)], z_args[(4, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_quadratic:
                    R += np.einsum('i, acz, ciy -> azy', t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                    R += (1 / 2) * np.einsum('i, aci, czy -> azy', t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                if truncation.z_at_least_cubic:
                    R += np.einsum('i, acjz, cijy -> azy', t_args[(0, 1)], h_args[(1, 1)], z_args[(3, 0)])
                    R += (1 / 2) * (
                        np.einsum('i, acij, cjzy -> azy', t_args[(0, 1)], h_args[(1, 1)], z_args[(3, 0)]) +
                        np.einsum('i, j, acj, cizy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)]) +
                        np.einsum('i, j, acz, cijy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                    )
                if truncation.z_at_least_quartic:
                    R += (1 / 2) * (
                        np.einsum('i, acj, cijzy -> azy', t_args[(0, 1)], h_args[(0, 1)], z_args[(4, 0)]) +
                        np.einsum('i, j, acjk, cikzy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(4, 0)]) +
                        np.einsum('i, j, ackz, cijky -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(4, 0)])
                    )
                    R += (1 / 4) * np.einsum('i, j, k, ack, cijzy -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
                    R += (1 / 6) * np.einsum('i, j, k, acz, cijky -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                if truncation.z_at_least_linear:
                    R += np.einsum('i, aciz, cy -> azy', t_args[(0, 1)], h_args[(2, 0)], z_args[(1, 0)])
                    R += (1 / 2) * np.einsum('i, aczy, ci -> azy', t_args[(0, 1)], h_args[(2, 0)], z_args[(1, 0)])
                if truncation.z_at_least_quadratic:
                    R += np.einsum('i, j, acjz, ciy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(2, 0)])
                    R += (1 / 4) * (
                        np.einsum('i, j, aczy, cij -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(2, 0)]) +
                        np.einsum('i, j, acij, czy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(2, 0)])
                    )
                if truncation.z_at_least_cubic:
                    R += (1 / 12) * np.einsum('i, j, k, aczy, cijk -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(3, 0)])
                    R += (1 / 2) * np.einsum('i, j, k, ackz, cijy -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(3, 0)])
                    R += (1 / 4) * np.einsum('i, j, k, acjk, cizy -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 48) * np.einsum('i, j, k, l, aczy, cijkl -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(4, 0)])
                    R += (1 / 6) * np.einsum('i, j, k, l, aclz, cijky -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(4, 0)])
                    R += (1 / 8) * np.einsum('i, j, k, l, ackl, cijzy -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


# --------------------------------------------- RESIDUAL FUNCTIONS --------------------------------------------- #
def compute_m0_n0_amplitude(A, N, ansatz, truncation, t_args, h_args, z_args):
    """Compute the operator(name='', rank=0, m=0, n=0) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A), dtype=complex)

    # add the terms
    add_m0_n0_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args)
    add_m0_n0_eT_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args)
    return R


def compute_m0_n1_amplitude(A, N, ansatz, truncation, t_args, h_args, z_args):
    """Compute the operator(name='b', rank=1, m=0, n=1) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, N), dtype=complex)

    # add the terms
    add_m0_n1_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args)
    add_m0_n1_eT_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args)
    return R


def compute_m0_n2_amplitude(A, N, ansatz, truncation, t_args, h_args, z_args):
    """Compute the operator(name='bb', rank=2, m=0, n=2) amplitude."""
    truncation.confirm_at_least_singles()
    truncation.confirm_at_least_doubles()

    # the residual tensor
    R = np.zeros(shape=(A, N, N), dtype=complex)

    # add the terms
    add_m0_n2_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args)
    add_m0_n2_eT_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args)
    return R

# ------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------- OPTIMIZED FUNCTIONS -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #

# --------------------------------------------- INDIVIDUAL TERMS --------------------------------------------- #


# -------------- operator(name='', rank=0, m=0, n=0) TERMS -------------- #
def add_m0_n0_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, opt_einsum):
    """ Optimized calculation of the operator(name='', rank=0, m=0, n=0) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_linear:
                R += np.einsum('aci, ci -> a', h_args[(0, 1)], z_args[(1, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.z_at_least_quadratic:
                R += (1 / 2) * np.einsum('acij, cij -> a', h_args[(0, 2)], z_args[(2, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def add_m0_n0_eT_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, opt_einsum):
    """ Optimized calculation of the operator(name='', rank=0, m=0, n=0) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_linear:
                R += np.einsum('i, ac, ci -> a', t_args[(0, 1)], h_args[(0, 0)], z_args[(1, 0)])
            if truncation.z_at_least_quadratic:
                R += (1 / 2) * np.einsum('i, j, ac, cij -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 0)], z_args[(2, 0)])
            if truncation.z_at_least_cubic:
                R += (1 / 6) * np.einsum('i, j, k, ac, cijk -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 0)], z_args[(3, 0)])
            if truncation.z_at_least_quartic:
                R += (1 / 24) * np.einsum('i, j, k, l, ac, cijkl -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 0)], z_args[(4, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                R += np.einsum('i, aci, c -> a', t_args[(0, 1)], h_args[(1, 0)], z_args[(0, 0)])
                if truncation.z_at_least_linear:
                    R += (
                        np.einsum('i, acij, cj -> a', t_args[(0, 1)], h_args[(1, 1)], z_args[(1, 0)]) +
                        np.einsum('i, j, acj, ci -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(1, 0)])
                    )
                if truncation.z_at_least_quadratic:
                    R += (
                        np.einsum('i, acj, cij -> a', t_args[(0, 1)], h_args[(0, 1)], z_args[(2, 0)]) +
                        np.einsum('i, j, acjk, cik -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(2, 0)])
                    )
                    R += (1 / 2) * np.einsum('i, j, k, ack, cij -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                if truncation.z_at_least_cubic:
                    R += (1 / 2) * (
                        np.einsum('i, j, ack, cijk -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 1)], z_args[(3, 0)]) +
                        np.einsum('i, j, k, ackl, cijl -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(3, 0)])
                    )
                    R += (1 / 6) * np.einsum('i, j, k, l, acl, cijk -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 6) * (
                        np.einsum('i, j, k, acl, cijkl -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 1)], z_args[(4, 0)]) +
                        np.einsum('i, j, k, l, aclm, cijkm -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(4, 0)])
                    )

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                R += (1 / 2) * np.einsum('i, j, acij, c -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(0, 0)])
                if truncation.z_at_least_linear:
                    R += (1 / 2) * np.einsum('i, j, k, acjk, ci -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(1, 0)])
                if truncation.z_at_least_quadratic:
                    R += (1 / 4) * np.einsum('i, j, k, l, ackl, cij -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(2, 0)])
                if truncation.z_at_least_cubic:
                    R += (1 / 2) * np.einsum('i, acjk, cijk -> a', t_args[(0, 1)], h_args[(0, 2)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 4) * np.einsum('i, j, ackl, cijkl -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 2)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  1 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='b', rank=1, m=0, n=1) TERMS -------------- #
def add_m0_n1_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, opt_einsum):
    """ Optimized calculation of the operator(name='b', rank=1, m=0, n=1) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            R += np.einsum('acz, c -> az', h_args[(1, 0)], z_args[(0, 0)])
            if truncation.z_at_least_linear:
                R += np.einsum('aciz, ci -> az', h_args[(1, 1)], z_args[(1, 0)])
            if truncation.z_at_least_quadratic:
                R += np.einsum('aci, ciz -> az', h_args[(0, 1)], z_args[(2, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.z_at_least_cubic:
                R += (1 / 2) * np.einsum('acij, cijz -> az', h_args[(0, 2)], z_args[(3, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def add_m0_n1_eT_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, opt_einsum):
    """ Optimized calculation of the operator(name='b', rank=1, m=0, n=1) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_quadratic:
                R += np.einsum('i, ac, ciz -> az', t_args[(0, 1)], h_args[(0, 0)], z_args[(2, 0)])
            if truncation.z_at_least_cubic:
                R += (1 / 2) * np.einsum('i, j, ac, cijz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 0)], z_args[(3, 0)])
            if truncation.z_at_least_quartic:
                R += (1 / 6) * np.einsum('i, j, k, ac, cijkz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 0)], z_args[(4, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_linear:
                    R += (
                        np.einsum('i, acz, ci -> az', t_args[(0, 1)], h_args[(1, 0)], z_args[(1, 0)]) +
                        np.einsum('i, aci, cz -> az', t_args[(0, 1)], h_args[(1, 0)], z_args[(1, 0)])
                    )
                if truncation.z_at_least_quadratic:
                    R += (
                        np.einsum('i, acij, cjz -> az', t_args[(0, 1)], h_args[(1, 1)], z_args[(2, 0)]) +
                        np.einsum('i, acjz, cij -> az', t_args[(0, 1)], h_args[(1, 1)], z_args[(2, 0)]) +
                        np.einsum('i, j, acj, ciz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                    )
                    R += (1 / 2) * np.einsum('i, j, acz, cij -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                if truncation.z_at_least_cubic:
                    R += (
                        np.einsum('i, acj, cijz -> az', t_args[(0, 1)], h_args[(0, 1)], z_args[(3, 0)]) +
                        np.einsum('i, j, acjk, cikz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(3, 0)])
                    )
                    R += (1 / 2) * (
                        np.einsum('i, j, ackz, cijk -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(3, 0)]) +
                        np.einsum('i, j, k, ack, cijz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                    )
                    R += (1 / 6) * np.einsum('i, j, k, acz, cijk -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 2) * (
                        np.einsum('i, j, ack, cijkz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 1)], z_args[(4, 0)]) +
                        np.einsum('i, j, k, ackl, cijlz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(4, 0)])
                    )
                    R += (1 / 24) * np.einsum('i, j, k, l, acz, cijkl -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
                    R += (1 / 6) * (
                        np.einsum('i, j, k, aclz, cijkl -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(4, 0)]) +
                        np.einsum('i, j, k, l, acl, cijkz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
                    )

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                R += np.einsum('i, aciz, c -> az', t_args[(0, 1)], h_args[(2, 0)], z_args[(0, 0)])
                if truncation.z_at_least_linear:
                    R += np.einsum('i, j, acjz, ci -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(1, 0)])
                    R += (1 / 2) * np.einsum('i, j, acij, cz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(1, 0)])
                if truncation.z_at_least_quadratic:
                    R += (1 / 2) * (
                        np.einsum('i, j, k, acjk, ciz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(2, 0)]) +
                        np.einsum('i, j, k, ackz, cij -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(2, 0)])
                    )
                if truncation.z_at_least_cubic:
                    R += (1 / 4) * np.einsum('i, j, k, l, ackl, cijz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(3, 0)])
                    R += (1 / 6) * np.einsum('i, j, k, l, aclz, cijk -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 2) * np.einsum('i, acjk, cijkz -> az', t_args[(0, 1)], h_args[(0, 2)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  2 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='bb', rank=2, m=0, n=2) TERMS -------------- #
def add_m0_n2_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, opt_einsum):
    """ Optimized calculation of the operator(name='bb', rank=2, m=0, n=2) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_linear:
                R += np.einsum('acz, cy -> azy', h_args[(1, 0)], z_args[(1, 0)])
            if truncation.z_at_least_quadratic:
                R += np.einsum('aciz, ciy -> azy', h_args[(1, 1)], z_args[(2, 0)])
            if truncation.z_at_least_cubic:
                R += (1 / 2) * np.einsum('aci, cizy -> azy', h_args[(0, 1)], z_args[(3, 0)])

        if truncation.h_at_least_quadratic:
            R += (1 / 2) * np.einsum('aczy, c -> azy', h_args[(2, 0)], z_args[(0, 0)])
            if truncation.z_at_least_quartic:
                R += (1 / 4) * np.einsum('acij, cijzy -> azy', h_args[(0, 2)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def add_m0_n2_eT_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, opt_einsum):
    """ Optimized calculation of the operator(name='bb', rank=2, m=0, n=2) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_cubic:
                R += (1 / 2) * np.einsum('i, ac, cizy -> azy', t_args[(0, 1)], h_args[(0, 0)], z_args[(3, 0)])
            if truncation.z_at_least_quartic:
                R += (1 / 4) * np.einsum('i, j, ac, cijzy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 0)], z_args[(4, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_quadratic:
                    R += np.einsum('i, acz, ciy -> azy', t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                    R += (1 / 2) * np.einsum('i, aci, czy -> azy', t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                if truncation.z_at_least_cubic:
                    R += np.einsum('i, acjz, cijy -> azy', t_args[(0, 1)], h_args[(1, 1)], z_args[(3, 0)])
                    R += (1 / 2) * (
                        np.einsum('i, acij, cjzy -> azy', t_args[(0, 1)], h_args[(1, 1)], z_args[(3, 0)]) +
                        np.einsum('i, j, acj, cizy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)]) +
                        np.einsum('i, j, acz, cijy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                    )
                if truncation.z_at_least_quartic:
                    R += (1 / 2) * (
                        np.einsum('i, acj, cijzy -> azy', t_args[(0, 1)], h_args[(0, 1)], z_args[(4, 0)]) +
                        np.einsum('i, j, acjk, cikzy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(4, 0)]) +
                        np.einsum('i, j, ackz, cijky -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 1)], z_args[(4, 0)])
                    )
                    R += (1 / 4) * np.einsum('i, j, k, ack, cijzy -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
                    R += (1 / 6) * np.einsum('i, j, k, acz, cijky -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                if truncation.z_at_least_linear:
                    R += np.einsum('i, aciz, cy -> azy', t_args[(0, 1)], h_args[(2, 0)], z_args[(1, 0)])
                    R += (1 / 2) * np.einsum('i, aczy, ci -> azy', t_args[(0, 1)], h_args[(2, 0)], z_args[(1, 0)])
                if truncation.z_at_least_quadratic:
                    R += np.einsum('i, j, acjz, ciy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(2, 0)])
                    R += (1 / 4) * (
                        np.einsum('i, j, aczy, cij -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(2, 0)]) +
                        np.einsum('i, j, acij, czy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(2, 0)])
                    )
                if truncation.z_at_least_cubic:
                    R += (1 / 12) * np.einsum('i, j, k, aczy, cijk -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(3, 0)])
                    R += (1 / 2) * np.einsum('i, j, k, ackz, cijy -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(3, 0)])
                    R += (1 / 4) * np.einsum('i, j, k, acjk, cizy -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 48) * np.einsum('i, j, k, l, aczy, cijkl -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(4, 0)])
                    R += (1 / 6) * np.einsum('i, j, k, l, aclz, cijky -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(4, 0)])
                    R += (1 / 8) * np.einsum('i, j, k, l, ackl, cijzy -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(2, 0)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


# --------------------------------------------- RESIDUAL FUNCTIONS --------------------------------------------- #
def compute_m0_n0_amplitude_optimized(A, N, ansatz, truncation, t_args, h_args, z_args, opt_paths):
    """Compute the operator(name='', rank=0, m=0, n=0) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A), dtype=complex)

    # unpack the optimized paths
    optimized_HZ_paths, optimized_eT_HZ_paths = opt_paths

    # add the terms
    add_m0_n0_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, optimized_HZ_paths)
    add_m0_n0_eT_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, optimized_eT_HZ_paths)
    return R


def compute_m0_n1_amplitude_optimized(A, N, ansatz, truncation, t_args, h_args, z_args, opt_paths):
    """Compute the operator(name='b', rank=1, m=0, n=1) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, N), dtype=complex)

    # unpack the optimized paths
    optimized_HZ_paths, optimized_eT_HZ_paths = opt_paths

    # add the terms
    add_m0_n1_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, optimized_HZ_paths)
    add_m0_n1_eT_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, optimized_eT_HZ_paths)
    return R


def compute_m0_n2_amplitude_optimized(A, N, ansatz, truncation, t_args, h_args, z_args, opt_paths):
    """Compute the operator(name='bb', rank=2, m=0, n=2) amplitude."""
    truncation.confirm_at_least_singles()
    truncation.confirm_at_least_doubles()

    # the residual tensor
    R = np.zeros(shape=(A, N, N), dtype=complex)

    # unpack the optimized paths
    optimized_HZ_paths, optimized_eT_HZ_paths = opt_paths

    # add the terms
    add_m0_n2_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, optimized_HZ_paths)
    add_m0_n2_eT_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, optimized_eT_HZ_paths)
    return R


# --------------------------------------------- OPTIMIZED PATHS FUNCTION --------------------------------------------- #