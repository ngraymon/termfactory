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

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                R += np.einsum('i, aci, c -> a', t_args[(0, 1)], h_args[(1, 0)], z_args[(0, 0)])
                if truncation.z_at_least_linear:
                    R += np.einsum('i, j, acj, ci -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(1, 0)])
                if truncation.z_at_least_quadratic:
                    R += np.einsum('i, acj, cij -> a', t_args[(0, 1)], h_args[(0, 1)], z_args[(2, 0)])
                    R += (1 / 2) * np.einsum('i, j, k, ack, cij -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                if truncation.z_at_least_cubic:
                    R += (1 / 2) * np.einsum('i, j, ack, cijk -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 1)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 6) * np.einsum('i, j, k, acl, cijkl -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 1)], z_args[(4, 0)])
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
            if truncation.z_at_least_quadratic:
                R += np.einsum('aci, ciz -> az', h_args[(0, 1)], z_args[(2, 0)])
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
                    R += np.einsum('i, j, acj, ciz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                    R += (1 / 2) * np.einsum('i, j, acz, cij -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                if truncation.z_at_least_cubic:
                    R += np.einsum('i, acj, cijz -> az', t_args[(0, 1)], h_args[(0, 1)], z_args[(3, 0)])
                    R += (1 / 2) * np.einsum('i, j, k, ack, cijz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                    R += (1 / 6) * np.einsum('i, j, k, acz, cijk -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 2) * np.einsum('i, j, ack, cijkz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 1)], z_args[(4, 0)])
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
            if truncation.z_at_least_cubic:
                R += (1 / 2) * np.einsum('aci, cizy -> azy', h_args[(0, 1)], z_args[(3, 0)])
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
                    R += (1 / 2) * (
                        np.einsum('i, j, acj, cizy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)]) +
                        np.einsum('i, j, acz, cijy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                    )
                if truncation.z_at_least_quartic:
                    R += (1 / 2) * np.einsum('i, acj, cijzy -> azy', t_args[(0, 1)], h_args[(0, 1)], z_args[(4, 0)])
                    R += (1 / 4) * np.einsum('i, j, k, ack, cijzy -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
                    R += (1 / 6) * np.einsum('i, j, k, acz, cijky -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  3 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='bbb', rank=3, m=0, n=3) TERMS -------------- #
def add_m0_n3_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args):
    """ Calculate the operator(name='bbb', rank=3, m=0, n=3) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_quadratic:
                R += (1 / 2) * np.einsum('acz, cyx -> azyx', h_args[(1, 0)], z_args[(2, 0)])
            if truncation.z_at_least_quartic:
                R += (1 / 6) * np.einsum('aci, cizyx -> azyx', h_args[(0, 1)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def add_m0_n3_eT_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args):
    """ Calculate the operator(name='bbb', rank=3, m=0, n=3) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_quartic:
                R += (1 / 6) * np.einsum('i, ac, cizyx -> azyx', t_args[(0, 1)], h_args[(0, 0)], z_args[(4, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_cubic:
                    R += (1 / 2) * np.einsum('i, acz, ciyx -> azyx', t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                    R += (1 / 6) * np.einsum('i, aci, czyx -> azyx', t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 4) * np.einsum('i, j, acz, cijyx -> azyx', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
                    R += (1 / 6) * np.einsum('i, j, acj, cizyx -> azyx', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  4 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='bbbb', rank=4, m=0, n=4) TERMS -------------- #
def add_m0_n4_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args):
    """ Calculate the operator(name='bbbb', rank=4, m=0, n=4) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_cubic:
                R += (1 / 6) * np.einsum('acz, cyxw -> azyxw', h_args[(1, 0)], z_args[(3, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def add_m0_n4_eT_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args):
    """ Calculate the operator(name='bbbb', rank=4, m=0, n=4) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_quartic:
                    R += (1 / 24) * np.einsum('i, aci, czyxw -> azyxw', t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
                    R += (1 / 6) * np.einsum('i, acz, ciyxw -> azyxw', t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
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


def compute_m0_n3_amplitude(A, N, ansatz, truncation, t_args, h_args, z_args):
    """Compute the operator(name='bbb', rank=3, m=0, n=3) amplitude."""
    truncation.confirm_at_least_singles()
    truncation.confirm_at_least_doubles()
    truncation.confirm_at_least_triples()

    # the residual tensor
    R = np.zeros(shape=(A, N, N, N), dtype=complex)

    # add the terms
    add_m0_n3_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args)
    add_m0_n3_eT_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args)
    return R


def compute_m0_n4_amplitude(A, N, ansatz, truncation, t_args, h_args, z_args):
    """Compute the operator(name='bbbb', rank=4, m=0, n=4) amplitude."""
    truncation.confirm_at_least_singles()
    truncation.confirm_at_least_doubles()
    truncation.confirm_at_least_triples()
    truncation.confirm_at_least_quadruples()

    # the residual tensor
    R = np.zeros(shape=(A, N, N, N, N), dtype=complex)

    # add the terms
    add_m0_n4_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args)
    add_m0_n4_eT_HZ_terms(R, ansatz, truncation, t_args, h_args, z_args)
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

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                R += np.einsum('i, aci, c -> a', t_args[(0, 1)], h_args[(1, 0)], z_args[(0, 0)])
                if truncation.z_at_least_linear:
                    R += np.einsum('i, j, acj, ci -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(1, 0)])
                if truncation.z_at_least_quadratic:
                    R += np.einsum('i, acj, cij -> a', t_args[(0, 1)], h_args[(0, 1)], z_args[(2, 0)])
                    R += (1 / 2) * np.einsum('i, j, k, ack, cij -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                if truncation.z_at_least_cubic:
                    R += (1 / 2) * np.einsum('i, j, ack, cijk -> a', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 1)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 6) * np.einsum('i, j, k, acl, cijkl -> a', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 1)], z_args[(4, 0)])
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
            if truncation.z_at_least_quadratic:
                R += np.einsum('aci, ciz -> az', h_args[(0, 1)], z_args[(2, 0)])
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
                    R += np.einsum('i, j, acj, ciz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                    R += (1 / 2) * np.einsum('i, j, acz, cij -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(2, 0)])
                if truncation.z_at_least_cubic:
                    R += np.einsum('i, acj, cijz -> az', t_args[(0, 1)], h_args[(0, 1)], z_args[(3, 0)])
                    R += (1 / 2) * np.einsum('i, j, k, ack, cijz -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                    R += (1 / 6) * np.einsum('i, j, k, acz, cijk -> az', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 2) * np.einsum('i, j, ack, cijkz -> az', t_args[(0, 1)], t_args[(0, 1)], h_args[(0, 1)], z_args[(4, 0)])
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
            if truncation.z_at_least_cubic:
                R += (1 / 2) * np.einsum('aci, cizy -> azy', h_args[(0, 1)], z_args[(3, 0)])
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
                    R += (1 / 2) * (
                        np.einsum('i, j, acj, cizy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)]) +
                        np.einsum('i, j, acz, cijy -> azy', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                    )
                if truncation.z_at_least_quartic:
                    R += (1 / 2) * np.einsum('i, acj, cijzy -> azy', t_args[(0, 1)], h_args[(0, 1)], z_args[(4, 0)])
                    R += (1 / 4) * np.einsum('i, j, k, ack, cijzy -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
                    R += (1 / 6) * np.einsum('i, j, k, acz, cijky -> azy', t_args[(0, 1)], t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  3 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='bbb', rank=3, m=0, n=3) TERMS -------------- #
def add_m0_n3_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, opt_einsum):
    """ Optimized calculation of the operator(name='bbb', rank=3, m=0, n=3) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_quadratic:
                R += (1 / 2) * np.einsum('acz, cyx -> azyx', h_args[(1, 0)], z_args[(2, 0)])
            if truncation.z_at_least_quartic:
                R += (1 / 6) * np.einsum('aci, cizyx -> azyx', h_args[(0, 1)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def add_m0_n3_eT_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, opt_einsum):
    """ Optimized calculation of the operator(name='bbb', rank=3, m=0, n=3) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_quartic:
                R += (1 / 6) * np.einsum('i, ac, cizyx -> azyx', t_args[(0, 1)], h_args[(0, 0)], z_args[(4, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_cubic:
                    R += (1 / 2) * np.einsum('i, acz, ciyx -> azyx', t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                    R += (1 / 6) * np.einsum('i, aci, czyx -> azyx', t_args[(0, 1)], h_args[(1, 0)], z_args[(3, 0)])
                if truncation.z_at_least_quartic:
                    R += (1 / 4) * np.einsum('i, j, acz, cijyx -> azyx', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
                    R += (1 / 6) * np.einsum('i, j, acj, cizyx -> azyx', t_args[(0, 1)], t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  4 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='bbbb', rank=4, m=0, n=4) TERMS -------------- #
def add_m0_n4_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, opt_einsum):
    """ Optimized calculation of the operator(name='bbbb', rank=4, m=0, n=4) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_cubic:
                R += (1 / 6) * np.einsum('acz, cyxw -> azyxw', h_args[(1, 0)], z_args[(3, 0)])
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented!')

    return


def add_m0_n4_eT_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, opt_einsum):
    """ Optimized calculation of the operator(name='bbbb', rank=4, m=0, n=4) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_quartic:
                    R += (1 / 24) * np.einsum('i, aci, czyxw -> azyxw', t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
                    R += (1 / 6) * np.einsum('i, acz, ciyxw -> azyxw', t_args[(0, 1)], h_args[(1, 0)], z_args[(4, 0)])
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


def compute_m0_n3_amplitude_optimized(A, N, ansatz, truncation, t_args, h_args, z_args, opt_paths):
    """Compute the operator(name='bbb', rank=3, m=0, n=3) amplitude."""
    truncation.confirm_at_least_singles()
    truncation.confirm_at_least_doubles()
    truncation.confirm_at_least_triples()

    # the residual tensor
    R = np.zeros(shape=(A, N, N, N), dtype=complex)

    # unpack the optimized paths
    optimized_HZ_paths, optimized_eT_HZ_paths = opt_paths

    # add the terms
    add_m0_n3_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, optimized_HZ_paths)
    add_m0_n3_eT_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, optimized_eT_HZ_paths)
    return R


def compute_m0_n4_amplitude_optimized(A, N, ansatz, truncation, t_args, h_args, z_args, opt_paths):
    """Compute the operator(name='bbbb', rank=4, m=0, n=4) amplitude."""
    truncation.confirm_at_least_singles()
    truncation.confirm_at_least_doubles()
    truncation.confirm_at_least_triples()
    truncation.confirm_at_least_quadruples()

    # the residual tensor
    R = np.zeros(shape=(A, N, N, N, N), dtype=complex)

    # unpack the optimized paths
    optimized_HZ_paths, optimized_eT_HZ_paths = opt_paths

    # add the terms
    add_m0_n4_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, optimized_HZ_paths)
    add_m0_n4_eT_HZ_terms_optimized(R, ansatz, truncation, t_args, h_args, z_args, optimized_eT_HZ_paths)
    return R


# --------------------------------------------- OPTIMIZED PATHS FUNCTION --------------------------------------------- #