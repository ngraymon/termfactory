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
        raise Exception('Hot Band amplitudes not implemented!')

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

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                Z += np.einsum('i, i, c -> c', t_conj[(0, 1)], dT[(1, 0)], z_args[(0, 0)])

    else:
        raise Exception('Hot Band amplitudes not implemented!')

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
        raise Exception('Hot Band amplitudes not implemented!')

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

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_linear:
                    Z += (
                        np.einsum('i, z, ci -> cz', t_conj[(0, 1)], dT[(1, 0)], z_args[(1, 0)]) +
                        np.einsum('i, i, cz -> cz', t_conj[(0, 1)], dT[(1, 0)], z_args[(1, 0)])
                    )

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                Z += np.einsum('i, iz, c -> cz', t_conj[(0, 1)], dT[(2, 0)], z_args[(0, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  2 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='bb', rank=2, m=0, n=2) TERMS -------------- #
def compute_m0_n2_HZ_LHS(Z, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """ Calculate the operator(name='bb', rank=2, m=0, n=2) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_linear:
                Z += np.einsum('z, cy -> czy', dT[(1, 0)], z_args[(1, 0)])

        if truncation.h_at_least_quadratic:
            Z += (1 / 2) * np.einsum('zy, c -> czy', dT[(2, 0)], z_args[(0, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented!')

    return


def compute_m0_n2_eT_HZ_LHS(Z, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """ Calculate the operator(name='bb', rank=2, m=0, n=2) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_cubic:
                Z += (1 / 2) * np.einsum('i, cizy -> czy', t_conj[(0, 1)], dz_args[(3, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_quadratic:
                    Z += np.einsum('i, z, ciy -> czy', t_conj[(0, 1)], dT[(1, 0)], z_args[(2, 0)])
                    Z += (1 / 2) * np.einsum('i, i, czy -> czy', t_conj[(0, 1)], dT[(1, 0)], z_args[(2, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                if truncation.z_at_least_linear:
                    Z += np.einsum('i, iz, cy -> czy', t_conj[(0, 1)], dT[(2, 0)], z_args[(1, 0)])
                    Z += (1 / 2) * np.einsum('i, zy, ci -> czy', t_conj[(0, 1)], dT[(2, 0)], z_args[(1, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  3 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='bbb', rank=3, m=0, n=3) TERMS -------------- #
def compute_m0_n3_HZ_LHS(Z, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """ Calculate the operator(name='bbb', rank=3, m=0, n=3) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_quadratic:
                Z += (1 / 2) * np.einsum('z, cyx -> czyx', dT[(1, 0)], z_args[(2, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.z_at_least_linear:
                Z += (1 / 2) * np.einsum('zy, cx -> czyx', dT[(2, 0)], z_args[(1, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented!')

    return


def compute_m0_n3_eT_HZ_LHS(Z, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """ Calculate the operator(name='bbb', rank=3, m=0, n=3) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_quartic:
                Z += (1 / 6) * np.einsum('i, cizyx -> czyx', t_conj[(0, 1)], dz_args[(4, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_cubic:
                    Z += (1 / 2) * np.einsum('i, z, ciyx -> czyx', t_conj[(0, 1)], dT[(1, 0)], z_args[(3, 0)])
                    Z += (1 / 6) * np.einsum('i, i, czyx -> czyx', t_conj[(0, 1)], dT[(1, 0)], z_args[(3, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                if truncation.z_at_least_quadratic:
                    Z += (1 / 2) * (
                        np.einsum('i, zy, cix -> czyx', t_conj[(0, 1)], dT[(2, 0)], z_args[(2, 0)]) +
                        np.einsum('i, iz, cyx -> czyx', t_conj[(0, 1)], dT[(2, 0)], z_args[(2, 0)])
                    )
    else:
        raise Exception('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  4 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='bbbb', rank=4, m=0, n=4) TERMS -------------- #
def compute_m0_n4_HZ_LHS(Z, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """ Calculate the operator(name='bbbb', rank=4, m=0, n=4) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_cubic:
                Z += (1 / 6) * np.einsum('z, cyxw -> czyxw', dT[(1, 0)], z_args[(3, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.z_at_least_quadratic:
                Z += (1 / 4) * np.einsum('zy, cxw -> czyxw', dT[(2, 0)], z_args[(2, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented!')

    return


def compute_m0_n4_eT_HZ_LHS(Z, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """ Calculate the operator(name='bbbb', rank=4, m=0, n=4) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_quartic:
                    Z += (1 / 24) * np.einsum('i, i, czyxw -> czyxw', t_conj[(0, 1)], dT[(1, 0)], z_args[(4, 0)])
                    Z += (1 / 6) * np.einsum('i, z, ciyxw -> czyxw', t_conj[(0, 1)], dT[(1, 0)], z_args[(4, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                if truncation.z_at_least_cubic:
                    Z += (1 / 4) * np.einsum('i, zy, cixw -> czyxw', t_conj[(0, 1)], dT[(2, 0)], z_args[(3, 0)])
                    Z += (1 / 6) * np.einsum('i, iz, cyxw -> czyxw', t_conj[(0, 1)], dT[(2, 0)], z_args[(3, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented!')

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


def compute_m0_n2_amplitude(A, N, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """Compute the operator(name='bb', rank=2, m=0, n=2) amplitude."""
    truncation.confirm_at_least_singles()
    truncation.confirm_at_least_doubles()

    # the residual tensor
    dz = np.zeros(shape=(A, N, N), dtype=complex)

    # compute the terms
    compute_m0_n2_HZ_LHS(dz, ansatz, truncation, t_conj, dT, z_args, dz_args)
    compute_m0_n2_eT_HZ_LHS(dz, ansatz, truncation, t_conj, dT, z_args, dz_args)
    return dz


def compute_m0_n3_amplitude(A, N, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """Compute the operator(name='bbb', rank=3, m=0, n=3) amplitude."""
    truncation.confirm_at_least_singles()
    truncation.confirm_at_least_doubles()
    truncation.confirm_at_least_triples()

    # the residual tensor
    dz = np.zeros(shape=(A, N, N, N), dtype=complex)

    # compute the terms
    compute_m0_n3_HZ_LHS(dz, ansatz, truncation, t_conj, dT, z_args, dz_args)
    compute_m0_n3_eT_HZ_LHS(dz, ansatz, truncation, t_conj, dT, z_args, dz_args)
    return dz


def compute_m0_n4_amplitude(A, N, ansatz, truncation, t_conj, dT, z_args, dz_args):
    """Compute the operator(name='bbbb', rank=4, m=0, n=4) amplitude."""
    truncation.confirm_at_least_singles()
    truncation.confirm_at_least_doubles()
    truncation.confirm_at_least_triples()
    truncation.confirm_at_least_quadruples()

    # the residual tensor
    dz = np.zeros(shape=(A, N, N, N, N), dtype=complex)

    # compute the terms
    compute_m0_n4_HZ_LHS(dz, ansatz, truncation, t_conj, dT, z_args, dz_args)
    compute_m0_n4_eT_HZ_LHS(dz, ansatz, truncation, t_conj, dT, z_args, dz_args)
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
        raise Exception('Hot Band amplitudes not implemented!')

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

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                Z += np.einsum('i, i, c -> c', t_conj[(0, 1)], dT[(1, 0)], z_args[(0, 0)])

    else:
        raise Exception('Hot Band amplitudes not implemented!')

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
        raise Exception('Hot Band amplitudes not implemented!')

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

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_linear:
                    Z += (
                        np.einsum('i, z, ci -> cz', t_conj[(0, 1)], dT[(1, 0)], z_args[(1, 0)]) +
                        np.einsum('i, i, cz -> cz', t_conj[(0, 1)], dT[(1, 0)], z_args[(1, 0)])
                    )

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                Z += np.einsum('i, iz, c -> cz', t_conj[(0, 1)], dT[(2, 0)], z_args[(0, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  2 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='bb', rank=2, m=0, n=2) TERMS -------------- #
def compute_m0_n2_HZ_LHS_optimized(Z, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_einsum):
    """ Optimized calculation of the operator(name='bb', rank=2, m=0, n=2) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_linear:
                Z += np.einsum('z, cy -> czy', dT[(1, 0)], z_args[(1, 0)])

        if truncation.h_at_least_quadratic:
            Z += (1 / 2) * np.einsum('zy, c -> czy', dT[(2, 0)], z_args[(0, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented!')

    return


def compute_m0_n2_eT_HZ_LHS_optimized(Z, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_einsum):
    """ Optimized calculation of the operator(name='bb', rank=2, m=0, n=2) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_cubic:
                Z += (1 / 2) * np.einsum('i, cizy -> czy', t_conj[(0, 1)], dz_args[(3, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_quadratic:
                    Z += np.einsum('i, z, ciy -> czy', t_conj[(0, 1)], dT[(1, 0)], z_args[(2, 0)])
                    Z += (1 / 2) * np.einsum('i, i, czy -> czy', t_conj[(0, 1)], dT[(1, 0)], z_args[(2, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                if truncation.z_at_least_linear:
                    Z += np.einsum('i, iz, cy -> czy', t_conj[(0, 1)], dT[(2, 0)], z_args[(1, 0)])
                    Z += (1 / 2) * np.einsum('i, zy, ci -> czy', t_conj[(0, 1)], dT[(2, 0)], z_args[(1, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  3 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='bbb', rank=3, m=0, n=3) TERMS -------------- #
def compute_m0_n3_HZ_LHS_optimized(Z, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_einsum):
    """ Optimized calculation of the operator(name='bbb', rank=3, m=0, n=3) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_quadratic:
                Z += (1 / 2) * np.einsum('z, cyx -> czyx', dT[(1, 0)], z_args[(2, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.z_at_least_linear:
                Z += (1 / 2) * np.einsum('zy, cx -> czyx', dT[(2, 0)], z_args[(1, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented!')

    return


def compute_m0_n3_eT_HZ_LHS_optimized(Z, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_einsum):
    """ Optimized calculation of the operator(name='bbb', rank=3, m=0, n=3) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.t_singles:
            if truncation.z_at_least_quartic:
                Z += (1 / 6) * np.einsum('i, cizyx -> czyx', t_conj[(0, 1)], dz_args[(4, 0)])

        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_cubic:
                    Z += (1 / 2) * np.einsum('i, z, ciyx -> czyx', t_conj[(0, 1)], dT[(1, 0)], z_args[(3, 0)])
                    Z += (1 / 6) * np.einsum('i, i, czyx -> czyx', t_conj[(0, 1)], dT[(1, 0)], z_args[(3, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                if truncation.z_at_least_quadratic:
                    Z += (1 / 2) * (
                        np.einsum('i, zy, cix -> czyx', t_conj[(0, 1)], dT[(2, 0)], z_args[(2, 0)]) +
                        np.einsum('i, iz, cyx -> czyx', t_conj[(0, 1)], dT[(2, 0)], z_args[(2, 0)])
                    )
    else:
        raise Exception('Hot Band amplitudes not implemented!')

    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  4 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='bbbb', rank=4, m=0, n=4) TERMS -------------- #
def compute_m0_n4_HZ_LHS_optimized(Z, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_einsum):
    """ Optimized calculation of the operator(name='bbbb', rank=4, m=0, n=4) HZ terms.
    These terms have no vibrational contribution from the e^T operator.
    This reduces the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.z_at_least_cubic:
                Z += (1 / 6) * np.einsum('z, cyxw -> czyxw', dT[(1, 0)], z_args[(3, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.z_at_least_quadratic:
                Z += (1 / 4) * np.einsum('zy, cxw -> czyxw', dT[(2, 0)], z_args[(2, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented!')

    return


def compute_m0_n4_eT_HZ_LHS_optimized(Z, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_einsum):
    """ Optimized calculation of the operator(name='bbbb', rank=4, m=0, n=4) eT_HZ terms.
    These terms include the vibrational contributions from the e^T operator.
    This increases the number of possible non-zero permutations of creation/annihilation operators.
    """

    if ansatz.ground_state:
        if truncation.h_at_least_linear:
            if truncation.t_singles:
                if truncation.z_at_least_quartic:
                    Z += (1 / 24) * np.einsum('i, i, czyxw -> czyxw', t_conj[(0, 1)], dT[(1, 0)], z_args[(4, 0)])
                    Z += (1 / 6) * np.einsum('i, z, ciyxw -> czyxw', t_conj[(0, 1)], dT[(1, 0)], z_args[(4, 0)])

        if truncation.h_at_least_quadratic:
            if truncation.t_singles:
                if truncation.z_at_least_cubic:
                    Z += (1 / 4) * np.einsum('i, zy, cixw -> czyxw', t_conj[(0, 1)], dT[(2, 0)], z_args[(3, 0)])
                    Z += (1 / 6) * np.einsum('i, iz, cyxw -> czyxw', t_conj[(0, 1)], dT[(2, 0)], z_args[(3, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented!')

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


def compute_m0_n2_amplitude_optimized(A, N, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_paths):
    """Compute the operator(name='bb', rank=2, m=0, n=2) amplitude."""
    truncation.confirm_at_least_singles()
    truncation.confirm_at_least_doubles()

    # the residual tensor
    dz = np.zeros(shape=(A, N, N), dtype=complex)

    # unpack the optimized paths
    optimized_HZ_paths, optimized_eT_HZ_paths = opt_paths

    # compute the terms
    compute_m0_n2_HZ_LHS_optimized(dz, ansatz, truncation, t_conj, dT, z_args, dz_args, optimized_HZ_paths)
    compute_m0_n2_eT_HZ_LHS_optimized(dz, ansatz, truncation, t_conj, dT, z_args, dz_args, optimized_eT_HZ_paths)
    return dz


def compute_m0_n3_amplitude_optimized(A, N, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_paths):
    """Compute the operator(name='bbb', rank=3, m=0, n=3) amplitude."""
    truncation.confirm_at_least_singles()
    truncation.confirm_at_least_doubles()
    truncation.confirm_at_least_triples()

    # the residual tensor
    dz = np.zeros(shape=(A, N, N, N), dtype=complex)

    # unpack the optimized paths
    optimized_HZ_paths, optimized_eT_HZ_paths = opt_paths

    # compute the terms
    compute_m0_n3_HZ_LHS_optimized(dz, ansatz, truncation, t_conj, dT, z_args, dz_args, optimized_HZ_paths)
    compute_m0_n3_eT_HZ_LHS_optimized(dz, ansatz, truncation, t_conj, dT, z_args, dz_args, optimized_eT_HZ_paths)
    return dz


def compute_m0_n4_amplitude_optimized(A, N, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_paths):
    """Compute the operator(name='bbbb', rank=4, m=0, n=4) amplitude."""
    truncation.confirm_at_least_singles()
    truncation.confirm_at_least_doubles()
    truncation.confirm_at_least_triples()
    truncation.confirm_at_least_quadruples()

    # the residual tensor
    dz = np.zeros(shape=(A, N, N, N, N), dtype=complex)

    # unpack the optimized paths
    optimized_HZ_paths, optimized_eT_HZ_paths = opt_paths

    # compute the terms
    compute_m0_n4_HZ_LHS_optimized(dz, ansatz, truncation, t_conj, dT, z_args, dz_args, optimized_HZ_paths)
    compute_m0_n4_eT_HZ_LHS_optimized(dz, ansatz, truncation, t_conj, dT, z_args, dz_args, optimized_eT_HZ_paths)
    return dz


# --------------------------------------------- OPTIMIZED PATHS FUNCTION --------------------------------------------- #