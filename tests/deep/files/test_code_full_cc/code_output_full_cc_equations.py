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
def add_m0_n0_fully_connected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='', rank=0, m=0, n=0) fully_connected terms."""

    if ansatz.ground_state:
        R += h_args[(0, 0)]

        if truncation.at_least_linear:
            if truncation.singles:
                R += np.einsum('aci, cbi -> ab', h_args[(0, 1)], t_args[(1, 0)])
    else:
        R += h_args[(0, 0)]

        if truncation.at_least_linear:
            if truncation.singles:
                R += np.einsum('aci, cbi -> ab', h_args[(1, 0)], t_args[(0, 1)])
    return


def add_m0_n0_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='', rank=0, m=0, n=0) linked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return


def add_m0_n0_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='', rank=0, m=0, n=0) unlinked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  1 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='b', rank=1, m=0, n=1) TERMS -------------- #
def add_m0_n1_fully_connected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='b', rank=1, m=0, n=1) fully_connected terms."""

    if ansatz.ground_state:
        R += h_args[(1, 0)]

        if truncation.at_least_linear:
    else:
        R += h_args[(1, 0)]

        if truncation.at_least_linear:
    return


def add_m0_n1_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='b', rank=1, m=0, n=1) linked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return


def add_m0_n1_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='b', rank=1, m=0, n=1) unlinked_disconnected terms."""

    if ansatz.ground_state:
        if truncation.singles:
            R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])

        if truncation.at_least_linear:
    else:
        if truncation.singles:
            R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])

        if truncation.at_least_linear:
    return


# -------------- operator(name='d', rank=1, m=1, n=0) TERMS -------------- #
def add_m1_n0_fully_connected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='d', rank=1, m=1, n=0) fully_connected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return


def add_m1_n0_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='d', rank=1, m=1, n=0) linked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return


def add_m1_n0_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='d', rank=1, m=1, n=0) unlinked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        if truncation.singles:
            R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(0, 1)])

        if truncation.at_least_linear:
    return


# --------------------------------------------- RESIDUAL FUNCTIONS --------------------------------------------- #
def compute_m0_n0_amplitude(A, N, ansatz, truncation, h_args, t_args):
    """Compute the operator(name='', rank=0, m=0, n=0) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A), dtype=complex)

    # add each of the terms
    add_m0_n0_fully_connected_terms(R, ansatz, truncation, h_args, t_args)
    add_m0_n0_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
    add_m0_n0_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
    return R


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


def compute_m1_n0_amplitude(A, N, ansatz, truncation, h_args, t_args):
    """Compute the operator(name='d', rank=1, m=1, n=0) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A, N), dtype=complex)

    # add each of the terms
    add_m1_n0_fully_connected_terms(R, ansatz, truncation, h_args, t_args)
    add_m1_n0_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
    add_m1_n0_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
    return R

# ------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------- OPTIMIZED FUNCTIONS -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #

# --------------------------------------------- INDIVIDUAL TERMS --------------------------------------------- #


# -------------- operator(name='', rank=0, m=0, n=0) TERMS -------------- #
def add_m0_n0_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
    """Optimized calculation of the operator(name='', rank=0, m=0, n=0) fully_connected terms."""

    if ansatz.ground_state:
        R += h_args[(0, 0)]

        if truncation.at_least_linear:
            if truncation.singles:
                R += np.einsum('aci, cbi -> ab', h_args[(0, 1)], t_args[(1, 0)])
    else:
        R += h_args[(0, 0)]

        if truncation.at_least_linear:
            if truncation.singles:
                R += np.einsum('aci, cbi -> ab', h_args[(1, 0)], t_args[(0, 1)])
    return


def add_m0_n0_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
    """Optimized calculation of the operator(name='', rank=0, m=0, n=0) linked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return


def add_m0_n0_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
    """Optimized calculation of the operator(name='', rank=0, m=0, n=0) unlinked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  1 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='b', rank=1, m=0, n=1) TERMS -------------- #
def add_m0_n1_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
    """Optimized calculation of the operator(name='b', rank=1, m=0, n=1) fully_connected terms."""

    if ansatz.ground_state:
        R += h_args[(1, 0)]

        if truncation.at_least_linear:
    else:
        R += h_args[(1, 0)]

        if truncation.at_least_linear:
    return


def add_m0_n1_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
    """Optimized calculation of the operator(name='b', rank=1, m=0, n=1) linked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return


def add_m0_n1_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
    """Optimized calculation of the operator(name='b', rank=1, m=0, n=1) unlinked_disconnected terms."""

    if ansatz.ground_state:
        if truncation.singles:
            R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])

        if truncation.at_least_linear:
    else:
        if truncation.singles:
            R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])

        if truncation.at_least_linear:
    return


# -------------- operator(name='d', rank=1, m=1, n=0) TERMS -------------- #
def add_m1_n0_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
    """Optimized calculation of the operator(name='d', rank=1, m=1, n=0) fully_connected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return


def add_m1_n0_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
    """Optimized calculation of the operator(name='d', rank=1, m=1, n=0) linked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return


def add_m1_n0_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
    """Optimized calculation of the operator(name='d', rank=1, m=1, n=0) unlinked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        if truncation.singles:
            R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(0, 1)])

        if truncation.at_least_linear:
    return


# --------------------------------------------- RESIDUAL FUNCTIONS --------------------------------------------- #
def compute_m0_n0_amplitude_optimized(A, N, ansatz, truncation, h_args, t_args, opt_paths):
    """Compute the operator(name='', rank=0, m=0, n=0) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A), dtype=complex)

    # unpack the optimized paths
    optimized_connected_paths, optimized_linked_paths, optimized_unlinked_paths = opt_paths

    # add each of the terms
    add_m0_n0_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, optimized_connected_paths)
    add_m0_n0_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, optimized_linked_paths)
    add_m0_n0_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, optimized_unlinked_paths)
    return R


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


def compute_m1_n0_amplitude_optimized(A, N, ansatz, truncation, h_args, t_args, opt_paths):
    """Compute the operator(name='d', rank=1, m=1, n=0) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A, N), dtype=complex)

    # unpack the optimized paths
    optimized_connected_paths, optimized_linked_paths, optimized_unlinked_paths = opt_paths

    # add each of the terms
    add_m1_n0_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, optimized_connected_paths)
    add_m1_n0_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, optimized_linked_paths)
    add_m1_n0_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, optimized_unlinked_paths)
    return R


# --------------------------------------------- OPTIMIZED PATHS FUNCTION --------------------------------------------- #