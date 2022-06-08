# system imports
import itertools as it
# third party imports

# local imports
from common_imports import tab, old_print_wrapper

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------  HELPER FUNCTIONS  --------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


def unique_permutations(iterable):
    """Return a sorted list of unique permutations of the items in some iterable."""
    return sorted(list(set(it.permutations(iterable))))


def build_symmetrizing_function(max_order=5, show_perm=False): #pragma: no cover
    """ x """
    string = ""
    string += (
        f"\ndef symmetrize_tensor(tensor, order):\n"
        f"{tab}'''Symmetrizing a tensor (the W operator or the residual) by tracing over all permutations.'''\n"
        f"{tab}X = np.zeros_like(tensor, dtype=complex)\n"
    )

    string += (
        f"{tab}if order == 0:\n"
        f"{tab}{tab}return tensor\n"
        f"{tab}if order == 1:\n"
        f"{tab}{tab}return tensor\n"
    )

    for n in range(2, max_order+1):
        string += f"{tab}if order == {n}:\n"
        for p in it.permutations(range(2, n+2)):
            string += f"{tab}{tab}X += np.transpose(tensor, {(0,1) + p})\n"

    string += f"{tab}return X\n"
    return string


def print_residual_data(R_lists, term_lists, print_equations=False, print_tuples=False): #pragma: no cover
    """Print to stdout in a easily readable format the residual terms and term tuples."""
    if print_equations:
        for i, R in enumerate(R_lists):
            old_print_wrapper(f"{'':-<30} R_{i} {'':-<30}")
            for a in R:
                old_print_wrapper(f"{tab} - {a}")
        old_print_wrapper(f"{'':-<65}\n{'':-<65}\n")

    if print_tuples:
        for i, terms in enumerate(term_lists):
            old_print_wrapper(f"{'':-<30} R_{i} {'':-<30}")
            for term in terms:
                old_print_wrapper(f"{tab} - {term}")
        old_print_wrapper(f"{'':-<65}\n{'':-<65}\n")

    return


def _partitions(number):
    """Return partitions of n. See `https://en.wikipedia.org/wiki/Partition_(number_theory)`"""
    answer = set()
    answer.add((number,))
    for x in range(1, number):
        for y in _partitions(number - x):
            answer.add(tuple(sorted((x, ) + y, reverse=True)))

    return sorted(list(answer), reverse=True)


def generate_partitions_of_n(n): 
    """Return partitions of n. Such as (5,), (4, 1), (3, 1, 1), (2, 2, 1) ... etc."""
    return _partitions(n)


def generate_mixed_partitions_of_n(n): #pragma: no cover
    """Return partitions of n that include at most one number greater than 1.
    Such as (5,), (4, 1), (3, 1, 1), (2, 1, 1, 1) ... etc, but not (3, 2) or (2, 2, 1)
    """
    return [p for p in _partitions(n) if n - max(p) + 1 == len(p)]


def genereate_connected_partitions_of_n(n): #pragma: no cover
    """Return partitions of n which are only comprised of 1's.
    Such as (1, 1), or (1, 1, 1). The max value should only ever be 1.
    """
    return tuple([1]*n)


def generate_linked_disconnected_partitions_of_n(n):
    """Return partitions of n that include at most one number greater than 1 and not `n`.
    Such as (4, 1), (3, 1, 1), (2, 1, 1, 1) ... etc, but not (5,), (3, 2), (2, 2, 1)
    """
    return [p for p in _partitions(n) if n - max(p) + 1 == len(p) and max(p) < n]


def generate_un_linked_disconnected_partitions_of_n(n): #pragma: no cover
    """Return partitions of n that represent the unlinked disconnected wave operator parts.
    Such as (3, 2), (2, 2, 1) ... etc, but not (5,), (4, 1), (3, 1, 1), (2, 1, 1, 1)
    """
    new_set = set(_partitions(n)) - set(generate_mixed_partitions_of_n(n))
    return sorted(list(new_set), reverse=True)


# ----------------------------------------------------------------------------------------------- #
# Helper functions for writing code / spacing the text out

def named_line(name, width):
    """ Generate a header like
        (# ---- <name> ---- #)
    `width` argument determines how many `-` chars on BOTH sides.
    """
    return "# " + "-"*width + f" {name} " + "-"*width + " #"


def spaced_named_line(name, width=28, spacing_line=f"# {'-'*75} #\n"):
    """ Generate a header like
        (# ---------------- #)
        (# ---- <name> ---- #)
        (# ---------------- #)
    `width` argument determines how many `-` chars on BOTH sides in the middle line
    """
    return spacing_line + named_line(name, width) + '\n' + spacing_line


def long_spaced_named_line(name, width=45, large_spacing_line=f"# {'-'*109} #\n"):
    """ Wrapper for `spaced_named_line` to distinguish larger headers. """
    return spaced_named_line(name, width, spacing_line=large_spacing_line)


# ----------------------------------------------------------------------------------------------- #
