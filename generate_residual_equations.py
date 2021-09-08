# system imports
import os
import sys
import math
import itertools as it
from collections import namedtuple

# third party imports
import numpy as np

# local imports
import reference_latex_headers as headers


# define
tab_length = 4
tab = " "*tab_length

# use these if we define \newcommand to map textbf to \bt and so forth
if True:
    bold_t_latex = "\\bt"
    bold_h_latex = "\\bh"
    bold_w_latex = "\\bw"
    bold_c_latex = "\\bc"
    bold_d_latex = "\\bd"
    bold_z_latex = "\\bz"
    bold_s_latex = "\\bs"
    bold_G_latex = "\\bG"
else:
    bold_t_latex = "\\textbf{t}"
    bold_h_latex = "\\textbf{h}"
    bold_w_latex = "\\textbf{w}"
    bold_c_latex = "\\textbf{c}"
    bold_d_latex = "\\textbf{d}"
    bold_z_latex = "\\textbf{z}"
    bold_s_latex = "\\textbf{s}"
    bold_G_latex = "\\textbf{G}"


def dump_all_stdout_to_devnull():
    sys.stdout = open(os.devnull, 'w')


def old_print_wrapper(*args, **kwargs):
    """ wrapper for turning all old prints on/off"""

    # delayed default argument
    if 'suppress_print' not in kwargs:
        kwargs['suppress_print'] = True

    if not kwargs['suppress_print']:
        del kwargs['suppress_print']  # remove `suppress_print` flag
        print(*args, **kwargs)


# our Hamiltonian is
# H = (h_0 + omega + h^1 + h_1) + h^1_1 + h^2 + h_2
# but we can ignore the omega when calculating residuals as we add it back in at a later point
# so we use this H = h_0 + h^1 + h_1 + h^1_1 + h^2 + h_2

# ----------------------------------------------------------------------------------------------- #
# -------------------------------  NAMED TUPLES DEFINITIONS  ------------------------------------ #
# ----------------------------------------------------------------------------------------------- #
# the building blocks for the h & w components of each residual term are stored in named tuples


# rterm_namedtuple = namedtuple('rterm_namedtuple', ['prefactor', 'h', 'w'])
# we originally defined a class so that we can overload the `__eq__` operator
# because we needed to compare the rterm tuples, however I think I might have removed that code
# so Shanmei or I should check if we even need the class anymore
class residual_term(namedtuple('residual_term', ['prefactor', 'h', 'w'])):
    __slots__ = ()

    #
    def __eq__(self, other_term):
        return bool(
            self.prefactor == other_term.prefactor and
            np.array_equal(self.h, other_term.h) and
            np.array_equal(self.w, other_term.w)
        )


# tuple for a general operator as described on page 1, eq 1
general_operator_namedtuple = namedtuple('operator', ['name', 'rank', 'm', 'n'])

# connected_namedtuple = namedtuple('connected', ['name', 'm', 'n'])
# linked_namedtuple = namedtuple('linked', ['name', 'm', 'n'])
# unlinked_namedtuple = namedtuple('unlinked', ['name', 'm', 'n'])

hamiltonian_namedtuple = namedtuple('hamiltonian', ['maximum_rank', 'operator_list'])

"""rather than just lists and dictionaries using namedtuples makes the code much more concise
we can write things like `h.max_i` instead of `h[0]` and the label of the member explicitly
describes what value it contains making the code more readable and user friendly """
h_namedtuple = namedtuple('h_term', ['max_i', 'max_k'])
w_namedtuple = namedtuple('w_term', ['max_i', 'max_k', 'order'])

# used in building the W operators
t_term_namedtuple = namedtuple('t_term_namedtuple', ['string', 'order', 'shape'])

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------  HELPER FUNCTIONS  --------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


def unique_permutations(iterable):
    """Return a sorted list of unique permutations of the items in some iterable."""
    return sorted(list(set(it.permutations(iterable))))


def build_symmetrizing_function(max_order=5, show_perm=False):
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


def print_residual_data(R_lists, term_lists, print_equations=False, print_tuples=False):
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


def generate_mixed_partitions_of_n(n):
    """Return partitions of n that include at most one number greater than 1.
    Such as (5,), (4, 1), (3, 1, 1), (2, 1, 1, 1) ... etc, but not (3, 2) or (2, 2, 1)
    """
    return [p for p in _partitions(n) if n - max(p) + 1 == len(p)]


def genereate_connected_partitions_of_n(n):
    """Return partitions of n which are only comprised of 1's.
    Such as (1, 1), or (1, 1, 1). The max value should only ever be 1.
    """
    return tuple([1]*n)


def generate_linked_disconnected_partitions_of_n(n):
    """Return partitions of n that include at most one number greater than 1 and not `n`.
    Such as (4, 1), (3, 1, 1), (2, 1, 1, 1) ... etc, but not (5,), (3, 2), (2, 2, 1)
    """
    return [p for p in _partitions(n) if n - max(p) + 1 == len(p) and max(p) < n]


def generate_un_linked_disconnected_partitions_of_n(n):
    """Return partitions of n that represent the unlinked disconnected wave operator parts.
    Such as (3, 2), (2, 2, 1) ... etc, but not (5,), (4, 1), (3, 1, 1), (2, 1, 1, 1)
    """
    new_set = set(_partitions(n)) - set(generate_mixed_partitions_of_n(n))
    return sorted(list(new_set), reverse=True)

# ----------------------------------------------------------------------------------------------- #
# --------------------------------  GENERATING RESIDUAL DATA  ----------------------------------- #
# ----------------------------------------------------------------------------------------------- #


def generate_hamiltonian_operator(maximum_h_rank=2):
    """Return a `hamiltonian_namedtuple`.
    It contains an `operator_list` of namedtuples for each term that looks like equation 6 based on `maximum_h_rank`
    (which is the sum of the ranks (m,n)).
    Equation 6 is a Hamiltonian with `maximum_h_rank` of 2
    """
    return_list = []

    for m in range(maximum_h_rank + 1):              # m is the upper label
        for n in range(maximum_h_rank + 1 - m):      # n is the lower label
            if m == 0 and n == 0:
                h_operator = general_operator_namedtuple("h_0", 0, 0, 0)
            elif m == 0:
                h_operator = general_operator_namedtuple(f"h_{n}", 0+n, 0, n)
            elif n == 0:
                h_operator = general_operator_namedtuple(f"h^{m}", m+0, m, 0)
            else:
                h_operator = general_operator_namedtuple(f"h^{m}_{n}", m+n, m, n)

            return_list.append(h_operator)

    return hamiltonian_namedtuple(maximum_h_rank, return_list)


# ------------- constructing the prefactor -------------- #

def extract_numerator_denominator_from_string(s):
    """Return the number part of the numerator and denominator
    from a string (s) of a fraction w factorial in denominator part.
    """

    if "*" in s:  # we are only trying to get the first fraction
        s = s.split('*')[0]

    # old_print_wrapper(s)
    numer, denom = s.replace('(', '').replace(')', '').split(sep="/")
    denom = denom.split(sep='!')[0]

    # old_print_wrapper(f"numer: {numer} denom: {denom}")
    return [int(numer), int(denom)]


def simplified_prefactor(pre):
    """Creates the simplified form of the given prefactor f."""

    arr = extract_numerator_denominator_from_string(pre)
    numerator, denominator = arr[0], arr[1]

    if pre == "*(1/2)":    # case when 1/2 is the only prefactor, delete * sign
        pre = "1/2"
    elif denominator == numerator:
        # case when (1/1!) or (2/2!) is present, which will both be recognized as 1
        if "*(1/2)" in pre:
            if denominator == 1 or denominator == 2:
                pre = "(1/2)"     # 1*(1/2) = (1/2)
            else:
                pre = f"({numerator}/(2*{denominator}))"
        else:
            if denominator == 1 or denominator == 2:
                pre = ""   # use empty string to represent 1

    # case when 1/2 is multiplied to the prefactor
    elif "*(1/2)" in pre:
        if numerator % 2 == 0:
            pre[0] = str(numerator // 2)
            pre = pre[:-6]  # get rid of "*(1/2)"
        else:
            pre = pre[:3] + "(2*" + pre[3:-6] + ")"  # add "2*" to the front of the denominator
    return pre


def construct_prefactor(h, p, simplify_flag=False):
    """Creates the string for the prefactor of the tensor h.
    Returns condensed fraction by default.
    If `simplify_flag` is true it reduces the fraction using the gcf(greatest common factor).
    If the prefactor is equal to 1 or there is no prefactor then returns an empty string.
    """

    # is there a 'master' equations / a general approach for any number of i's / k's

    # Should be able to simplify down to 3 cases? maybe?
    # case 1 - only 1 k, any number of i's or only 1 i, any number of k's
    # case 2 - 2 or more k's AND 2 or more i's
    # case 3 - only i or only k presents

    if h.m > p:
        return ""

    prefactor = ""
    i_number_w = p - h.m
    total_number_w = p - h.m + h.n

    # special case when there is no w operator and h^2 doesn't present
    if total_number_w == 0:
        if h.m == 2:
            return "(1/2)"
        else:
            return ""

    # case 1. when there is only 1 k or i label on w operator
    if i_number_w == 1 or h.n == 1:
        prefactor = f"({total_number_w}/{total_number_w}!)"  # needs simplification: n/n! = 1/(n-1)

    # case 2. when 2 or more k's AND 2 or more i's on w
    elif h.n > 1 and i_number_w > 1:
        prefactor += f"({sum(x for x in range(total_number_w))}/{total_number_w}!)"

    # case 3. when only i or only k presents on w operator
    elif h.n == 0 or i_number_w == 0:
        prefactor += f"(1/{total_number_w}!)"

    # special case: when h^2 is included, 1/2 needs to be multiplies to the term
    if h.m == 2:
        prefactor += "*(1/2)"

    # if simplification is needed
    if simplify_flag:            # call simplification step
        prefactor = simplified_prefactor(prefactor)

    return prefactor


# ---- construct string labels for each residual term --- #

def construct_upper_w_label(h, p):
    """Creates the string for the "upper" labels of the tensor w.
    Returns `^{str}` if there are upper labels
    Otherwise returns an empty string
    """
    if (h.m == p and h.n == 0) or h.m > p:   # case when there is no w operator needed
        return ""

    w_label = "^{"   # if w operator exist, initialize w_label
    for a in range(h.m+1, p+1):   # add i_p
        w_label += f"i_{a},"
    for b in range(1, h.n+1):     # add k_h
        w_label += f"k_{b},"

    assert w_label != "^{", "Whoops you missed a case, check the logic!!!"

    w_label = w_label[:-1] + "}"  # delete extra comma and add close bracket
    return w_label


def construct_upper_h_label(h, p):
    """Creates the string for the "upper" labels of the tensor h.
    Returns `^{str}` if there are upper labels
    Otherwise returns an empty string
    """
    if h.m == 0 or h.m > p:  # case when there is no upper i label or no proper h operator
        return ""

    upper_h_result = "^{"  # initialize the return string if upper label exist (m!=0)
    for c in range(1, h.m+1):
        upper_h_result += f"i_{c},"

    upper_h_result = upper_h_result[:-1] + "}"  # delete extra comma and add close bracket
    return upper_h_result


def construct_lower_h_label(h, p):
    """Creates the string for the "lower" labels of the tensor h.
    Returns `_{str}` if there are lower labels
    Otherwise returns an empty string
    """
    # case when h_o presents
    if h.m == 0 and h.n == 0:
        return "_0"

    # case when h operator doesn't have lower label
    if h.n == 0 or h.m > p:
        return ""

    # initialize the return string if lower label exist (n!=0)
    lower_h_result = "_{"
    for d in range(1, h.n+1):
        lower_h_result += f"k_{d},"

    lower_h_result = lower_h_result[:-1] + "}"
    return lower_h_result


# ------- constructing individual residual terms -------- #

def generate_p_term(str_fac):
    """Generate a floating point number calculated by the fraction in the input string fac_str"""

    # check if the prefactor is 1
    if str_fac == "":
        return "1.0"

    if str_fac == "(1/2) * ":
        return "0.5"

    arr = extract_numerator_denominator_from_string(str_fac)
    numerator, denominator = arr[0], arr[1]

    if "/(2*" in str_fac:
        return ("(" + str(numerator) + "/(2*" + str(math.factorial(denominator)) + "))")
    else:
        return ("(" + str(numerator) + "/" + str(math.factorial(denominator)) + ")")


def generate_h_term(str_h):
    """Generate an h_namedtuple; contains max_i and max_k of h operator"""
    if "0" in str_h:
        # special case for h_0
        return h_namedtuple(0, 0)

    return h_namedtuple(str_h.count("i"), str_h.count("k"))


def generate_w_term(str_w):
    """Generate an w_namedtuple; contains max_i and max_k of w operator"""
    if str_w == "":
        # if there is no w operator, return [0,0,0]
        return w_namedtuple(0, 0, 0)

    max_i = str_w.count("i")
    max_k = str_w.count("k")
    return w_namedtuple(max_i, max_k, max_i+max_k)

# ------------------------------------------------------- #


def generate_residual_string_list(hamiltonian, order):
    """Return a list of strings that will represent each term in the equations 59-63; based on order === p

    the indices i go from i_1, i_2 to i_order

    For each term in the Hamiltonian (see hamiltonian.png) we use the value of order === p
    to determine how much pairing happens when projecting (see residual_equation.png)
    such that we produce a string like the one we see in equation 59
    Refer back to operator_equation.png to remind yourself of the indices for each of the terms in the Hamiltonian
    """
    return_list = []
    term_list = []

    for h_operator in hamiltonian:

        # initialize the new hamiltonian operator, the w operator and the numeric part of each hamiltonian operator
        h_result, w_result, prefactor = "h", "w", ""

        # construct each hamiltonian operator and w operator
        h_result += construct_upper_h_label(h_operator, order)
        h_result += construct_lower_h_label(h_operator, order)
        w_result += construct_upper_w_label(h_operator, order)
        prefactor = construct_prefactor(h_operator,  order, True)

        if h_result == "h":
            continue

        # make terms here
        term_list += [residual_term(generate_p_term(prefactor), generate_h_term(h_result), generate_w_term(w_result)), ]

        # add '*' symbols for the text representation
        w_result = "" if (w_result == "w") else (" * " + w_result)
        if not (prefactor == ""):
            prefactor = prefactor + " * "

        return_list += [prefactor + h_result + w_result, ]

    return return_list, term_list


def generate_residual_data(H, max_order):
    """Return two lists of length `max_order` containing the data on the residual equations.
    The `R_lists` contains string representations of the residual equations, to be used to generate latex documents.
    The `term_lists` contains tuple representations of the residuals
    to be used to generate python code for use in simulation.
    `term_lists` is a list of lists, each of which contain tuples `(prefactor, h, w)`
    representing terms of that particular residual.
    """
    lst = [tuple(generate_residual_string_list(H, order=order)) for order in range(max_order+1)]
    R_lists = [tup[0] for tup in lst]
    term_lists = [tup[1] for tup in lst]
    return R_lists, term_lists


# ----------------------------------------------------------------------------------------------- #
# --------------------------------  GENERATING FULL CC LATEX  ----------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# this serves the same purpose as `general_operator_namedtuple`
# we changed the typename from "operator" to "h_operator" to distinguish the specific role and improve debugging
h_operator_namedtuple = namedtuple('h_operator', ['rank', 'm', 'n'])


# these serve the same purpose as `hamiltonian_namedtuple`
# we changed the typename from "operator" to distinguish the specific role and improve debugging
s_operator_namedtuple = namedtuple('s_operator', ['maximum_rank', 'operator_list'])
omega_namedtuple = namedtuple('Omega', ['maximum_rank', 'operator_list'])

"""
Note that the attributes for these `namedtuples` operate in a specific manner.
The general rule that can be followed is the m/n refers to the object it is an attribute of:
`t.m_h` refers to the `m` value of `t`, contracting with `h`, which implies that it contracts with the `n` of `h`.
The `m_h` attribute counts how many of the omega or t amplitudes's m indices contract with the n indices of the h operator.
So a h_1 couples with t^1 but the values inside the t^1 `namedtuple` object would be `m_h`=1, `n_h`=0, `m_o`=0, `n_o`=0.
"""


# the `m_t` and `n_t` are lists of integers whose length is == number of t's
# so o^2_1 h^1 t_1 t_1 would mean m_t = [1, 1]
connected_omega_operator_namedtuple = namedtuple('connected_omega', ['rank', 'm', 'n', 'm_h', 'n_h', 'm_t', 'n_t'])
connected_h_operator_namedtuple = namedtuple('connected_h', ['rank', 'm', 'n', 'm_o', 'n_o', 'm_t', 'n_t'])

# namedtuples for the t amplitudes
connected_namedtuple = namedtuple('connected', ['m_h', 'n_h', 'm_o', 'n_o'])
disconnected_namedtuple = namedtuple('disconnected', ['m_h', 'n_h', 'm_o', 'n_o'])

""" These are the indices used to label the h and t's in the generated latex"""
summation_indices = 'ijklmno'
unlinked_indices = 'zyxwuv'


# ---------------------- generating basic operators ------------------------- #
def generate_omega_operator(maximum_cc_rank=2, omega_max_order=3):
    """Return an `omega_namedtuple` whose attributes are determined by `maximum_cc_rank`.

    The `operator_list` contains all permutations of (`m`,`n`) for `m` & `n` in `range(maximum_cc_rank + 1)`.
    The name is a string of the chars `d` and `b` according to `m` and `n` respectively.
    `m` is associated with creation operators (d) and `n` is associated with annihilation operators (b).

    For m == n == 0 we generate `operator(name='', m=0, n=0)` which represents the zero order equation
    """

    return_list = []

    for m in range(maximum_cc_rank + 1):          # m is the upper label (creation operators)
        for n in range(maximum_cc_rank + 1 - m):  # n is the lower label (annihilation operators)

            name = "d"*m + "b"*n

            if m+n <= omega_max_order:
                return_list.append(general_operator_namedtuple(name, m+n, m, n))

    return_list.sort(key=lambda x: len(x.name))

    return omega_namedtuple(maximum_cc_rank, return_list)


def generate_full_cc_hamiltonian_operator(maximum_rank=2):
    """Return a `hamiltonian_namedtuple`.
    This function is different from `generate_hamiltonian_operator` in that the operators are `h_operator_namedtuple`s
    and not `general_operator_namedtuple`s. It is specifically for generating the full CC latex
    """
    return_list = []

    for m in range(maximum_rank + 1):              # m is the upper label
        for n in range(maximum_rank + 1 - m):      # n is the lower label
            return_list.append(h_operator_namedtuple(m+n, m, n))

    return hamiltonian_namedtuple(maximum_rank, return_list)


def generate_s_operator(maximum_cc_rank=2, only_ground_state=False):
    """Return an `s_operator_namedtuple` whose attributes are determined by `maximum_cc_rank`.

    The `operator_list` contains all permutations of (`m`,`n`) for `m` & `n` in `range(maximum_cc_rank + 1)`.
    The name is a string of the chars `d` and `b` according to `m` and `n` respectively.
    `m` is associated with creation operators (d) and `n` is associated with annihilation operators (b).
    The Boolean flag `only_ground_state` restricts `n` to be 0 for all operators.
    Only creation operators can act on a system in the ground state, => `n` is required to be 0.
    """
    return_list = []

    for m in range(maximum_cc_rank + 1):          # m is the upper label (creation operators)
        for n in range(maximum_cc_rank + 1 - m):  # n is the lower label (annihilation operators)

            # skip any S with annihilation operator when `only_ground_state` is True
            if only_ground_state and n > 0:
                continue

            # we account for the zero order S operator in `_generate_s_taylor_expansion`
            if m == n == 0:
                continue

            name = "s"
            name += f"^{m}" if m > 0 else ""
            name += f"_{n}" if n > 0 else ""

            return_list.append(general_operator_namedtuple(name, m+n, m, n))

    return s_operator_namedtuple(maximum_cc_rank, return_list)


def generate_s_taylor_expansion(maximum_cc_rank=2, s_taylor_max_order=3, only_ground_state=False):
    """Return a list of lists of `s_operator_namedtuple`s.

    Expanding e^{S} by Taylor series gives you 1 + S + S^2 + S^3 ... etc.
    Each of the terms in that sum (1, S, S^2, ...) is represented by a list inside the returned list.
    So `s_taylor_expansion[0]` is the 1 term and has a single `s_operator_namedtuple`.
    Then `s_taylor_expansion[1]` represents the `S` term and is a list of `s_operator_namedtuple`s
    generated by `generate_s_operator` from s_1, s^1 all the way to s^m_n as determined by `maximum_cc_rank`.
    The third list `s_taylor_expansion[2]` is all the S^2 terms, and so on.

    For terms S^2 and higher we compute all possible products, including non unique ones.
    This means for S^3 we will compute s^1 * s^1 * s^1 three times, however later on we will remove the duplicate terms.
    The duplicate terms are used to account for multiple possible index label orders.

    The Boolean flag `only_ground_state` is passed to `generate_s_operator` which restricts `n` to be 0 for all operators.
    Only creation operators can act on a system in the ground state, => `n` is required to be 0.
    """

    # The s_operator_namedtuple's
    S = generate_s_operator(maximum_cc_rank, only_ground_state)

    # create the list
    s_taylor_expansion = [None, ]*(s_taylor_max_order+1)

    s_taylor_expansion[0] = general_operator_namedtuple("1", 0, 0, 0)  # 1 term
    s_taylor_expansion[1] = S.operator_list                         # S term

    """ We compute all combinations including non unique ones ON PURPOSE!!
    The products of S operators do not have indices mapping them to omega and H.
    Therefore s^2 * s^1 === s^1 * s^2.
    Later when we add indices, the non unique combinations will become unique,
    due to the nature of the process of assigning the indices.
    For example, if omega = b and h_{ij}:
     - s^2 * s^1 can become s^{ij} * s^{z}, which would be a disconnected term.
     - s^1 * s^2 can become s^{i} * s^{jz}, which would be a connected term.
    """
    for n in range(2, s_taylor_max_order+1):
        s_taylor_expansion[n] = [list(tup) for tup in it.product(S.operator_list, repeat=n)]

    return s_taylor_expansion


# ------------------- generating assorted helper functions --------------------- #
def _debug_print_valid_term_list(valid_term_list, sn_terms=False):
    """ If debugging and need to print valid terms in `valid_term_list`.
    By default only prints S^0 and S^1 terms. If `sn_terms` is `True` also prints S^n terms.
    There can be hundreds or thousands of S^n terms.
    """

    s_1_start_index = 0
    s_n_start_index = 0

    for idx, term in enumerate(valid_term_list):
        s_list = term[2]
        if len(s_list) == 1 and s_list[0] == disconnected_namedtuple(0, 0, 0, 0):
            s_1_start_index += 1
        elif len(s_list) >= 2:
            s_n_start_index = idx
            break

    log.info(f"{s_1_start_index=} {s_n_start_index=}")

    for term in valid_term_list[0:s_1_start_index]:
        log.info(f"S^0: {term}")

    for term in valid_term_list[s_1_start_index:s_n_start_index]:
        log.info(f"S^1: {term}")

    if sn_terms:
        for term in valid_term_list[s_n_start_index:]:
            log.info(f"S^n: {term}")

    return


def _debug_print_different_types_of_terms(fully, linked, unlinked):
    """ If debugging and need to print all terms in `fully` `linked` and `unlinked`."""
    log.debug("-"*40 + "FULLY CONNECTED TERMS" + "-"*40)
    for term in fully:
        log.debug(term)

    log.debug("-"*40 + "LINKED DISCONNECTED TERMS" + "-"*40)
    for term in linked:
        log.debug(term)

    log.debug("-"*40 + "UNLINKED DISCONNECTED TERMS" + "-"*40)
    for term in unlinked:
        log.debug(term)


def _build_latex_h_string(h, unsummed=False):
    """ x """
    string = "h"

    # we need to use different indices
    index_list = summation_indices if not unsummed else unlinked_indices

    if h.m > 0:
        string += f"^{{{index_list[0:h.m]}}}"

    if h.n > 0:
        string += f"_{{{index_list[h.m:h.m+h.n]}}}"

    if string == "h":
        string = "h_0"

    return string.replace("h", bold_h_latex)


def _build_latex_t_string(s):
    """ x """
    string = "t"

    if (s.h_n > 0) or (s.o_n > 0):
        string += f"^{{{summation_indices[s.h_m:s.h_m+s.h_n]}{unlinked_indices[s.o_m:s.o_m+s.o_n]}}}"

    if (s.h_m > 0) or (s.o_m > 0):
        string += f"_{{{summation_indices[0:s.h_m]}{unlinked_indices[0:s.o_m]}}}"

    if string == "t":
        string = ""

    return string.replace('t', bold_t_latex)


# --------------------- Validating operator pairings ------------------------ #
def _t_joining_with_t_terms(omega, h, s_list, nof_creation_ops):
    """Remove terms like `b h^1 t^2 t_2` which require the t^2 to join with t_2.

    We count the number of annihilation operators `b` and creation operators `d`
    provided by the Omega and H operators. Next we count the number of operators (`b`,`d`)
    required by all the t operators. If the t operators require more operators than
    Omega or H provide this implies that they would be contracting/joining with each other.
    Theoretically this doesn't exist, and therefore we reject this term.
    """
    available_b = omega.n + h.n
    available_d = omega.m + h.m

    required_b = sum([s.m for s in s_list])
    required_d = sum([s.n for s in s_list])

    if (required_b > available_b) or (required_d > available_d):
        return True

    return False


def _omega_joining_with_itself(omega, h, s_list):
    """Remove terms like `bd h_0` which require Omega to join with itself.

    We already know that the number of operators is balanced, as we check
    that before calling this function. So here we check if the s or h terms have any b/d
    operators for omega to join with. If all these terms are h^0_0 and/or s^0_0 then omega
    must be joining with itself. Theoretically this doesn't exist, and therefore we reject this term.
    """

    # omega can't join with itself unless it has both creation and annihilation operators
    if (omega.m == 0) or (omega.n == 0):
        return False

    if (omega.n > 0 and h.m > 0) or (omega.m > 0 and h.n > 0):
        return False

    for s in s_list:
        if (omega.n > 0 and s.m > 0) or (omega.m > 0 and s.n > 0):
            return False

    return True


def _h_joining_with_itself(omega, h, s_list):
    """Remove terms like `h^1_1` which require h to join with itself.

    We already know that the number of operators is balanced, as we check
    that before calling this function. So here we check if the s or omega terms have any b/d
    operators for h to join with. If all these terms are o^0_0 and/or s^0_0 then h
    must be joining with itself. Theoretically this doesn't exist, and therefore we reject this term.
    """

    # h can't join with itself unless it has both creation and annihilation operators
    if (h.m == 0) or (h.n == 0):
        return False

    if (h.m > 0 and omega.n > 0) or (h.n > 0 and omega.m > 0):
        return False

    for s in s_list:
        if (h.m > 0 and s.n > 0) or (h.n > 0 and s.m > 0):
            return False

    return True


def _build_h_string(h):
    """ x """
    string = "h"

    if h.m > 0:
        string += f"^{{{h.m}}}"

    if h.n > 0:
        string += f"_{{{h.n}}}"

    if string == "h":
        string = "h_0"

    return string


# ----------------------- generating operators -------------------------- #
def _generate_valid_s_n_operator_permutations(omega, h, s_series_term):
    """ Remove s permutations whose b/d operators don't add up (theoretically can't exist)
    For example O^1 h_1 is allowed but not O^1 h_1 t^1 because we have 2 d operators but only 1 b operator.
    Additionally we need to make sure the t operators are not joining with themselves.
    This means that the b/d operators from Omega and h need to be sufficient to balance the b/d's from the t's.
    So O^1_1, h_2, t^1_1, t^2 is allowed but not O_1, h_1, t^1_1, t^2 because the t^1_1 term has to pair with the
    t^2, or in other words the only sources of d operators are t terms so the b operator from t^1_1 has to pair with
    a d from a t term. This is not allowed.
    """

    valid_permutations = []

    # generate all the possible valid permutations
    for perm in s_series_term:

        nof_creation_ops = omega.m + h.m + sum([s.m for s in perm])
        nof_annhiliation_ops = omega.n + h.n + sum([s.n for s in perm])
        cannot_pair_off_b_d_operators = bool(nof_creation_ops != nof_annhiliation_ops)

        # only terms which can pair off all operators are non zero
        if cannot_pair_off_b_d_operators:
            log.debug('Bad Permutation (b and d are not balanced):', omega, h, perm)
            continue

        # omega and H need to satisfy all b/d requirements of the t terms, t terms cannot join with each other!!
        if _t_joining_with_t_terms(omega, h, perm, nof_creation_ops):
            log.debug('Bad Permutation (t joins with itself):', omega, h, perm)
            continue

        # Omega must be able to connect with at least 1 b/d operator from h or a t_term otherwise it 'joins' with itself
        if _omega_joining_with_itself(omega, h, perm):
            log.debug('Bad Permutation (omega joins with itself):', omega, h, perm)
            continue

        # h must connect with at least 1 b/d operator from omega or a t_term otherwise it 'joins' with itself
        if _h_joining_with_itself(omega, h, perm):
            log.debug('Bad Permutation (h joins with itself):', omega, h, perm)
            continue

        # record a valid permutation
        valid_permutations.append(perm)
        log.debug('Good Permutation', omega, h, perm)

    return valid_permutations


def _generate_all_valid_t_connection_permutations(omega, h, s_term_list, log_invalid=True):
    """ Generate all possible valid combinations of t terms
    with omega and h over all index distributions.
    By convention the tuples are (o, h).
    """

    valid_upper_perm_combinations = []
    valid_lower_perm_combinations = []

    m_perms, n_perms = [], []

    # generate all possible individual t assignments
    for s_term in s_term_list:
        M, N = s_term.m, s_term.n

        m_list = list(range(0, M+1))
        n_list = list(range(0, N+1))

        m_perms.append([(i, M-i) for i in m_list])
        n_perms.append([(i, N-i) for i in n_list])

    # validate upper pairing
    combined_m_perms = list(it.product(*m_perms))
    for m_perm in combined_m_perms:

        total_o_m = sum([t[0] for t in m_perm])
        total_h_m = sum([t[1] for t in m_perm])

        if (total_h_m <= h.n) and (total_o_m <= omega.n):
            log.debug(f"Valid upper perm:   h={total_h_m}, o={total_o_m}   {m_perm}")
            valid_upper_perm_combinations.append(m_perm)

        elif log_invalid:
            log.debug(
                "Invalid upper perm: "
                f"h={total_h_m} <= {h.n} and o={total_o_m} <= {omega.n}"
                f" {m_perm}"
            )

    # validate lower pairing
    combined_n_perms = list(it.product(*n_perms))
    for n_perm in combined_n_perms:

        total_o_n = sum([t[0] for t in n_perm])
        total_h_n = sum([t[1] for t in n_perm])

        if (total_h_n <= h.m) and (total_o_n <= omega.m):
            log.debug(f"Valid lower perm:   h={total_h_n}, o={total_o_n}   {n_perm}")
            valid_lower_perm_combinations.append(n_perm)

        elif log_invalid:
            log.debug(
                "Invalid lower perm: "
                f"h={total_h_n} <= {h.m} and o={total_o_n} <= {omega.m}"
                f" {n_perm}"
            )

    return valid_upper_perm_combinations, valid_lower_perm_combinations


def _generate_all_omega_h_connection_permutations(omega, h, valid_permutations, found_it_bool=False):
    """ Generate all possible permutations of matching with omega and h for t_terms """

    annotated_permutations = []  # store output here

    for perm in valid_permutations:
        upper_perms, lower_perms = _generate_all_valid_t_connection_permutations(omega, h, perm)
        log.debug(f"{upper_perms=}")
        log.debug(f"{lower_perms=}")

        for upper in upper_perms:
            for lower in lower_perms:
                assert len(upper) == len(lower)

                s_list = []
                # for each s operator we make a `connected_namedtuple` or a `disconnected_namedtuple`
                for i in range(len(upper)):
                    s_kwargs = {
                        'm_o': upper[i][0], 'm_h': upper[i][1],
                        'n_o': lower[i][0], 'n_h': lower[i][1],
                    }

                    # no connection to Hamiltonian means this term is disconnected
                    if s_kwargs['m_h'] == 0 and s_kwargs['n_h'] == 0:
                        s_list.append(disconnected_namedtuple(**s_kwargs))

                    # any connection to Hamiltonian means this term is connected
                    else:
                        s_list.append(connected_namedtuple(**s_kwargs))

                log.debug(f"{s_list}")
                annotated_permutations.append(s_list)

    return annotated_permutations


def _remove_duplicate_s_permutations(s_list):
    """ x """
    duplicate_set = set()

    for i, a in enumerate(s_list):
        a.sort()
        a = tuple(a)
        # old_print_wrapper(h_string, i, a)
        if a not in duplicate_set:
            # old_print_wrapper("Added")
            duplicate_set.add(a)
        else:
            pass
            # log.debug("Duplicate")

    return duplicate_set


def _generate_explicit_connections(omega, h, unique_s_permutations):
    """ Generate new namedtuples for omega and h explicitly labeling how they connect with each other and t.
    We make `connected_omega_operator_namedtuple` and `connected_h_operator_namedtuple`.
    The output `labeled_permutations` is a list where each element is `[new_omega, new_h, s_list]`.
    We also check to make sure each term is valid.
    """

    labeled_permutations = []  # store output here

    for s_list in unique_s_permutations:

        o_kwargs = {'m_t': [s.n_o for s in s_list], 'n_t': [s.m_o for s in s_list]}
        h_kwargs = {'m_t': [s.n_h for s in s_list], 'n_t': [s.m_h for s in s_list]}

        o_kwargs.update({'rank': omega.m + omega.n, 'm': omega.m, 'n': omega.n})
        h_kwargs.update({'rank': h.m + h.n, 'm': h.m, 'n': h.n})

        o_kwargs['m_h'] = o_kwargs['m'] - sum(o_kwargs['m_t'])
        o_kwargs['n_h'] = o_kwargs['n'] - sum(o_kwargs['n_t'])
        h_kwargs['m_o'] = h_kwargs['m'] - sum(h_kwargs['m_t'])
        h_kwargs['n_o'] = h_kwargs['n'] - sum(h_kwargs['n_t'])

        assert o_kwargs['m_h'] >= 0 and o_kwargs['n_h'] >= 0
        assert h_kwargs['m_o'] >= 0 and h_kwargs['n_o'] >= 0

        if h_kwargs['m_o'] != o_kwargs['n_h']:
            term_string = f"{tab}{omega}, {h}, {s_list}\n{tab}{o_kwargs=}\n{tab}{h_kwargs=}\n"
            log.debug(f"Found an invalid term (h.m_o != o.n_h)\n{term_string}")
            continue

        elif h_kwargs['n_o'] != o_kwargs['m_h']:
            term_string = f"{tab}{omega}, {h}, {s_list}\n{tab}{o_kwargs=}\n{tab}{h_kwargs=}\n"
            log.debug(f"Found an invalid term (h.n_o != o.m_h)\n{term_string}")
            continue

        new_omega = connected_omega_operator_namedtuple(**o_kwargs)
        new_h = connected_h_operator_namedtuple(**h_kwargs)

        labeled_permutations.append([new_omega, new_h, s_list])

    return labeled_permutations


def _remove_f_zero_terms(labeled_permutations):
    """ Remove terms which have a db contraction (generating an f term) """

    return_list = []

    for term in labeled_permutations:
        omega, h, perm = term

        # if omega is contracting with h like `db` this generates an f prefactor
        # therefore we remove this term
        if omega.m_h >= 1:
            log.debug(f"Found a f term {omega} {h} {perm}")
            continue
        else:
            return_list.append(term)

    return return_list


def _filter_out_valid_s_terms(omega, H, s_series_term, term_list, total_list, remove_f_terms=True):
    """ fill up the `term_list` and `total_list` for the S^n term
    first we find out what term (in the taylor expansion of e^S) `s_series_term` represents
    set a boolean flag, and wrap the lower order terms in lists so that they have the same
    structure as the s_n case (a list of lists of `general_operator_namedtuple`s)
    """

    # S^0 operator is simply 1 in this case
    if isinstance(s_series_term, general_operator_namedtuple):
        log.info("S^0\n")
        s_order = 0
        s_series_term = [[s_series_term, ], ]  # wrap in a list of lists

    # S^1 operator, straightforward
    elif isinstance(s_series_term, list) and isinstance(s_series_term[0], general_operator_namedtuple):
        log.info("S^1\n")
        s_order = 1
        s_series_term = [[term, ] for term in s_series_term]  # wrap in a list

    # S^n operator, most complicated
    elif isinstance(s_series_term, list) and isinstance(s_series_term[0], list) and len(s_series_term[0]) >= 2:
        log.info("S^n\n")
        s_order = 'n'
        # no wrapping necessary

    # next we process the s operators inside s_series_term
    for h in H.operator_list:

        if True:  # debug
            nof_terms = sum([len(x) for x in s_series_term])
            if s_order == 0 or nof_terms < 5:
                log.debug(f"Checking the S^{s_order} term: {omega} {h} {s_series_term}")
            else:
                log.debug(f"Checking the S^{s_order} term: {omega} {h} number of s terms: {nof_terms}")

        # valid pairings of s operators given a specific `omega` and `h`
        valid_permutations = _generate_valid_s_n_operator_permutations(omega, h, s_series_term)

        # if no valid operators continue to the next h
        if valid_permutations == []:
            continue

        # we need to generate all possible combinations of s with the omega and h operators
        s_connection_permutations = _generate_all_omega_h_connection_permutations(omega, h, valid_permutations)

        # remove all duplicate permutations
        unique_s_permutations = _remove_duplicate_s_permutations(s_connection_permutations)

        # generate all the explicit connections
        # this also removes all invalid terms
        labeled_permutations = _generate_explicit_connections(omega, h, unique_s_permutations)

        """ We check for all h operators that generate an f prefactor, i.e. a db contraction.
        f = 0 that implies this term is zero and so we remove it from the sum.
        We have a flag because we might want to leave the terms with a f prefactor so that they
        they are included in the final latex code, for checking purposes.
        To even generate an f term, omega.m and h.n both have to be at least 1
        """
        if remove_f_terms and (omega.m >= 1 and h.n >= 1):
            labeled_permutations = _remove_f_zero_terms(labeled_permutations)

        # we record
        for term in labeled_permutations:
            log.debug(f"{term=}")
            if term[2] != set():
                # if it is not an empty set
                total_list.append(term)

    return


def _seperate_s_terms_by_connection(total_list):
    """ x """
    fully, linked, unlinked = [], [], []

    for term in total_list:
        omega, h, s_term_list = term
        linked_flag = False

        # simple check for h terms with no t terms
        if (len(s_term_list) == 1) and s_term_list[0] == disconnected_namedtuple(0, 0, 0, 0):
            fully.append(term)
            continue

        for i, s in enumerate(s_term_list):

            if isinstance(s, connected_namedtuple):
                continue  # continue to check if each s in term is connected

            elif isinstance(s, disconnected_namedtuple):

                # unlinked disconnected
                if (s.m_o == omega.n) and (s.n_o == omega.m):  # probably need to improve this
                    # if even one s in the term is unlinked disconnected
                    # then the whole group is unlinked disconnected
                    unlinked.append(term)
                    break

                # linked disconnected
                else:
                    # this shouldn't happen, but we check just in case
                    if omega.rank == 1:
                        old_print_wrapper('??', s, term)
                        raise Exception("Linear terms should always be connected or disconnected")

                    """ We record that we found a linked disconnected term, but we don't stop here.
                    We still need to check all s terms in case there is another s in the `term`
                    that is unlinked disconnected, in which case this `term` would be unlinked.
                    """
                    linked_flag = True
                    continue

            # this shouldn't happen, but we check just in case
            else:
                old_print_wrapper('??', s, term)
                raise Exception("term contains something other than connected/disconnected namedtuple??\n")

        # if we never broke out of the loop this implies we didn't see any unlinked disconnected terms
        else:

            # if we saw at least one linked disconnected term
            if linked_flag:
                linked.append(term)

            # we saw no disconnected terms
            else:
                fully.append(term)

    return fully, linked, unlinked


# --------------- assigning of upper/lower latex indices ------------------------- #
def _build_h_term_latex_labels(h, condense_offset=0, color=True):
    """ Builds latex code for labeling a `connected_h_operator_namedtuple`.

    The `condense_offset` is an optional argument which is needed when creating latex code
    for linked disconnected terms in a condensed format.
    """
    if h.rank == 0:
        return f"{bold_h_latex}_0"

    string = bold_h_latex

    # do the upper indices first
    if not color:
        upper_indices = summation_indices[0:h.m - h.m_o]
        upper_indices += unlinked_indices[condense_offset:condense_offset+h.m_o]
    else:
        upper_indices = r'\blue{' + summation_indices[0:h.m - h.m_o] + '}'
        upper_indices += r'\red{' + unlinked_indices[condense_offset:condense_offset+h.m_o] + '}'
    string += f"^{{{upper_indices}}}"

    # now do the lower indices
    h_offset = h.m - h.m_o
    if not color:
        lower_indices = summation_indices[h_offset:h_offset + (h.n - h.n_o)]
        lower_indices += unlinked_indices[condense_offset+h.m_o:condense_offset+h.m_o+h.n_o]
    else:
        lower_indices = r'\blue{' + summation_indices[h_offset:h_offset + (h.n - h.n_o)] + '}'
        lower_indices += r'\red{' + unlinked_indices[condense_offset+h.m_o:condense_offset+h.m_o+h.n_o] + '}'
    string += f"_{{{lower_indices}}}"

    return string


def _build_t_term_latex_labels(term, offset_dict, color=True):
    """Return a latex string representation of `s_term`.
    Should use `_build_t_term_latex` primarily to build single t-strings.

    It is CRITICAL to remember that `h_m` represents pairing with h^m indices; therefore
    the indices need to be represented as a subscript on the t_term since h^m pairs with t_n.
    The `term.h_m` attribute is a count of how many t_n subscripts pair with h^m superscripts.

    Also remember that in `_build_latex_h_string` the h term assigns indices to the superscripts
    before the subscripts. So to match our indices we need to assign indices connected to h in the
    opposite order.
    """
    up_label, down_label = "", ""

    # subscript indices
    if (term.n_h > 0) or (term.n_o > 0):
        a, b = offset_dict['summation_lower'], offset_dict['unlinked']

        if not color:
            down_label = summation_indices[a:a+term.n_h] + unlinked_indices[b:b+term.n_o]
        else:
            down_label = r'\blue{' + summation_indices[a:a+term.n_h] + '}'
            down_label += r'\red{' + unlinked_indices[b:b+term.n_o] + '}'

        # record the change in the offset
        offset_dict['summation_lower'] += term.n_h
        offset_dict['unlinked'] += term.n_o

    # superscript indices
    if (term.m_h > 0) or (term.m_o > 0):
        a, b = offset_dict['summation_upper'], offset_dict['unlinked']

        if not color:
            up_label = summation_indices[a:a+term.m_h] + unlinked_indices[b:b+term.m_o]
        else:
            up_label = r'\blue{' + summation_indices[a:a+term.m_h] + '}'
            up_label += r'\red{' + unlinked_indices[b:b+term.m_o] + '}'

        # record the change in the offset
        offset_dict['summation_upper'] += term.m_h
        offset_dict['unlinked'] += term.m_o

    return f"^{{{up_label}}}_{{{down_label}}}"


def _build_t_term_latex(s, h=None):
    """ Wrapper for `_build_t_term_latex_group`.
    If we want to get the latex for a single t term.
    """
    lst = _build_t_term_latex_group([s, ], h)
    return lst[0]


def _build_t_term_latex_group(s_list, h=None, offset_dict=None):
    """Return a list of strings for each term in `s_list`

    if the t term we are generating is paired with an h term
    then we need to account for the sub/super script index ordering
    by offsetting the upper labels of the t term by the number of upper
    labels of the h term. However we don't offset the lower labels.

    We need to account for the `h.m` and `h.n` attributes.
    We do that by passing the h term to `_build_t_term_latex_labels`.
    """

    t_list = []

    # count
    if offset_dict is None:
        offset_dict = {'summation_upper': 0, 'summation_lower': 0, 'unlinked': 0}

    # here we have to account for the indices already use in h
    if h is not None:
        offset_dict['summation_upper'] += (h.m - h.m_o)
        offset_dict['unlinked'] += h.m_o + h.n_o

    log.debug(offset_dict)

    for s in s_list:
        t_labels = _build_t_term_latex_labels(s, offset_dict)
        log.debug(s, t_labels, offset_dict)

        if t_labels != "^{}_{}":
            t_list.append(bold_t_latex + t_labels)

    return t_list


# ------------------------  the writing of latex ------------------------------- #
def _validate_s_terms(s_list):
    # make sure all s_terms are valid objects
    for s_term in s_list:
        assert isinstance(s_term, (connected_namedtuple, disconnected_namedtuple)), (
                "The following term is not a connected_namedtuple"
                f" or a disconnected_namedtuple but a {type(s_term)}\n"
                f"{s_term}\n"
            )
    return


def _generate_linked_common_terms(term_list):
    """ x """
    aset = set()
    omega = term_list[0][0]
    # length = max(omega.m, omega.n)

    for x in term_list:
        o, t = x[0], x[2]
        # old_print_wrapper('x', o, t)

        a = [t[i] for i in range(len(o.m_t)) if isinstance(t[i], disconnected_namedtuple)]
        # old_print_wrapper('a', a)
        aset.add(tuple(a))

    old_print_wrapper(f'Final {omega}\n', aset,)
    # for x in aset:
    #     old_print_wrapper([f't^{y.m_o}_{y.n_o}' for y in x])

    # make sure the list is in descending order
    z = sorted([list(x) for x in aset], key=len, reverse=True)
    # old_print_wrapper('\n\n')

    # for a in z:
    #     old_print_wrapper(len(a))

    # old_print_wrapper(z)
    # old_print_wrapper('\n\n')

    # old_print_wrapper(z)
    # sys.exit(0)
    return z

    # old_print_wrapper('\n\n')
    # for x in an:
    #     old_print_wrapper([f't^{y.m_o}_{y.n_o}' for y in x])

    # if omega.m > 0:
    #     old_print_wrapper(omega)
    #     sys.exit(0)

    return


def prepare_condensed_terms(term_list, linked_condense=False, unlinked_condense=False):
    """ x """

    # extract information about omega
    omega = term_list[0][0]

    # straightforward
    if unlinked_condense:
        log.warning("This has not been tested for Hamiltonians of rank >= 3")

        # create the common factor
        common_factor = disconnected_namedtuple(m_h=0, n_h=0, m_o=omega.n, n_o=omega.m)

        for term in term_list:
            s_list = term[2]
            # make sure that it is present
            assert common_factor in s_list, f"{common_factor=} not present in {s_list=}"

            # remove the common factor
            # del s_list[s_list.index(common_factor)]

        return common_factor

    # more complicated
    elif linked_condense:

        assert omega.rank in [2, 3], (
            "Condensing linked disconnected terms "
            "is only supported for omega's of rank (2,3) "
            f"not rank {omega.rank}"
        )

        if omega.rank == 1:
            raise Exception("Linear omega operators (omega.rank == 1) do not have linked disconnected terms!")

        if omega.rank == 2:
            linked_commonfactor_list = _generate_linked_common_terms(term_list)

            # we store the different common factors in here
            # linked_commonfactor_list = [None, ]*omega.rank

            # old_print_wrapper(omega)
            # for char in omega.name:
            #     if char == "b":
            #         term = disconnected_namedtuple(h_m=0, h_n=0, o_m=1, o_n=0)
            #     elif char == "d":
            #         term = disconnected_namedtuple(h_m=0, h_n=0, o_m=0, o_n=1)

            #     linked_commonfactor_list.append(term)

            # # # we store the latex representations of the common factors in here
            # # linked_commonfactor_latex_list = _build_t_term_latex_group(linked_commonfactor_list)
            # old_print_wrapper(omega, linked_commonfactor_list)

            # generate common factor lists here
            # _generate_linked_common_terms(term_list)

        if omega.rank == 3:
            linked_commonfactor_list = _generate_linked_common_terms(term_list)

        return linked_commonfactor_list


def _build_latex_prefactor(h, t_list, simplify_flag=True):
    """Attempt to return latex code representing appropriate prefactor term.

    All prefactors begin with 1/n! where n is the number of t amplitudes in given term.
    This comes from the prefactors of the taylor expansion of e^S = 1 + S^1 + S^2 ....
    We also need to account for the prefactors of the Hamiltonian as defined in the theory.
    There are also permutations of indices to take into account.
    Overall the entire affair is quite complicated.

    In general the rules are:
        1 - 1/n! where n is the number of t terms (taylor series)
        1 - All t terms are balanced/neutral (they contribute m!n!/m!n! = 1)
        1 - h contributes x!/m!n! where x is the number of UNIQUE t terms that h contracts with

    A single h with no t list is a special case where the prefactor is always 1.
    """

    # special case, single h
    if len(t_list) == 1 and t_list[0] == disconnected_namedtuple(0, 0, 0, 0):
        return ''

    numerator_list = []
    denominator_list = []

    # account for prefactor from Hamiltonian term `h`
    connected_ts = [t for t in t_list if t.m_h > 0 or t.n_h > 0]
    x = len(set(connected_ts))

    debug_flag = bool(
        h.n == 2
        and h.m == 0
        and len(t_list) == 2
        and t_list[0].m_h == 1
        and t_list[0].m_o == 1
        and t_list[1].m_h == 1
        and t_list[1].m_o == 1
    )

    if debug_flag:
        old_print_wrapper('\n\n\nzzzzzzzzz')
        old_print_wrapper(connected_ts)
        old_print_wrapper(set(connected_ts))
        old_print_wrapper(len(set(connected_ts)))
        old_print_wrapper('zzzzzzzzz\n\n\n')

    if x > 1:
        numerator_list.append(f'{x}!')
    if h.m > 1:
        denominator_list.append(f'{h.m}!')
    if h.n > 1:
        denominator_list.append(f'{h.n}!')

    # account for the number of permutations of all t-amplitudes
    if len(t_list) > 1:
        denominator_list.append(f'{len(t_list)}!')

    # simplify
    if simplify_flag:
        numerator_list, denominator_list = _simplify_full_cc_python_prefactor(numerator_list, denominator_list)

    # glue the numerator and denominator together
    numerator = '1' if (numerator_list == []) else f"{''.join(numerator_list)}"
    denominator = '1' if (denominator_list == []) else f"{''.join(denominator_list)}"

    if numerator == '1' and denominator == '1':
        return ''
    else:
        return f"\\frac{{{numerator}}}{{{denominator}}}"


def _linked_condensed_adjust_t_terms(common_linked_factor_list, h, t_list):
    """ x """

    t_offset_dict = {
        'summation_upper': 0,
        'summation_lower': 0,
        'unlinked': 0,
    }

    linked_list_index = 0

    for i, factor in enumerate(common_linked_factor_list):
        indxs, offset = [], 0
        try:
            for t in factor:
                indxs.append(t_list.index(t, offset))
                offset = indxs[-1] + 1

        except ValueError as e:
            if 'not in' in str(e):
                continue
            else:
                log.debug("Wrong Error?")
                raise e
        else:
            if (indxs != []) and len(set(indxs)) == len(indxs):
                linked_list_index = i

                # remove the common factors
                new_t_list = [t for i, t in enumerate(t_list) if i not in indxs]

                if 0 == h.m_o == h.n_o:
                    t_offset_dict['unlinked'] += sum([t.m_o + t.n_o for t in factor])

                # need to create offset dictionary to change remaining t terms
                else:
                    t_offset_dict['unlinked'] += h.m_o + h.n_o + sum([t.m_o + t.n_o for t in factor])

                break

            else:
                new_t_list = t_list

    old_print_wrapper(new_t_list)
    return linked_list_index, new_t_list, t_offset_dict


def _creates_f_prefactor(omega, h):
    """ Define this check as a function to allow for modification later.
    For example if we change the definition of d and b operators or do the thermal theory later.
    """
    return bool(omega.m_h >= 1 and h.n_o >= 1)


# added by shanmei
def _creates_fbar_prefactor(omega, h):
    """ Define this check as a function to allow for modification later.
    For example if we change the definition of d and b operators or do the thermal theory later.
    """
    return bool(omega.n_h >= 1 and h.m_o >= 1)


def _make_latex(rank, term_list, linked_condense=False, unlinked_condense=False, print_prefactors=True, color=False):
    """Return the latex commands to write the provided terms.

    the `color` argument in this case wraps all disconnected terms in a `\\colorbox{yellow}` if True
    """

    return_list = []  # store output here

    # prepare the common factors for condensing
    if unlinked_condense:
        common_unlinked_factor = prepare_condensed_terms(term_list, unlinked_condense=True)

    elif linked_condense:
        common_linked_factor_list = prepare_condensed_terms(term_list, linked_condense=True)
        # where we store the lists of latex code
        linked_return_list = [[] for i in range(len(common_linked_factor_list))]

    # prepare all the latex strings
    for term in term_list:
        # extract elements of list `term`
        omega, h, t_list = term[0], term[1], term[2]
        old_print_wrapper(type(t_list), t_list)

        # make sure all s_terms are valid objects
        _validate_s_terms(t_list)

        term_string = ''

        # if needed add f prefactors
        # if h has unpaired lower terms this implies it would contract with omega in a `db` fashion
        if _creates_f_prefactor(omega, h):
            if omega.m_h == 1:
                term_string += "f"
            else:
                term_string += f"f^{{{omega.m_h}}}"

        # added by shanmei, which of f and fbar should go first?
        # if needed add fbar prefactors
        # if h has unpaired upper terms this implies it would contract with omega in a `bd` fashion
        if _creates_fbar_prefactor(omega, h):
            if omega.n_h == 1:
                term_string += "\\bar{f}"
            else:
                term_string += f"\\bar{{f}}^{{{omega.n_h}}}"

        # add any prefactors if they exist
        if print_prefactors:
            term_string += _build_latex_prefactor(h, t_list)

        # special treatment for condensing the linked disconnected terms
        if linked_condense:
            linked_list_index, new_t_list, t_offset_dict = _linked_condensed_adjust_t_terms(common_linked_factor_list, h, t_list)

            # prepare the t-amplitude terms
            t_string_list = _build_t_term_latex_group(new_t_list, h, t_offset_dict)

            # build the latex code representing this term in the sum
            h_offset = sum([t.m_o + t.n_o for t in common_linked_factor_list[linked_list_index]])
            term_string += _build_h_term_latex_labels(h, h_offset) + ''.join(t_string_list)

            # store the result
            linked_return_list[linked_list_index].append(term_string)

        else:
            # prepare the t-amplitude terms
            t_string_list = _build_t_term_latex_group(t_list, h=h)

            # build the latex code representing this term in the sum
            term_string += _build_h_term_latex_labels(h) + ''.join(t_string_list)

            # store the result
            return_list.append(term_string)

    # glue all the latex code together!!

    # special treatment to condense the unlinked disconnected terms
    if unlinked_condense:
        log.warning("This has not been tested for Hamiltonians of rank >= 3")

        if not color:
            # remove the common factor from each term
            common_latex = _build_t_term_latex(common_unlinked_factor)
            return_list = [term.replace(common_latex, '') for term in return_list]
        # adding colour
        else:
            common_latex = _build_t_term_latex(common_unlinked_factor)
            return_list = [r'\disconnected{' + term.replace(common_latex, '') + r'}' for term in return_list]

        # old_print_wrapper(return_list)

        return f"({' + '.join(return_list)}){common_latex}"

    # special treatment to condense the linked disconnected terms
    if linked_condense:
        log.warning("This has not been tested for Hamiltonians of rank >= 3")

        return_strings = []

        for i, factor in enumerate(common_linked_factor_list):
            common_latex = ''.join(_build_t_term_latex_group(factor))
            # old_print_wrapper('z', i, linked_return_list[i], '\n\n')
            # old_print_wrapper(factor)

            # if common_latex == '\\bt^{z}_{}\\bt^{y}_{}':
            #     old_print_wrapper('z', common_latex)
            #     for term in linked_return_list[i]:
            #         if common_latex in term:
            #             old_print_wrapper('\n\n', term)
            #             sys.exit(0)

            #             if "\\bh^{z}_{}\\bt^{}_{y}\\bt^{x}_{}" in term:
            #                 old_print_wrapper('\n\n', term)
            #                 sys.exit(0)

            # linked_return_list[i] = [term.replace(common_latex, '') for term in linked_return_list[i]]
            # old_print_wrapper('z', i, linked_return_list[i])
            return_strings.append(f"({' + '.join(linked_return_list[i])}){common_latex}")

        # join the lists with the equation splitting string
        splitting_string = r'\\  &+  % split long equation'
        final_string = f"\n{tab}{splitting_string}\n".join(return_strings)

        return final_string

        # if rank == 1:
        #     return _glue_linked_disconnected_latex_together(rank, term_list, common_linked_factor_list, linked_return_list)
        # if rank == 2:
        #     return _glue_linked_disconnected_latex_together(rank, term_list, common_linked_factor_list, linked_return_list)
        # if rank == 3:
        #     return _glue_linked_disconnected_latex_together(rank, term_list, common_linked_factor_list, linked_return_list)

    # otherwise we simply glue everything together
    else:
        # the maximum number of terms on 1 horizontal line (in latex)
        # change this as needed to fit equations on page
        split_number = 7

        # if the line is so short we don't need to split
        if len(return_list) < split_number*2:
            return f"({' + '.join(return_list)})"

        # make a list of each line
        split_equation_list = []
        for i in range(0, len(return_list) // split_number):
            split_equation_list.append(' + '.join(return_list[i*split_number:(i+1)*split_number]))

        # join the lists with the equation splitting string
        splitting_string = r'\\  &+  % split long equation'
        final_string = f"\n{tab}{splitting_string}\n".join(split_equation_list)

        # and we're done!
        return f"(\n{final_string}\n)"


def _write_cc_latex_from_lists(rank, fully, linked, unlinked):
    """Return the latex commands to write the provided terms.
    We use `join` to insert two backward's slashes \\ BETWEEN each line
    rather then adding them to end and having extra trailing slashes on the last line.
    The user is expected to manually copy the relevant lines from the text file into a latex file
    and generate the pdf themselves.
    """
    return_string = ""

    # special case for zero order equation
    if rank == 0:
        return_string += ' + '.join([
            _make_latex(rank, fully),
            _make_latex(rank, linked),
            _make_latex(rank, unlinked),
        ])
        return return_string.replace("^{}", "").replace("_{}", "")

    # no ____ terms
    no_fully = ' '*4 + r'\textit{no fully connected terms}'
    no_linked = ' '*4 + r'\textit{no linked disconnected terms}'
    no_unlinked = ' '*4 + r'\textit{no unlinked disconnected terms}'

    return_string += _make_latex(rank, fully) if fully != [] else no_fully
    return_string += '\n%\n%\n\\\\  &+\n%\n%\n'

    """ special treatment for linear, quadratic, and cubic
    What we are doing is grouping the linked disconnected terms into groups such as:
    (.... )t^k + (....)t_k + (...)t^k_j and so forth.
    This is purely for readability / visual appeal.
    For any other `rank` we explicitly print all linked disconnected terms.
    """
    if rank > 1:
        return_string += _make_latex(rank, linked, linked_condense=True) if linked != [] else no_linked
    else:
        return_string += _make_latex(rank, linked, linked_condense=False) if linked != [] else no_linked

    return_string += '\n%\n%\n\\\\  &+\n%\n%\n'
    return_string += _make_latex(rank, unlinked, unlinked_condense=True) if unlinked != [] else no_unlinked

    # remove all empty ^{}/_{} terms that are no longer needed
    return return_string.replace("^{}", "").replace("_{}", "")


# ------------------------------------------------------------------------ #
def _generate_cc_latex_equations(omega, H, s_taylor_expansion, remove_f_terms=True):
    """Return a string containing latex code to be placed into a .tex file.
    For a given set of input arguments: (`omega`, `H`, `s_taylor_expansion`) we generate
    all possible and valid CC terms. Note that:
        - `omega` is an `omega_namedtuple` object
        - `H` is a `hamiltonian_namedtuple` object
        - `s_taylor_expansion` is one of :
            - a single `general_operator_namedtuple`
            - a list of `general_operator_namedtuple`s
            - a list of lists of `general_operator_namedtuple`s

    One possible input could be:
        - `omega` is the creation operator d
        - `H` is a Hamiltonian of rank two
        - `s_taylor_expansion` is the S^1 Taylor expansion term
    """

    simple_repr_list = []  # old list, not so important anymore, might remove in future
    valid_term_list = []   # store all valid Omega * h * (s*s*...s) terms here

    """ First we want to generate a list of valid terms.
    We start with the list of lists `s_taylor_expansion` which is processed by `_filter_out_valid_s_terms`.
    This function identifies valid pairings AND places those pairings in the `valid_term_list`.
    Specifically we replace the `general_operator_namedtuple`s with `connected_namedtuple`s and/or
    `disconnected_namedtuple`s.
    """
    for count, s_series_term in enumerate(s_taylor_expansion):

        if False:  # debugging
            old_print_wrapper(s_series_term, "-"*100, "\n\n")

        _filter_out_valid_s_terms(omega, H, s_series_term, simple_repr_list, valid_term_list, remove_f_terms=remove_f_terms)

    """ Next we take all terms and separate them into their respective groups """
    fully, linked, unlinked = _seperate_s_terms_by_connection(valid_term_list)

    if False:  # extra heavy debugging
        _debug_print_valid_term_list(valid_term_list)
        _debug_print_different_types_of_terms(fully, linked, unlinked)

    # write and return the latex code
    return _write_cc_latex_from_lists(omega.rank, fully, linked, unlinked)


# ------------------------------------------------------------------------ #

def _generate_left_hand_side(omega):
    """ Generate the latex code for the LHS (left hand side) of the CC equation.
    The order of the `omega` operator determines all terms on the LHS.
    """

    omega_order = omega.m + omega.n

    if omega_order == 0:
        return r'''i\left(\varepsilon\right)'''

    # generate all possible tuples (m, n) representing t terms t^m_n
    single_t_list = [[m, n] for m in range(0, omega_order+1) for n in range(0, omega_order+1) if ((n == m != 0) or (n != m))]

    all_combinations_list = []

    # generate all possible combinations of t^m_n
    # such as t_1, t^2, t^1 * t^1, t^1 * t^2_3, ... etc
    for length in range(1, omega_order+1):
        all_combinations_list.append(list(it.product(single_t_list, repeat=length)))

    """ Next we filter out the combinations that don't match omega.
    Suppose omega is o^2_1:
        - it can match with  (t^1 * t^1 * t_1) or (t^2_1) and so forth
        - it cant match with (t^1 * t^1 * t^1) or (t^1_1) and so forth
    """
    matched_set = set()
    for list_of_t_terms in all_combinations_list:
        for t_terms in list_of_t_terms:
            upper_sum = sum([t[0] for t in t_terms])  # the sum over all m superscripts (t^m)
            lower_sum = sum([t[1] for t in t_terms])  # the sum over all n superscripts (t_n)

            # remember that a t_1 contracts with o^1
            # so the `lower_sum` needs to be compared to o^n
            if omega.m == lower_sum and omega.n == upper_sum:
                """ The "filtering" is accomplished by adding sorted tuples of tuples to a set.
                We cannot use lists because sets require hashable elements (immutable) such as tuples.
                However with tuples, we can end up with duplicates like ((2, 0), (0, 1)) and ((0, 1), (2, 0)).
                So we have to:
                    - generate lists of tuples:             [(0, 1), (0, 2)]
                    - sort those lists in reverse order:    [(0, 2), (0, 1)]
                    - make a tuple from the list:           ((0, 2), (0, 1))
                    - add the tuple to `matched_set`
                """
                matched_set.add(tuple(sorted([tuple(x) for x in t_terms], reverse=True)))

    """ Transform the set into a list of lists sort by increasing length
    Two examples:
        - omega is `operator(name='bb', m=0, n=2)`
        then `sorted_list` is [[(2, 0)], [(1, 0), (1, 0)]]

        - omega is `operator(name='ddd', m=3, n=0)`
        then `sorted_list` is [[(0, 3)], [(0, 2), (0, 1)], [(0, 1), (0, 1), (0, 1)]]
    """
    sorted_list = sorted([[b for b in a] for a in matched_set], key=len)

    """ Next we generate the latex code for each t term represented by the (m, n) tuples
    One possible `t_group` could be:
        - [[(0, 2)], [(0, 1), (0, 1)]]
    and its corresponding `group_list` would be:
        - [['\\bt^{}_{ij}'], ['\\bt^{}_{i}', '\\bt^{}_{j}']]
    """
    latex_t_terms_list = []  # store the latex code for each valid t term here
    for t_group in sorted_list:
        count = 0  # keep track of what index labels have been used
        group_list = []

        for t in t_group:
            upper_label = summation_indices[count:count+t[0]]
            count += t[0]

            lower_label = summation_indices[count:count+t[1]]
            count += t[1]

            group_list.append(f"{bold_t_latex}^{{{upper_label}}}_{{{lower_label}}}")

        latex_t_terms_list.append(group_list)

    # create epsilon terms
    epsilon_list = [''.join(term) + r'\varepsilon' for term in latex_t_terms_list]

    # create derivative terms
    derivative_list = []
    for t_group in latex_t_terms_list:
        # loop over each t in the group and take the derivative of that specific t
        for i in range(len(t_group)):
            string = ""

            # if there are t terms to the left of the current index
            if len(t_group[0:i]) > 0:
                string += f"{''.join(t_group[0:i])}"

            # the t term we are currently taking the derivative of
            string += rf'\dv{{{t_group[i]}}}{{\tau}}'

            # if there are t terms to the right of the current index
            if len(t_group[i:]) > 0:
                string += f"{''.join(t_group[i+1:])}"

            derivative_list.append(string)

    # order the derivative terms before the epsilon terms
    return_string = ' + '.join([*derivative_list, *epsilon_list])
    return rf'''i\left({return_string}\right)'''


def _wrap_align_environment(omega, rank_name, lhs, eqns):
    """ x """

    omega_string = ""

    for i, char in enumerate(omega.name):
        if char == "d":
            omega_string += f'\\up{{{summation_indices[i]}}}'
        elif char == "b":
            omega_string += f'\\down{{{summation_indices[i]}}}'

    if omega_string == "":
        omega_string = "1"

    string = (
        '\\begin{align}\\begin{split}\n'
        f'{tab}\\hat{{\\Omega}} = {omega_string}\n'
        r'\\ LHS &='
        '\n'
        f"{tab}{lhs}\n"
        r'\\ RHS &='
        '\n%\n%\n'
        f'{eqns}\n'
        r'\end{split}\end{align}'
        '\n\n'
    )

    return string


def generate_full_cc_latex(truncations, only_ground_state=False, path="./generated_latex.txt"):
    """Generates and saves to a file the latex equations for full CC expansion."""

    assert len(truncations) == 4, "truncations argument needs to be tuple of four integers!!"
    maximum_h_rank, maximum_cc_rank, s_taylor_max_order, omega_max_order = truncations

    master_omega = generate_omega_operator(maximum_cc_rank, omega_max_order)
    H = generate_full_cc_hamiltonian_operator(maximum_h_rank)
    s_taylor_expansion = generate_s_taylor_expansion(maximum_cc_rank, s_taylor_max_order, only_ground_state)

    latex_code = ""  # store result in here

    rank_name_list = [
        "0 order", "LINEAR", "QUADRATIC", "CUBIC", "QUARTIC", "QUINTIC", "SEXTIC", "SEPTIC", "OCTIC"
    ]

    for i, omega_term in enumerate(master_omega.operator_list):

        # for debugging purposes
        # if you only want to generate the linear terms for example; change the False to True
        if False and omega_term.rank not in [1, ]:
            continue

        # generate the i(dt/dtau + t*epsilon) latex
        lhs_string = _generate_left_hand_side(omega_term)

        # where we do all the work of generating the latex
        equations_string = _generate_cc_latex_equations(omega_term, H, s_taylor_expansion, remove_f_terms=False)

        # header for the sub section
        latex_code += '%\n%\n%\n%\n%\n\n'
        rank_name = rank_name_list[omega_term.rank]

        if omega_term.rank > master_omega.operator_list[i-1].rank:
            latex_code += '\\newpage\n' if omega_term.rank >= 2 else ''
            latex_code += f'\\subsection{{{rank_name.capitalize()} Equations}}\n\n'

        latex_code += _wrap_align_environment(omega_term, rank_name, lhs_string, equations_string)

    if only_ground_state:
        # use the predefined header in `reference_latex_headers.py`
        header = headers.ground_state_vecc_latex_header
    else:
        # use the predefined header in `reference_latex_headers.py`
        header = headers.full_vecc_latex_header

    # write the new header with latex code attached
    with open(path, 'w') as fp:
        fp.write(header + latex_code + r'\end{document}')

    return


# ----------------------------------------------------------------------------------------------- #
# -------------------------  GENERATING FULL CC PYTHON EQUATIONS  ------------------------------- #
# ----------------------------------------------------------------------------------------------- #

def _rank_of_t_term_namedtuple(t):
    """ Calculate the rank of a `connected_namedtuple` or `disconnected_namedtuple`. """
    return sum([v for v in t._asdict().values()])


# ------------------------------------------------------- #
def _full_cc_einsum_electronic_components(t_list):
    """ x """
    electronic_surface_indices = 'cdefgh'
    electronic_components = ['ac', ]

    for i in range(len(t_list)):
        electronic_components.append(electronic_surface_indices[i:i+2])

    # change the last term to end with `b`
    electronic_components[-1] = electronic_components[-1][0] + 'b'

    return electronic_components


def _build_h_term_python_labels(h, condense_offset=0):
    """ x """
    sum_label, unlinked_label = "", ""

    if h.rank == 0:
        return sum_label, unlinked_label

    # do the upper indices first
    sum_label += summation_indices[0:h.m - h.m_o]
    unlinked_label += unlinked_indices[condense_offset:condense_offset+h.m_o]

    # now do the lower indices
    h_offset = h.m - h.m_o
    sum_label += summation_indices[h_offset:h_offset + (h.n - h.n_o)]
    unlinked_label += unlinked_indices[condense_offset+h.m_o:condense_offset+h.m_o+h.n_o]

    return sum_label, unlinked_label


def _build_t_term_python_labels(term, offset_dict):
    """ x """
    sum_label, unlinked_label = "", ""

    # subscript indices
    if (term.n_h > 0) or (term.n_o > 0):
        a, b = offset_dict['summation_lower'], offset_dict['unlinked']

        sum_label += summation_indices[a:a+term.n_h]
        unlinked_label += unlinked_indices[b:b+term.n_o]

        # record the change in the offset
        offset_dict['summation_lower'] += term.n_h
        offset_dict['unlinked'] += term.n_o

    # superscript indices
    if (term.m_h > 0) or (term.m_o > 0):
        a, b = offset_dict['summation_upper'], offset_dict['unlinked']

        sum_label += summation_indices[a:a+term.m_h]
        unlinked_label += unlinked_indices[b:b+term.m_o]

        # record the change in the offset
        offset_dict['summation_upper'] += term.m_h
        offset_dict['unlinked'] += term.m_o

    return sum_label, unlinked_label


def _build_t_term_python_group(t_list, h):
    """ x """

    sum_list, unlinked_list = [], []

    offset_dict = {'summation_upper': 0, 'summation_lower': 0, 'unlinked': 0}

    # here we have to account for the indices already use in h
    offset_dict['summation_upper'] += (h.m - h.m_o)
    offset_dict['unlinked'] += h.m_o + h.n_o

    log.info(offset_dict)

    for t in t_list:
        sum_label, unlinked_label = _build_t_term_python_labels(t, offset_dict)
        log.debug(t, sum_label, unlinked_label)
        sum_list.append(sum_label)
        unlinked_list.append(unlinked_label)

    return sum_list, unlinked_list


def _full_cc_einsum_vibrational_components(h, t_list):
    """ x """
    vibrational_components = []  # store return values here

    old_print_wrapper(h, t_list)

    h_labels = _build_h_term_python_labels(h)

    alist, blist = _build_t_term_python_group(t_list, h)

    vibrational_components.append(h_labels[0] + h_labels[1])
    for i in range(len(alist)):
        vibrational_components.append(alist[i] + blist[i])

    remaining_list = [h_labels[1], ]
    for i in range(len(alist)):
        remaining_list.append(blist[i])

    return vibrational_components, ''.join(remaining_list)


def _full_cc_einsum_subscript_generator(h, t_list):
    """ x """
    return_string = ""

    electronic_components = _full_cc_einsum_electronic_components(t_list)

    vibrational_components, remaining_indices = _full_cc_einsum_vibrational_components(h, t_list)

    summation_subscripts = ", ".join([
        f"{electronic_components[i]}{vibrational_components[i]}" for i in range(len(electronic_components))
    ])

    return_string = f"{summation_subscripts} -> ab{remaining_indices}"

    return return_string


def _full_cc_einsum_prefactor(term):
    """ x """
    string = ""

    return string


def _simplify_full_cc_python_prefactor(numerator_list, denominator_list):
    """ x """

    # if one of the lists is empty there is no easy simplification
    if numerator_list == [] or denominator_list == []:
        return numerator_list, denominator_list

    numerator_set = set(numerator_list)
    denominator_set = set(denominator_list)

    # if the numerator and denominator do not share the same factors
    # there is no easy simplification
    if numerator_set.isdisjoint(denominator_set):
        return numerator_list, denominator_list
    else:
        intersection = numerator_set & denominator_set

    numerator_dict = dict([(key, 0) for key in numerator_set])
    for string in numerator_list:
        numerator_dict[string] += 1

    old_print_wrapper('nnnn', numerator_dict)

    denominator_dict = dict([(key, 0) for key in denominator_set])
    for string in denominator_list:
        denominator_dict[string] += 1

    old_print_wrapper('dddd', denominator_dict)

    for key in intersection:
        a, b = numerator_dict[key], denominator_dict[key]
        if a > b:
            denominator_dict[key] = 0
            numerator_dict[key] = a - b
        elif a < b:
            denominator_dict[key] = b - a
            numerator_dict[key] = 0
        elif a == b:
            denominator_dict[key] = 0
            numerator_dict[key] = 0

    # make updated lists
    numerator_list, denominator_list = [], []

    for k, v in numerator_dict.items():
        numerator_list.extend([k, ]*v)
    for k, v in denominator_dict.items():
        denominator_list.extend([k, ]*v)

    if len(numerator_list) > 2 or len(denominator_list) > 2:
        old_print_wrapper('xxxx', numerator_list)
        old_print_wrapper('yyyy', denominator_list)

    return numerator_list, denominator_list


def _build_full_cc_python_prefactor(h, t_list, simplify_flag=True):
    """Attempt to return python code representing appropriate prefactor term.

    All prefactors begin with 1/n! where n is the number of t amplitudes in given term.
    This comes from the prefactors of the taylor expansion of e^S = 1 + S^1 + S^2 ....
    We also need to account for the prefactors of the Hamiltonian as defined in the theory.
    There are also permutations of indices to take into account.
    Overall the entire affair is quite complicated.

    In general the rules are:
        1 - 1/n! where n is the number of t terms (taylor series)
        1 - All t terms are balanced/neutral (they contribute m!n!/m!n! = 1)
        1 - h contributes x!/m!n! where x is the number of UNIQUE t terms that h contracts with

    A single h with no t list is a special case where the prefactor is always 1.
    """

    # special case, single h
    if len(t_list) == 1 and t_list[0] == disconnected_namedtuple(0, 0, 0, 0):
        return ''

    numerator_list = []
    denominator_list = []

    # account for prefactor from Hamiltonian term `h`

    # connections = np.count_nonzero([h.m_t[i]+h.n_t[i] for i in range(len(h.m_t))])
    connected_ts = [t for t in t_list if t.m_h > 0 or t.n_h > 0]

    x = len(set(connected_ts))

    debug_flag = (
        h.n == 2
        and h.m == 0
        and len(t_list) == 2
        and t_list[0].m_h == 1
        and t_list[0].m_o == 1
        and t_list[1].m_h == 1
        and t_list[1].m_o == 1
    )

    if debug_flag:
        old_print_wrapper('\n\n\nzzzzzzzzz')
        old_print_wrapper(connected_ts)
        old_print_wrapper(set(connected_ts))
        old_print_wrapper(len(set(connected_ts)))
        old_print_wrapper('zzzzzzzzz\n\n\n')

    if x > 1:
        numerator_list.append(f'factorial({x})')
    if h.m > 1:
        denominator_list.append(f'factorial({h.m})')
    if h.n > 1:
        denominator_list.append(f'factorial({h.n})')

    # account for the number of permutations of all t-amplitudes
    if len(t_list) > 1:
        denominator_list.append(f'factorial({len(t_list)})')

    # simplify
    if simplify_flag:
        numerator_list, denominator_list = _simplify_full_cc_python_prefactor(numerator_list, denominator_list)

    # glue the numerator and denominator together
    numerator = '1' if (numerator_list == []) else f"({' * '.join(numerator_list)})"
    denominator = '1' if (denominator_list == []) else f"({' * '.join(denominator_list)})"

    if numerator == '1' and denominator == '1':
        return ''
    else:
        return f"{numerator}/{denominator} * "


def _multiple_perms_logic(term):
    """ x """
    omega, h, t_list = term

    unique_dict = {}
    for t in t_list:
        if t in unique_dict:
            unique_dict[t] += 1
        else:
            unique_dict[t] = 1

    # if there are no permutations to do
    if len(unique_dict) == 1 and next(iter(unique_dict.values())) == 1:
        return None, unique_dict

    # only permutations on one items
    if len(unique_dict) == 1:
        length = list(unique_dict.values())[0]
        return unique_permutations(range(length)), unique_dict

    # if permutations on multiple t's
    if len(unique_dict) > 1:
        return unique_permutations(range(len(t_list))), unique_dict
    #     # lst = []
    #     # for v in unique_dict.values():
    #     #     lst.append(*unique_permutations(range(v)))
    #     old_print_wrapper([unique_permutations(range(v)) for k, v in unique_dict.items()])
    #     return dict([(k, list(*unique_permutations(range(v)))) for k, v in unique_dict.items()]), unique_dict

    raise Exception("Shouldn't get here")


def _write_cc_einsum_python_from_list(rank, truncations, t_term_list, trunc_obj_name='truncation'):
    """ x """

    maximum_h_rank, maximum_cc_rank, s_taylor_max_order, omega_max_order = truncations

    if t_term_list == []:
        return ["pass  # no valid terms here", ]

    return_list = []

    hamiltonian_rank_list = []
    for i in range(maximum_h_rank+1):
        hamiltonian_rank_list.append(dict([(i, {}) for i in range(maximum_cc_rank+1)]))

    for term in t_term_list:

        omega, h, t_list = term

        h_operand = f"h_args[({h.m}, {h.n})]"

        if (len(t_list) == 1) and t_list[0] == disconnected_namedtuple(0, 0, 0, 0):
            return_list.append(f"R += {h_operand}")
            continue

        # logic about multiple permutations
        # generate lists of unique t terms
        permutations, unique_dict = _multiple_perms_logic(term)
        prefactor = _build_full_cc_python_prefactor(h, t_list)
        max_t_rank = max(_rank_of_t_term_namedtuple(t) for t in t_list)
        old_print_wrapper(omega, h, t_list, permutations, sep='\n')
        # if omega.rank == 1 and permutations != None:
        #     sys.exit(0)

        # we still need to account for output/omega permutations

        # -----------------------------------------------------------------------------------------
        # build with permutations
        hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor] = []

        if permutations is None:
            t_operands = ', '.join([f"t_args[({t.m_h + t.m_o}, {t.n_h + t.n_o})]" for t in t_list])

            e_a = _full_cc_einsum_electronic_components(t_list)
            v_a, remaining_indices = _full_cc_einsum_vibrational_components(h, t_list)

            # string = f"np.einsum('{summation_subscripts}', {h_operand}, {t_operands})"
            if remaining_indices == '':
                string = ", ".join([f"{e_a[i]}{v_a[i]}" for i in range(len(e_a))])
                string = f"np.einsum('{string} -> ab{remaining_indices}', {h_operand}, {t_operands})"
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

            elif len(remaining_indices) >= 1:
                for perm in unique_permutations(remaining_indices):
                    string = ", ".join([f"{e_a[i]}{v_a[i]}" for i in range(len(e_a))])
                    string = f"np.einsum('{string} -> ab{remaining_indices}', {h_operand}, {t_operands})"
                    old_print_wrapper(perm)
                    old_print_wrapper(remaining_indices)
                    hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

                if len(remaining_indices) >= 2:
                    # sys.exit(0)
                    pass

        elif len(unique_dict.keys()) == 1:

            t_operands = ', '.join([f"t_args[({t.m_h + t.m_o}, {t.n_h + t.n_o})]" for t in t_list])

            e_a = _full_cc_einsum_electronic_components(t_list)
            v_a, remaining_indices = _full_cc_einsum_vibrational_components(h, t_list)

            for perm in permutations:
                string = ", ".join([f"{e_a[0]}{v_a[0]}"] + [f"{e_a[i+1]}{v_a[p+1]}" for i, p in enumerate(perm)])
                string = f"np.einsum('{string} -> ab{remaining_indices}', {h_operand}, {t_operands})"
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

        elif len(unique_dict) > 1:

            e_a = _full_cc_einsum_electronic_components(t_list)
            v_a, remaining_indices = _full_cc_einsum_vibrational_components(h, t_list)

            for perm in permutations:
                t_operands = ', '.join([
                    f"t_args[({t_list[i].m_h + t_list[i].m_o}, {t_list[i].n_h + t_list[i].n_o})]"
                    for i in perm
                ])
                string = ", ".join([f"{e_a[0]}{v_a[0]}"] + [f"{e_a[i+1]}{v_a[p+1]}" for i, p in enumerate(perm)])
                string = f"np.einsum('{string} -> ab{remaining_indices}', {h_operand}, {t_operands})"
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

        else:
            raise Exception('')

    # -----------------------------------------------------------------------------------------
    # remove any duplicates
    # for h_rank_list in hamiltonian_rank_list:
    #     for t_rank_list in h_rank_list:
    #         for prefactor_list in t_rank_list:

    # -----------------------------------------------------------------------------------------

    def _handle_multiline_same_prefactor(return_list, prefactor, string_list, nof_tabs=0):
        tabber = tab*nof_tabs

        if len(string_list) > 1:
            return_list.append(f"{tabber}R += {prefactor}(")
            for string in string_list:
                return_list.append(f"{tabber}{tab}{string} +")
            # remove the last plus symbol
            return_list[-1] = return_list[-1][:-2]
            return_list.append(f"{tabber})")
        else:
            return_list.append(f"{tabber}R += {prefactor}{string_list[0]}")
        return

    # order the terms as we return them
    for prefactor, string_list in hamiltonian_rank_list[0][0].items():
        _handle_multiline_same_prefactor(return_list, prefactor, string_list, nof_tabs=0)

    for j in range(1, maximum_cc_rank+1):
        if hamiltonian_rank_list[0][j] != {}:
            return_list.append(f"if {trunc_obj_name}.{taylor_series_order_tag[j]}:")
            for prefactor, string_list in hamiltonian_rank_list[0][j].items():
                _handle_multiline_same_prefactor(return_list, prefactor, string_list, nof_tabs=1)

    for i in range(1, maximum_h_rank+1):
        return_list.append('')
        return_list.append(f"if {trunc_obj_name}.at_least_{hamiltonian_order_tag[i]}:")
        for prefactor, string_list in hamiltonian_rank_list[i][0].items():
            _handle_multiline_same_prefactor(return_list, prefactor, string_list, nof_tabs=1)

        for j in range(1, maximum_cc_rank+1):
            if hamiltonian_rank_list[i][j] != {}:
                return_list.append(f"{tab}if {trunc_obj_name}.{taylor_series_order_tag[j]}:")
                for prefactor, string_list in hamiltonian_rank_list[i][j].items():
                    _handle_multiline_same_prefactor(return_list, prefactor, string_list, nof_tabs=2)

    # old_print_wrapper(return_list)
    # sys.exit(0)

    return return_list


def _generate_full_cc_einsums(omega_term, truncations, only_ground_state=False, opt_einsum=False):
    """Return a string containing python code to be placed into a .py file.
    This does all the work of generating the einsums.
    """
    maximum_h_rank, maximum_cc_rank, s_taylor_max_order, omega_max_order = truncations

    H = generate_full_cc_hamiltonian_operator(maximum_h_rank)
    s_taylor_expansion = generate_s_taylor_expansion(maximum_cc_rank, s_taylor_max_order, only_ground_state)

    simple_repr_list = []  # old list, not so important anymore, might remove in future
    valid_term_list = []   # store all valid Omega * h * (s*s*...s) terms here

    """ First we want to generate a list of valid terms.
    We start with the list of lists `s_taylor_expansion` which is processed by `_filter_out_valid_s_terms`.
    This function identifies valid pairings AND places those pairings in the `valid_term_list`.
    Specifically we replace the `general_operator_namedtuple`s with `connected_namedtuple`s and/or
    `disconnected_namedtuple`s.
    """
    for count, s_series_term in enumerate(s_taylor_expansion):
        log.debug(s_series_term, "-"*100, "\n\n")
        _filter_out_valid_s_terms(omega_term, H, s_series_term, simple_repr_list, valid_term_list, remove_f_terms=True)

    # take all terms and separate them into their respective groups
    fully, linked, unlinked = _seperate_s_terms_by_connection(valid_term_list)

    if False:
        _debug_print_valid_term_list(valid_term_list)
        _debug_print_different_types_of_terms(fully, linked, unlinked)

    return_list = [
        _write_cc_einsum_python_from_list(omega_term.rank, truncations, fully),
        _write_cc_einsum_python_from_list(omega_term.rank, truncations, linked),
        _write_cc_einsum_python_from_list(omega_term.rank, truncations, unlinked),
    ]

    return return_list


def _generate_full_cc_compute_function(omega_term, truncations, only_ground_state=False, opt_einsum=False):
    """ x """
    maximum_h_rank, maximum_cc_rank, s_taylor_max_order, omega_max_order = truncations

    return_string = ""
    specifier_string = f"m{omega_term.m}_n{omega_term.n}"

    if not opt_einsum:

        # generate ground state einsums
        ground_state_only_einsums = _generate_full_cc_einsums(omega_term, truncations, only_ground_state=True)

        # generate einsums if not ground state
        if not only_ground_state:
            einsums = _generate_full_cc_einsums(omega_term, truncations)
        else:
            einsums = [("raise Exception('Hot Band amplitudes not implemented!')", ), ]*3
            # old_print_wrapper(einsums)
            # sys.exit(0)

        six_tab = "\n" + tab*6

        for i, term_type in enumerate(['fully_connected', 'linked_disconnected', 'unlinked_disconnected']):
            # write function definition
            function_string = f'''
                def add_{specifier_string}_{term_type}_terms(R, ansatz, truncation, h_args, t_args):
                    """Compute the {omega_term} {term_type} terms."""
                    if ansatz.ground_state:
                        {six_tab.join(ground_state_only_einsums[i])}
                    else:
                        {six_tab.join(einsums[i])}
                    return
            '''
            # remove space fr

            # remove 4 consecutive tabs from the multi-line string `function_string`
            function_string = "\n".join([line[tab_length*4:].rstrip() for line in function_string.splitlines()])
            # two lines between each function
            return_string += function_string + '\n'

    # optimized term calculations
    else:

        # generate ground state einsums
        ground_state_only_einsums = _generate_full_cc_einsums(omega_term, truncations, only_ground_state=True, opt_einsum=True)

        # generate einsums if not ground state
        if not only_ground_state:
            einsums = _generate_full_cc_einsums(omega_term, truncations, opt_einsum=True)
        else:
            einsums = [("raise Exception('Hot Band amplitudes not implemented!')", ), ]*3

        six_tab = "\n" + tab*6

        for i, term_type in enumerate(['fully_connected', 'linked_disconnected', 'unlinked_disconnected']):
            # write function definition
            function_string = f'''
                def add_{specifier_string}_{term_type}_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
                    """Optimized calculation of the of the {omega_term} {term_type} terms."""

                    if ansatz.ground_state:
                        {six_tab.join(ground_state_only_einsums[i])}
                    else:
                        {six_tab.join(einsums[i])}
                    return
            '''

            # remove 4 consecutive tabs from the multi-line string `function_string`
            function_string = "\n".join([line[tab_length*4:] for line in function_string.splitlines()])
            # two lines between each function
            return_string += function_string + '\n'

    return return_string


def _write_master_full_cc_compute_function(omega_term, opt_einsum=False):
    """Write the wrapper function which `vibronic_hamiltonian.py` calls.
    """

    specifier_string = f"m{omega_term.m}_n{omega_term.n}"

    if not opt_einsum:
        func_string = f'''
            def compute_{specifier_string}_amplitude(A, N, ansatz, truncation, h_args, t_args):
                """Compute the {omega_term} amplitude."""
                truncation.confirm_at_least_singles()

                # the residual tensor
                R = np.zeros(shape=({', '.join(['A','A',] + ['N',]*omega_term.rank)}), dtype=complex)

                # add each of the terms
                add_{specifier_string}_fully_connected_terms(R, ansatz, truncation, h_args, t_args)
                add_{specifier_string}_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
                add_{specifier_string}_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
                return R

        '''
    else:
        func_string = f'''
            def compute_{specifier_string}_amplitude_optimized(A, N, ansatz, truncation, h_args, t_args, opt_paths):
                """Compute the {omega_term} amplitude."""
                truncation.confirm_at_least_singles()

                # the residual tensor
                R = np.zeros(shape=({', '.join(['A','A',] + ['N',]*omega_term.rank)}), dtype=complex)

                # unpack the optimized paths
                optimized_connected_paths, optimized_linked_paths, optimized_unlinked_paths = opt_paths

                # add each of the terms
                add_{specifier_string}_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, optimized_connected_paths)
                add_{specifier_string}_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, optimized_linked_paths)
                add_{specifier_string}_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, optimized_unlinked_paths)
                return R

        '''

    # remove three indents from the multi-line string `func_string`
    lines = func_string.splitlines()
    # 3 consecutive tabs that we are ignoring/skipping
    trimmed_string = "\n".join([line[tab_length*3:] for line in lines])

    return trimmed_string


def _wrap_full_cc_generation(master_omega, s2, named_line, spaced_named_line, only_ground_state=False, opt_einsum=False):
    """ x """
    return_string = ""

    for i, omega_term in enumerate(master_omega.operator_list):

        # only print the header when we change rank (from linear to quadratic for example)
        if omega_term.rank > master_omega.operator_list[i-1].rank:
            return_string += spaced_named_line(f"RANK {omega_term.rank:2d} FUNCTIONS", s2) + '\n'

        # header
        return_string += '\n' + named_line(f"{omega_term} TERMS", s2//2)
        # functions
        return_string += _generate_full_cc_compute_function(omega_term, truncations, only_ground_state, opt_einsum)

    return return_string


def _generate_full_cc_python_file_contents(truncations, only_ground_state=False):
    """Return a string containing the python code to generate w operators up to (and including) `max_order`.
    Requires the following header: `"import numpy as np\nfrom math import factorial"`.
    """
    assert len(truncations) == 4, "truncations argument needs to be tuple of four integers!!"
    maximum_h_rank, maximum_cc_rank, s_taylor_max_order, omega_max_order = truncations
    master_omega = generate_omega_operator(maximum_cc_rank, omega_max_order)

    # ------------------------------------------------------------------------------------------- #
    # for generating labels and spacing
    def named_line(name, width):
        return "# " + "-"*width + f" {name} " + "-"*width + " #"

    s1, s2 = 75, 28
    spacing_line = "# " + "-"*s1 + " #\n"

    def spaced_named_line(name, width):
        return spacing_line + named_line(name, width) + '\n' + spacing_line

    s3, s4 = 109, 45
    large_spacing_line = "# " + "-"*s3 + " #\n"

    def long_spaced_named_line(name, width):
        return large_spacing_line + named_line(name, width) + '\n' + large_spacing_line
    # ------------------------------------------------------------------------------------------- #
    #
    # ------------------------------------------------------------------------------------------- #
    # header for default functions (as opposed to the optimized functions)
    string = long_spaced_named_line("DEFAULT FUNCTIONS", s4)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("INDIVIDUAL TERMS", s4) + '\n\n'
    # generate
    string += _wrap_full_cc_generation(master_omega, s2, named_line, spaced_named_line, only_ground_state)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("RESIDUAL FUNCTIONS", s4)
    # generate
    string += "".join([
        _write_master_full_cc_compute_function(omega_term)
        for omega_term in master_omega.operator_list
    ])
    # ------------------------------------------------------------------------------------------- #
    #
    # ------------------------------------------------------------------------------------------- #
    # header for optimized functions
    string += long_spaced_named_line("OPTIMIZED FUNCTIONS", s4-1)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("INDIVIDUAL TERMS", s4) + '\n\n'
    # generate
    string += _wrap_full_cc_generation(master_omega, s2, named_line, spaced_named_line, only_ground_state, opt_einsum=True)
    # ----------------------------------------------------------------------- #
    # generate
    string += '\n' + named_line("RESIDUAL FUNCTIONS", s4)
    string += "".join([
        _write_master_full_cc_compute_function(omega_term, opt_einsum=True)
        for omega_term in master_omega.operator_list
    ])
    # ------------------------------------------------------------------------------------------- #
    #
    # ------------------------------------------------------------------------------------------- #
    # header for optimized paths function
    string += '\n' + named_line("OPTIMIZED PATHS FUNCTION", s4)
    # write the code for generating optimized paths for full CC, this is probably different than the W code?!?
    # maybe... im not sure?
    # both VEMX and VECC
    # ------------------------------------------------------------------------------------------- #
    return string


def generate_full_cc_python(truncations, only_ground_state=False, path="./full_cc_equations.py"):
    """Generates and saves to a file the code to calculate the terms for the full CC approach."""

    # start with the import statements
    file_data = (
        "# system imports\n"
        "from math import factorial\n"
        "\n"
        "# third party imports\n"
        "import numpy as np\n"
        "import opt_einsum as oe\n"
        "\n"
        "# local imports\n"
        "from .symmetrize import symmetrize_tensor\n"
        "from ..log_conf import log\n"
        "\n"
    )

    # write the functions to calculate the W operators
    file_data += _generate_full_cc_python_file_contents(truncations, only_ground_state)

    # save data
    with open(path, 'w') as fp:
        fp.write(file_data)

    return


# ----------------------------------------------------------------------------------------------- #
# -----------------------------  GENERATING RESIDUAL EQUATIONS  --------------------------------- #
# ----------------------------------------------------------------------------------------------- #
h_dict = {
    (0, 0): 'h_ab',
    (0, 1): 'h_abI',
    (1, 0): 'h_abi',
    (1, 1): 'h_abIj',
    (0, 2): 'h_abIJ',
    (2, 0): 'h_abij',
}

w_dict = {
    1: 'w_i',
    2: 'w_ij',
    3: 'w_ijk',
    4: 'w_ijkl',
    5: 'w_ijklm',
    6: 'w_ijklmn',
    7: 'w_ijklmno',
    8: 'w_ijklmnop',
}


i_list = ['i', 'j', 'k', 'l']
k_list = ['m', 'n']


def _generate_einsum_h_indices(term):
    """Generate the indices for the h term in the residual's einsum equation
    upper or lower k's should map to the letters 'm' and 'n'
    upper or lower i's should map to the letters 'ijkl'
    """
    h_dims = 'ac'
    h_dims += ''.join(k_list[0:term.h.max_k])  # generate mn's
    h_dims += ''.join(i_list[0:term.h.max_i])  # generate jikl's
    return h_dims


def _generate_einsum_w_indices(term):
    """Generate the indices for the W operator in the residual's einsum equation
    upper or lower k's should map to the letters 'm' and 'n'
    upper or lower i's should map to the letters 'ijkl'
    """
    w_dims = 'cb'
    w_dims += ''.join(k_list[0:term.w.max_k])  # generate mn's
    w_dims += ''.join(i_list[term.h.max_i:(term.h.max_i + term.w.max_i)])  # generate jikl's

    return w_dims


def _generate_einsum_ouput_indices(term):
    """Generate the labels for the residual output."""
    # number of normal mode labels in output should be equal to the sum of the `max_i`'s
    return 'ab' + ''.join(i_list[0:(term.h.max_i + term.w.max_i)])


def _residual_terms_einsum(term, suppress_1_prefactor=True):
    """Returns a python code in `str` format which calculates the contribution to the residual from `term`."""

    # create the prefactor
    if suppress_1_prefactor and (term.prefactor == 1.0):  # if we don't want to print prefactors that are 1
        prefactor = ""
    else:
        prefactor = str(term.prefactor) + ' * '

    # create the string naming the h python object
    h = h_dict[(term.h.max_i, term.h.max_k)]

    # if there is no W operator then we simply return the h tensor multiplied by the prefactor
    if term.w.order == 0:
        return f"R += {prefactor}{h}\n"

    # create the string naming the w python object
    w = w_dict[term.w.order]

    # generate command
    h_dims = _generate_einsum_h_indices(term)
    w_dims = _generate_einsum_w_indices(term)
    out_dims = _generate_einsum_ouput_indices(term)

    return f"R += {prefactor}np.einsum('{h_dims},{w_dims}->{out_dims}', {h}, {w})\n"


def _same_w_order_term_list(current_term, term_list):
    """Returns a list of terms in the list whose W operator order is the same as the `current_term`'s"""
    return [term for term in term_list if term.w.order == current_term.w.order]


def write_residual_function_string(residual_terms_list, order):
    """ Output a string of python code to calculate a specific residual """

    string = ""  # we store the output in this string

    # this line of python code labels the terms in the `w_args` tuple
    w_args_string = ", ".join([w_dict[n] for n in range(1, order+3)] + ["*unusedargs"])
    # old_print_wrapper(f"{w_args_string=}")

    # the function definition and initialization of the residual array
    string += (
        f"\ndef calculate_order_{order}_residual(A, N, truncation, h_args, w_args):\n"
        f'{tab}"""Calculate the {order} order residual as a function of the W operators."""\n'
        f"{tab}h_ab, h_abI, h_abi, h_abIj, h_abIJ, h_abij = h_args\n"
        f"{tab}{w_args_string} = w_args\n"
        "\n"
        f"{tab}R = np.zeros(({', '.join(['A','A',] + ['N',]*order)}), dtype=complex)\n"
    )

    # add code to assert correct truncation
    if order <= 1:
        assertion_check = f"truncation.{taylor_series_order_tag[1]}"
    else:
        assertion_check = f"truncation.{taylor_series_order_tag[order]}"

    string += (
        "\n"
        f'{tab}assert {assertion_check}, \\\n'
        f'{tab}{tab}f"Cannot calculate order {order} residual for {{truncation.cc_truncation_order}}"\n'
    )

    # a list of all terms whose h operator is quadratic (2nd or higher order)
    quadratic_terms = [term for term in residual_terms_list if (term.h.max_i >= 2) or (term.h.max_k >= 2)]

    # each time we write a term to the `string` object append it to this list
    # so we don't write any terms twice
    already_printed_list = []

    for term in residual_terms_list:

        if term in already_printed_list:
            continue

        string += '\n'  # space out the einsum commands

        if term.w.order == 0 or term.w.order == 1:
            already_printed_list.append(term)
            string += f"{tab}" + _residual_terms_einsum(term)

        # if the h operator is 2nd or higher order
        elif (term.h.max_i >= 2) or (term.h.max_k >= 2):

            string += f"{tab}if truncation.quadratic:\n"

            # find all the other terms whose h operator is quadratic (2nd or higher order)
            for quad_term in quadratic_terms:
                already_printed_list.append(quad_term)
                # if the next quadratic term has a W operator which is 2nd or higher order
                if bool(quad_term.w.order >= 2):
                    string += (
                        f"{tab}{tab}if {w_dict[quad_term.w.order]} is not None:\n"
                        f"{tab}{tab}{tab}" + _residual_terms_einsum(quad_term)
                    )
                else:
                    string += (
                        f"{tab}{tab}else:\n"
                        f"{tab}{tab}{tab}" + _residual_terms_einsum(quad_term)
                    )

        # if the W operator is 2nd or higher order
        elif bool(term.w.order >= 2):
            string += f"{tab}if {w_dict[term.w.order]} is not None:\n"

            # find all the other terms whose W operator is the same order as `term`
            for same_w_term in _same_w_order_term_list(term, residual_terms_list):
                already_printed_list.append(same_w_term)
                string += f"{tab}{tab}" + _residual_terms_einsum(same_w_term)

        else:
            raise Exception("We shouldn't reach this else!")

    string += (f"\n{tab}return R")
    return string


def generate_python_code_for_residual_functions(term_lists, max_order):
    """Return a string containing the python code to generate residual functions up to (and including) `max_order`.
    `term_lists` is a list of lists, each of which contain tuples `(prefactor, h, w)`,
        representing terms of that particular residual.
    Requires the following header: `"import numpy as np"`.
    """
    lst = [write_residual_function_string(term_lists[order], order=order) for order in range(max_order+1)]
    return "\n\n".join(lst)


def generate_residual_equations_file(max_residual_order, maximum_h_rank, path="./residual_equations.py"):
    """Generates and saves to a file the code to calculate the residual equations for the CC approach."""

    # generate the Hamiltonian data
    H = generate_hamiltonian_operator(maximum_h_rank)
    for h in H.operator_list:
        old_print_wrapper(h)

    # generate the residual data
    R_lists, term_lists = generate_residual_data(H.operator_list, max_order=max_residual_order)

    # if we want to print stuff for debugging
    print_residual_data(R_lists, term_lists, print_equations=False, print_tuples=False)

    # start with the import statements
    file_data = (
        "# system imports\n"
        "\n"
        "# third party imports\n"
        "import numpy as np\n"
        "\n"
        "# local imports\n"
        "from .symmetrize import symmetrize_tensor\n"
        "\n"
    )

    # write the functions to calculate the residuals
    file_data += generate_python_code_for_residual_functions(term_lists, max_order=max_residual_order)
    file_data += '\n'  # EOF at end of file

    # save data
    with open(path, 'w') as fp:
        fp.write(file_data)

    return


# ----------------------------------------------------------------------------------------------- #
# ---------------------------  GENERATING W OPERATOR EQUATIONS  --------------------------------- #
# ----------------------------------------------------------------------------------------------- #
def _next_list(lst, n):
    ''' if items in `lst` are all one, or there is only one 1 in `lst`:
            -add 1 to the last item and delete the first 1 in the list like:
                [1, 1, 1] -> [1, 2] and [1, 2] -> [3];
        if there is no 1 in `lst`:
            -return the list itself;
        if there more than one 1 in `lst`:
            -delete one of 1 and add 1 to the last and the one before the last item separately, like:
                [1, 1, 2] -> [(1,3), (2, 2)]'''
    if lst.count(1) == 0:
        return [lst, ]
    elif lst.count(1) == n or lst.count(1) == 1:
        result = lst[1:-1] + [(lst[-1]+1), ]
        return [result, ]
    else:
        result = [(lst[1:-2] + [lst[-2]+1, ] + [lst[-1], ]), (lst[1:-1] + [(lst[-1]+1), ])]
        return result


def _generate_t_lists(n):
    ''' generates a list of lists in which each item is from 1 to n and the sum of items are n, for example:
        4 -> [[4], [1, 3], [2, 2], [1, 1, 2], [1, 1, 1, 1]]'''
    first = [1]*n
    result = []
    lst = [first]
    current = _next_list(lst[-1], n)

    # if there is no 1 in current(a list), lst gets all items, bit not in correct order
    while current != [lst[-1]]:
        lst += current
        current = _next_list(lst[-1], n)

    # sort lst to get the final result
    previous_max = max(lst[0])  # initialize previous_max to record the maximum value in the previous list

    for sub_lst in lst:
        current_max = max(sub_lst)  # record the maximum in the first list in lst

        # if the maximum item in sub-lst is larger than the max in previous sub-list
        # add sub_lst to the end of result list
        if current_max >= previous_max:
            result.append(sub_lst)
            previous_max = current_max
        # if the max in sub-lst is smaller than the max in previous sub-list
        # add sub_lst to the beginning of result list
        else:
            result = [sub_lst] + result
    result.reverse()
    return result


def _generate_t_terms_dictionary(n):
    ''' generates a dictionary in which keys are tuple that show the tag for t terms as number, and values are the actual t terms,
        for example: {(1, 4): ('t_i', 't_ijkl')}'''
    result = {}
    t_tag_list = ["i", "j", "k", "l", "m", "n"]
    t_num_list = _generate_t_lists(n)  # a list of lists like [1,1], [2,1,1]....
    for num_term in t_num_list:
        t_term = []  # stores t_i, t_ij, t_ijk....
        for n in num_term:
            t_str = "t_" + "".join(t_tag_list[:n])  # add i,j,k.....to "t_"
            # old_print_wrapper(n, t_str, t_terms[n-1].string)
            t_term.append(t_str)
        result[tuple(num_term)] = tuple(t_term)
    return result


def _permutation_function(l_t, l_fix, l_perm, n):
    ''' generates a list of character combinations which will be used in the einsum calculation,
        l_t contains combination of t terms group like [["t_i", "t_ij"], ["t_ij", "t_i"]];
        l_fix contains the fixed part of character combinations which will be used in the einsum calculation like
            ["ac", "cd", "db"] for (t_i, t_i, t_ij) group;
        l_perm contains groups of characters which will be added to the fixed part later, for example:
            [[i, j], [j, i]] for [t_i, t_i] and ["ac","cb"]
        '''

    result = []
    for char_group in l_perm:
        s_result = []
        for t_group in l_t:
            ss_result = [list(l_fix)]
            i = 0
            p_sum = 0
            for t_item in t_group:
                length = len(t_item) - 2  # delete the length of "t_" part
                ss_result[0][i] += "".join(char_group[p_sum:length+p_sum])
                p_sum += length
                i += 1
            if len(l_fix) != n:
                s_result += ss_result
        if len(l_fix) == n:
            s_result += ss_result
        result += [s_result]
    return result


def _write_permutations(perm_t, perm_char_list, W_array, prefactor):
    ''' generates lines for w function if permutations inside the einsum calculation is needed'''
    result = ""
    for p_group in perm_char_list:
        # list of command strings for einsum
        # cmd = ",".join(einsum_perm) + " -> " + einsum_result
        cmd_list = [", ".join(p_c) for p_c in p_group]
        # list of arguments to einsum call
        arg_list = [", ".join(perm) for perm in perm_t]
        # compile a list of the einsum calls (as strings)
        einsum_list = [f"np.einsum('{cmd_list[i]}', {arg_list[i]})" for i in range(len(p_group))]
        # join them together
        sum_of_einsums = " + ".join(einsum_list)
        # add the whole line to the main string
        result += f"{tab}{W_array} += {prefactor} * ({sum_of_einsums})\n"
    return result


# ------------------------------------------------------- #
t_terms = [
    None,
    t_term_namedtuple("t_i", 1, "(A, A, N)"),
    t_term_namedtuple("t_ij", 2, "(A, A, N, N)"),
    t_term_namedtuple("t_ijk", 3, "(A, A, N, N, N)"),
    t_term_namedtuple("t_ijkl", 4, "(A, A, N, N, N, N)"),
    t_term_namedtuple("t_ijklm", 5, "(A, A, N, N, N, N, N)"),
    t_term_namedtuple("t_ijklmn", 6, "(A, A, N, N, N, N, N, N)"),
    t_term_namedtuple("t_ijklmno", 7, "(A, A, N, N, N, N, N, N, N)"),
    t_term_namedtuple("t_ijklmnop", 8, "(A, A, N, N, N, N, N, N, N, N)"),
]


def _generate_w_operator_prefactor(tupl):
    """Generates the prefactor for each part of W terms.
    The theory from which the prefactors arise goes as follows:
        - The Taylor series contributes a `1/factorial(length(x))`.
        - Each integer `n` in the tuple contributes `1/factorial(n)`.
    We choose not to print out the 1/factorial(1) prefactors.
    """

    # if `tupl` is (1,)
    if max(tupl) == 1 and len(tupl) == 1:
        return ""

    # if all items in `tupl` are 1, such as (1, 1, 1, ...)
    elif max(tupl) == 1:
        return f"1/factorial({len(tupl)})"

    # if there is only one item in `tupl` and it's not 1
    elif len(tupl) == 1:
        return f"1/factorial({max(tupl)})"

    # otherwise `tupl` has 2 or more terms that are not all 1
    # we need to include a factorial of the length
    # and a factorial for each term that is greater than 1
    else:
        denominator_list = [f"factorial({len(tupl)})", ]
        for n in tupl:
            if n != 1:
                denominator_list.append(f"factorial({n})")
        denominator = " * ".join(denominator_list)
        return f"1/({denominator})"


num_tag = ["zero", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]
taylor_series_order_tag = ["", "singles", "doubles", "triples", "quadruples", "quintuples", "sextuples"]
hamiltonian_order_tag = ["", "linear", "quadratic", "cubic", "quartic", "quintic"]
einsum_surface_tags = "acdefghi"
tag_str = "ijklmnop"


# ------------------------------------------------------- #
def _generate_surface_index(partition):
    """Return a list of strings of length 2 [`ac`, `cd`, `db`].
    Where the first char in the list should be `a` and the last char should be `b`
    and the rest of the letters in between are in alphabetical order.
    """
    max_i = len(partition)-1
    assert max_i <= 7, "Our current `einsum_surface_tags` can't support 7th order W operators"
    return_list = [einsum_surface_tags[i:i+2] for i in range(max_i)]
    return_list.append(einsum_surface_tags[max_i] + "b")
    return return_list


def _generate_mode_index(partition, order):
    """Return a list of strings  [`ij`, `k`, `l`] representing the mode indices
    for the einsum of the W operator.
    """
    assert order <= 6, "Our current `tag_str` can't support 7th order W operators"
    combinations = unique_permutations(partition)
    # log.debug(f"{combinations=}")

    return_list = []
    for comb in combinations:
        N_list = list(tag_str[:order])
        comb_list = [''.join([N_list.pop(0) for _ in range(n)]) for n in comb]
        return_list.append(comb_list)
    return return_list


def _w_einsum_list(partition, order):
    """Returns a list of strings. Each string represents a call to np.einsum.
    The list is filled relative to the specific `partition` that is being calculated.
    """

    # the unique permutations of the `partition` of the integer `order`
    combinations = unique_permutations(partition)
    # the einsum indices for the surface dimensions
    surf_index = _generate_surface_index(partition)
    # the einsum indices for the normal mode dimensions
    mode_index = _generate_mode_index(partition, order)

    return_list = []

    # `combinations` is a list of tuples such as (2, 2, 1, ) or (5,)
    for i, tupl in enumerate(combinations):
        # the input dimensions are two character strings representing the surface dimensions
        # plus 1 or more characters representing the normal mode dimensions
        in_dims = ", ".join([surf_index[a]+mode_index[i][a] for a in range(len(surf_index))])
        # the output dimension is the same for all einsum calls for a given `partition` argument
        out_dims = f"ab{tag_str[0:order]}"
        # the names of the arguments to the einsum call are stored in the list `t_terms`
        # and the objects are accessed using each integer in the tuple (2, 1) -> ("t_ij", "t_i")
        pterms = ", ".join([t_terms[n].string for n in tupl])
        # old_print_wrapper(f"np.einsum('{in_dims}->{out_dims}', {pterms})")
        return_list.append(f"np.einsum('{in_dims}->{out_dims}', {pterms})")

    return return_list


def _optimized_w_einsum_list(partition, order, iterator_name='optimized_einsum'):
    """Returns a list of strings. Each string represents a call to np.einsum.
    The list is filled relative to the specific `partition` that is being calculated.

    This function and `_construct_w_function_definition` need to agree on the name of the
    iterator containing the optimized paths. At the moment it is named `optimized_einsum`.
    """

    # the unique permutations of the `partition` of the integer `order`
    combinations = unique_permutations(partition)

    return_list = []

    # `combinations` is a list of tuples such as (2, 2, 1, ) or (5,)
    for i, tupl in enumerate(combinations):
        # the names of the arguments to the einsum call are stored in the list `t_terms`
        # and the objects are accessed using each integer in the tuple (2, 1) -> ("t_ij", "t_i")
        pterms = ", ".join([t_terms[n].string for n in tupl])
        return_list.append(f"next({iterator_name})({pterms})")

    return return_list


# ------------------------------------------------------- #
def _construct_vemx_contributions_definition(return_string, order, opt_einsum=False, iterator_name='optimized_einsum'):
    """Return the string containing the python code to prepare the function definition and unpack the t_args.
    This function and `_optimized_w_einsum_list` need to agree on the name of the
    iterator containing the optimized paths. At the moment it is named `optimized_einsum`.
    """

    # this line of python code labels the terms in the `t_args` tuple
    t_arg_string = ", ".join([t_terms[n].string for n in range(1, order)] + ["*unusedargs"])

    W_array = f"W_{order}"  # name of the W operator

    if not opt_einsum:
        return_string += (
            f"\ndef _add_order_{order}_vemx_contributions({W_array}, t_args, truncation):\n"
            f'{tab}"""Calculate the order {order} VECI/CC (mixed) contributions to the W operator\n'
            f'{tab}for use in the calculation of the residuals.\n'
            f'{tab}"""\n'
            f"{tab}# unpack the `t_args`\n"
            f"{tab}{t_arg_string} = t_args\n"
        )
    else:
        return_string += (
            f"\ndef _add_order_{order}_vemx_contributions_optimized({W_array}, t_args, truncation, opt_path_list):\n"
            f'{tab}"""Calculate the order {order} VECI/CC (mixed) contributions to the W operator\n'
            f'{tab}for use in the calculation of the residuals.\n'
            f'{tab}Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'
            f'{tab}"""\n'
            f"{tab}# unpack the `t_args`\n"
            f"{tab}{t_arg_string} = t_args\n"
        )
        if order >= 2:
            return_string += (
                f"{tab}# make an iterable out of the `opt_path_list`\n"
                f"{tab}{iterator_name} = iter(opt_path_list)\n"
            )

    return return_string


def _generate_vemx_contributions(order, opt_einsum=False):
    """ x """
    assert order <= 6, "Can only handle up to 6th order due to `einsum_surface_tags` limit"

    return_string = ""  # we store the output in this string

    # prepare the function definition, unpacking of arguments, and initialization of the W array
    return_string += _construct_vemx_contributions_definition(return_string, order, opt_einsum)

    # special exception for order less than 2
    if order < 2:
        lst = return_string.split('"""')
        exception = (
            "\n"
            f"{tab}raise Exception(\n"
            f'{tab}{tab}"the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"\n'
            f'{tab}{tab}"which requires a W operator of at least 2nd order"\n'
            f"{tab})\n"
        )
        return '\"\"\"'.join([lst[0], "Exists for error checking.", exception])

    W_array = f"W_{order}"  # name of the W operator

    # for each partition (the mathematical term) of the integer `order`
    for partition in generate_linked_disconnected_partitions_of_n(order):

        contribution_name = taylor_series_order_tag[max(partition)].upper()
        contribution_string = ""

        # Label the order of this partition's contribution
        return_string += f"{tab}# {contribution_name} contribution\n"

        # if this partition is contributing doubles or higher then we
        # add the if statement to check for correct truncation level
        # note that we will add `{tab}`s to every line in `contribution_string`
        # before adding it to return_string (to account for indentation of if statement)
        if max(partition) >= 2:
            return_string += f"{tab}if truncation.{contribution_name.lower()}:\n"

        # make the prefactor
        prefactor = _generate_w_operator_prefactor(partition)

        # order 1 case is simple
        if len(partition) == 1:  # no permutation is needed for this term
            old_print_wrapper(partition, partition[0])
            # we have to space the line correct (how many tabs)
            if max(partition) >= 2:
                return_string += f"{tab}{tab}{W_array} += {prefactor} * {t_terms[partition[0]].string}\n"
            else:
                return_string += f"{tab}{W_array} += {prefactor} * {t_terms[partition[0]].string}\n"
            # continue onto the next partition
            continue

        # compile a list of the einsum calls
        einsum_list = _optimized_w_einsum_list(partition, order) if opt_einsum else _w_einsum_list(partition, order)

        if len(partition) == order:
            # join them together
            sum_of_einsums = " + ".join(einsum_list)
            # add the whole line to the main string
            contribution_string += f"{tab}{W_array} += {prefactor} * ({sum_of_einsums})\n"
        else:
            # join them together
            sum_of_einsums = f" +\n{tab}{tab}".join(einsum_list)
            # add the whole line to the main string
            contribution_string += (
                f"{tab}{W_array} += {prefactor} * (\n"
                f"{tab}{tab}{sum_of_einsums}\n"
                f"{tab})\n"
            )

        # we need to add a `{tab}` to every line in the `contribution_string`
        # to space the lines correctly to account for the if statement
        if max(partition) >= 2:
            lines = contribution_string.splitlines()
            contribution_string = "".join([f"{tab}{line}\n" for line in lines])
        # add the contribution to the `return_string`
        return_string += contribution_string
    # end of loop
    return_string += f"{tab}return\n"
    return return_string


# ------------------------------------------------------- #
def _construct_vecc_contributions_definition(return_string, order, opt_einsum=False, iterator_name='optimized_einsum'):
    """Return the string containing the python code to prepare the function definition and unpack the t_args.
    This function and `_optimized_w_einsum_list` need to agree on the name of the
    iterator containing the optimized paths. At the moment it is named `optimized_einsum`.
    """

    # this line of python code labels the terms in the `t_args` tuple
    t_arg_string = ", ".join([t_terms[n].string for n in range(1, order-1)] + ["*unusedargs"])

    W_array = f"W_{order}"  # name of the W operator

    if not opt_einsum:
        return_string += (
            f"\ndef _add_order_{order}_vecc_contributions({W_array}, t_args, truncation):\n"
            f'{tab}"""Calculate the order {order} VECC contributions to the W operator\n'
            f'{tab}for use in the calculation of the residuals.\n'
            f'{tab}"""\n'
            f"{tab}# unpack the `t_args`\n"
            f"{tab}{t_arg_string} = t_args\n"
        )
    else:
        return_string += (
            f"\ndef _add_order_{order}_vecc_contributions_optimized({W_array}, t_args, truncation, opt_path_list):\n"
            f'{tab}"""Calculate the order {order} VECC contributions to the W operator\n'
            f'{tab}"for use in the calculation of the residuals.\n'
            f'{tab}Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'
            f'{tab}"""\n'
            f"{tab}# unpack the `t_args`\n"
            f"{tab}{t_arg_string} = t_args\n"
        )
        if order >= 2:
            return_string += (
                f"{tab}# make an iterable out of the `opt_path_list`\n"
                f"{tab}{iterator_name} = iter(opt_path_list)\n"
            )

    return return_string


def _generate_vecc_contributions(order, opt_einsum=False):
    """ x """
    assert order <= 6, "Can only handle up to 6th order due to `einsum_surface_tags` limit"

    return_string = ""  # we store the output in this string

    # prepare the function definition, unpacking of arguments, and initialization of the W array
    return_string += _construct_vecc_contributions_definition(return_string, order, opt_einsum)

    # special exception for order less than 2
    if order < 4:
        lst = return_string.split('"""')
        exception = (
            "\n"
            f"{tab}raise Exception(\n"
            f'{tab}{tab}"the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n'
            f'{tab}{tab}"which requires a W operator of at least 4th order"\n'
            f"{tab})\n"
        )
        return '\"\"\"'.join([lst[0], "Exists for error checking.", exception])

    W_array = f"W_{order}"  # name of the W operator

    # for each partition (the mathematical term) of the integer `order`
    for partition in generate_un_linked_disconnected_partitions_of_n(order):

        contribution_name = taylor_series_order_tag[max(partition)].upper()
        contribution_string = ""

        # Label the order of this partition's contribution
        return_string += f"{tab}# {contribution_name} contribution\n"

        # if this partition is contributing doubles or higher then we
        # add the if statement to check for correct truncation level
        # note that we will add `{tab}`s to every line in `contribution_string`
        # before adding it to return_string (to account for indentation of if statement)
        if max(partition) >= 2:
            return_string += f"{tab}if truncation.{contribution_name.lower()}:\n"

        # make the prefactor
        prefactor = _generate_w_operator_prefactor(partition)

        # order 1 case is simple
        if len(partition) == 1:  # no permutation is needed for this term
            old_print_wrapper(partition, partition[0])
            # we have to space the line correct (how many tabs)
            if max(partition) >= 2:
                return_string += f"{tab}{tab}{W_array} += {prefactor} * {t_terms[partition[0]].string}\n"
            else:
                return_string += f"{tab}{W_array} += {prefactor} * {t_terms[partition[0]].string}\n"
            # continue onto the next partition
            continue

        # compile a list of the einsum calls
        einsum_list = _optimized_w_einsum_list(partition, order) if opt_einsum else _w_einsum_list(partition, order)

        if len(partition) == order:
            # join them together
            sum_of_einsums = " + ".join(einsum_list)
            # add the whole line to the main string
            contribution_string += f"{tab}{W_array} += {prefactor} * ({sum_of_einsums})\n"
        else:
            # join them together
            sum_of_einsums = f" +\n{tab}{tab}".join(einsum_list)
            # add the whole line to the main string
            contribution_string += (
                f"{tab}{W_array} += {prefactor} * (\n"
                f"{tab}{tab}{sum_of_einsums}\n"
                f"{tab})\n"
            )

        # we need to add a `{tab}` to every line in the `contribution_string`
        # to space the lines correctly to account for the if statement
        if max(partition) >= 2:
            lines = contribution_string.splitlines()
            contribution_string = "".join([f"{tab}{line}\n" for line in lines])
        # add the contribution to the `return_string`
        return_string += contribution_string
    # end of loop
    return_string += f"{tab}return\n"
    return return_string


# ------------------------------------------------------- #
def _construct_w_function_definition(return_string, order, opt_einsum=False, iterator_name='optimized_einsum'):
    """Return the string containing the python code to prepare the function definition and unpack the t_args.
    This function and `_optimized_w_einsum_list` need to agree on the name of the
    iterator containing the optimized paths. At the moment it is named `optimized_einsum`.
    """

    # this line of python code labels the terms in the `t_args` tuple
    t_arg_string = ", ".join([t_terms[n].string for n in range(1, order+1)] + ["*unusedargs"])

    if not opt_einsum:
        return_string += (
            f"\ndef _calculate_order_{order}_w_operator(A, N, t_args, ansatz, truncation):\n"
            f'{tab}"""Calculate the order {order} W operator for use in the calculation of the residuals."""\n'
            f"{tab}# unpack the `t_args`\n"
            f"{tab}{t_arg_string} = t_args\n"
        )
    else:
        return_string += (
            f"\ndef _calculate_order_{order}_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_opt_path_list, vecc_opt_path_list):\n"
            f'{tab}"""Calculate the order {order} W operator for use in the calculation of the residuals.\n'
            f'{tab}Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'
            f'{tab}"""\n'
            f"{tab}# unpack the `t_args`\n"
            f"{tab}{t_arg_string} = t_args\n"
        )

    return return_string


def _write_w_function_strings(order, opt_einsum=False):
    """ x """
    assert order <= 6, "Can only handle up to 6th order due to `einsum_surface_tags` limit"

    return_string = ""  # we store the output in this string

    # prepare the function definition, unpacking of arguments, and initialization of the W array
    return_string += _construct_w_function_definition(return_string, order, opt_einsum)

    W_array = f"W_{order}"  # name of the W operator

    # initialize the W array
    return_string += (
        f"{tab}# Creating the {num_tag[order]} order W operator\n"
        f"{tab}{W_array} = np.zeros(({', '.join(['A','A',] + ['N',]*order)}), dtype=complex)\n"
    )

    # special base cases, zero and 1st order
    if order == 0:
        raise Exception("We should not call `_write_w_function_strings` with order 0.")
    if order == 1:
        return_string += (
            f"{tab}# Singles contribution\n"
            f"{tab}{W_array} += t_i\n"
            f"{tab}return {W_array}\n"
        )
        # return, no more code is needed for order 1 case
        return return_string

    # ------------------------------------------------------- #
    return_string += "\n"  # spacing the text to make code easier to read
    # ------------------------------------------------------- #
    # generate the appropriate strings `vemx_operations` and `vecc_operations
    # which are used in the f-strings about 25 lines below

    opt_string = "_optimized" if opt_einsum else ""
    vemx_opt_path_string = ", vemx_opt_path_list" if opt_einsum else""
    vecc_opt_path_string = ", vecc_opt_path_list" if opt_einsum else""

    if order < 2:
        vemx_operations = f"{tab}{tab}pass  # no VECI/CC (mixed) contributions for order < 2\n"
        vecc_operations = f"{tab}{tab}pass  # no VECC contributions for order < 4\n"
    else:
        vemx_operations = (
            f"{tab}{tab}_add_order_{order}_vemx_contributions{opt_string}({W_array}, t_args, truncation{vemx_opt_path_string})\n"
        )
        if order < 4:
            vecc_operations = (
                f"{tab}{tab}_add_order_{order}_vemx_contributions{opt_string}({W_array}, t_args, truncation{vemx_opt_path_string})\n"
                f"{tab}{tab}pass  # no VECC contributions for order < 4\n"
            )
        else:
            vecc_operations = (
                f"{tab}{tab}_add_order_{order}_vemx_contributions{opt_string}({W_array}, t_args, truncation{vemx_opt_path_string})\n"
                f"{tab}{tab}_add_order_{order}_vecc_contributions{opt_string}({W_array}, t_args, truncation{vecc_opt_path_string})\n"
            )

    # ------------------------------------------------------- #
    veci_prefactor = _generate_w_operator_prefactor((order,))
    # finally, we add it all together
    return_string += (
        f"{tab}# add the VECI contribution\n"
        f"{tab}if truncation.{taylor_series_order_tag[order]}:\n"
        f"{tab}{tab}{W_array} += {veci_prefactor} * {t_terms[order].string}\n"
        # disconnected terms
        f"{tab}if ansatz.VE_MIXED:\n"
        f"{vemx_operations}"
        f"{tab}elif ansatz.VECC:\n"
        f"{vecc_operations}"
    )
    # ------------------------------------------------------- #
    return_string += "\n"  # spacing the text to make code easier to read
    # ------------------------------------------------------- #
    return_string += (
        f"{tab}# Symmetrize the W operator\n"
        f"{tab}symmetric_w = symmetrize_tensor(N, {W_array}, order={order})\n"
        f"{tab}return symmetric_w\n"
    )

    return return_string


# ------------------------------------------------------- #
def _write_master_w_compute_function(max_order, opt_einsum=False):
    """Write the wrapper function which `vibronic_hamiltonian.py` calls.
    This functions decides, up to what order of W operator is calculated,
    based on the truncation level inside the namedtuple `truncation`.
    """
    if not opt_einsum:
        function_string = "w_{0} = _calculate_order_{0}_w_operator(A, N, t_args, ansatz, truncation)"

        string = f'''
            def compute_w_operators(A, N, t_args, ansatz, truncation):
                """Compute a number of W operators depending on the level of truncation."""

                if not truncation.singles:
                    raise Exception(
                        "It appears that `singles` is not true, this cannot be.\\n"
                        "Something went terribly wrong!!!\\n\\n"
                        f"{{truncation}}\\n"
                    )

                {function_string.format(1)}
                {function_string.format(2)}
                {function_string.format(3)}

                if not truncation.doubles:
                    return w_1, w_2, w_3, None, None, None
                else:
                    {function_string.format(4)}

                if not truncation.triples:
                    return w_1, w_2, w_3, w_4, None, None
                else:
                    {function_string.format(5)}

                if not truncation.quadruples:
                    return w_1, w_2, w_3, w_4, w_5, None
                else:
                    {function_string.format(6)}

                if not truncation.quintuples:
                    return w_1, w_2, w_3, w_4, w_5, w_6
                else:
                    raise Exception(
                        "Attempting to calculate W^7 operator (quintuples)\\n"
                        "This is currently not implemented!!\\n"
                    )
        '''
    else:
        function_string = (
            "w_{0} = _calculate_order_{0}_w_operator_optimized("
            "A, N, t_args, ansatz, truncation, vemx_optimized_paths[{1}], vecc_optimized_paths[{1}]"
            ")"
        )

        string = f'''
            def compute_w_operators_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths, vecc_optimized_paths):
                """Compute a number of W operators depending on the level of truncation."""

                if not truncation.singles:
                    raise Exception(
                        "It appears that `singles` is not true, this cannot be.\\n"
                        "Something went terribly wrong!!!\\n\\n"
                        f"{{truncation}}\\n"
                    )

                {function_string.format(1, 0)}
                {function_string.format(2, 1)}
                {function_string.format(3, 2)}

                if not truncation.doubles:
                    return w_1, w_2, w_3, None, None, None
                else:
                    {function_string.format(4, 3)}

                if not truncation.triples:
                    return w_1, w_2, w_3, w_4, None, None
                else:
                    {function_string.format(5, 4)}

                if not truncation.quadruples:
                    return w_1, w_2, w_3, w_4, w_5, None
                else:
                    {function_string.format(6, 5)}

                if not truncation.quintuples:
                    return w_1, w_2, w_3, w_4, w_5, w_6
                else:
                    raise Exception(
                        "Attempting to calculate W^7 operator (quintuples)\\n"
                        "This is currently not implemented!!\\n"
                    )
        '''

    # remove three indents from string block
    lines = string.splitlines()
    # 3 consecutive tabs that we are ignoring/skipping
    trimmed_string = "\n".join([line[tab_length*3:] for line in lines])

    return trimmed_string


# ------------------------------------------------------- #
def _t_term_shape_string(order):
    """Return the string `(A, A, N, ...)` with `order` number of `N`'s."""
    return f"({', '.join(['A','A',] + ['N',]*order)})"


def _contracted_expressions(partition_list, order):
    """Return a list of each of the `oe.contract_expression` calls
    for a W operator of order `order`.
    """
    exp_list = []

    # for each partition (the mathematical term) of the integer `order`
    for partition in partition_list:

        # there is nothing to optimize for the N^th case
        # we simply add the largest t_term to the W operator
        # no multiplication required
        if len(partition) == 1:
            continue

        # the unique permutations of the `partition` of the integer `order`
        combinations = unique_permutations(partition)
        # the einsum indices for the surface dimensions
        surf_index = _generate_surface_index(partition)
        # the einsum indices for the normal mode dimensions
        mode_index = _generate_mode_index(partition, order)

        temp_list = []

        # `combinations` is a list of tuples such as (2, 2, 1, ) or (5,)
        for i, tupl in enumerate(combinations):
            # the input dimensions are two character strings representing the surface dimensions
            # plus 1 or more characters representing the normal mode dimensions
            in_dims = ", ".join([surf_index[a]+mode_index[i][a] for a in range(len(surf_index))])
            # the output dimension is the same for all einsum calls for a given `partition` argument
            out_dims = f"ab{tag_str[0:order]}"
            # the shape of the t_terms are (A, A, N, ...) where the number of N dimensions is
            # determined by the integer elements of each tuple `tupl`
            # so (2, 1) -> ("(A, A, N, N)", "(A, A, N")
            pterms = ", ".join([_t_term_shape_string(n) for n in tupl])
            # old_print_wrapper(f"np.einsum('{in_dims}->{out_dims}', {pterms})")
            temp_list.append(f"oe.contract_expression('{in_dims}->{out_dims}', {pterms}),\n")

        exp_list.append([max(partition), ] + temp_list)

    return exp_list


def _write_optimized_vemx_paths_function(max_order):
    """Return strings to write all the `oe.contract_expression` calls.
    Unfortunately the code got a lot messier when I had to add in the truncation if statements.
    It should get a rework/factorization at some point
    """
    assert max_order <= 6, "Only implemented up to 6th order"

    string = (
        f"\ndef compute_optimized_vemx_paths(A, N, truncation):\n"
        f'{tab}"""Calculate optimized paths for the VECI/CC (mixed) einsum calls up to `highest_order`."""\n'
        "\n"
        f"{tab}order_2_list, order_3_list = [], []\n"
        f"{tab}order_4_list, order_5_list, order_6_list = [], [], []\n"
        "\n"
    )

    # there are no VECI/CC (mixed) contributions for order < 2
    optimized_vemx_orders = list(range(2, max_order+1))

    for order in optimized_vemx_orders:

        # generate all the elements in the `order_{order}_list`
        partitions = generate_linked_disconnected_partitions_of_n(order)
        optimized_path_list = _contracted_expressions(partitions, order)

        for optimized_paths in optimized_path_list:
            current_max_order = optimized_paths[0]
            del optimized_paths[0]
            # old_print_wrapper('ZZ', optimized_paths)

            # the string representation (doubles, triples, quadruples... etc)
            contribution_name = lambda n: taylor_series_order_tag[n].lower()

            # we need to a big long string, and also remove the first two opt paths
            optimized_paths = "".join([
                s.replace("oe.contract", f"{tab}{tab}{tab}oe.contract") for s in optimized_paths
            ])

            string += (
                f"{tab}if truncation.{contribution_name(current_max_order)}:\n"
                f"{tab}{tab}order_{order}_list.extend([\n"
                f"{optimized_paths}"
                f"{tab}{tab}])\n"
                "\n"
            )

    return_list = ', '.join(['[]', ] + [f'order_{order}_list' for order in optimized_vemx_orders])
    string += f"\n{tab}return [{return_list}]\n"

    return string


def _write_optimized_vecc_paths_function(max_order):
    """Return strings to write all the `oe.contract_expression` calls.
    Unfortunately the code got a lot messier when I had to add in the truncation if statements.
    It should get a rework/factorization at some point
    """
    assert max_order <= 6, "Only implemented up to 6th order"

    string = (
        f"\ndef compute_optimized_vecc_paths(A, N, truncation):\n"
        f'{tab}"""Calculate optimized paths for the VECC einsum calls up to `highest_order`."""\n'
        "\n"
        f"{tab}order_4_list, order_5_list, order_6_list = [], [], []\n"
        "\n"
    )

    # there are no VECC contributions for order < 4
    optimized_vecc_orders = list(range(4, max_order+1))

    # since we need at least doubles for things to matter:
    string += (
        f"{tab}if not truncation.doubles:\n"
        f"{tab}{tab}log.warning('Did not calculate optimized VECC paths of the dt amplitudes')\n"
        f"{tab}{tab}return {[[], ]*6}\n"
        "\n"
    )

    for order in optimized_vecc_orders:

        # generate all the elements in the `order_{order}_list`
        partitions = generate_un_linked_disconnected_partitions_of_n(order)
        optimized_path_list = _contracted_expressions(partitions, order)

        for optimized_paths in optimized_path_list:
            current_max_order = optimized_paths[0]
            del optimized_paths[0]
            # old_print_wrapper('ZZ', optimized_paths)

            # the string representation (doubles, triples, quadruples... etc)
            contribution_name = lambda n: taylor_series_order_tag[n].lower()

            # we need to a big long string, and also remove the first two opt paths
            optimized_paths = "".join([
                s.replace("oe.contract", f"{tab}{tab}{tab}oe.contract") for s in optimized_paths
            ])

            string += (
                f"{tab}if truncation.{contribution_name(current_max_order)}:\n"
                f"{tab}{tab}order_{order}_list.extend([\n"
                f"{optimized_paths}"
                f"{tab}{tab}])\n"
                "\n"
            )

    return_list = ', '.join(['[]', '[]', '[]'] + [f'order_{order}_list' for order in optimized_vecc_orders])
    # return_list = ', '.join( + [f'order_{order}_list' for order in optimized_vemx_orders])
    string += f"\n{tab}return [{return_list}]\n"

    return string


# ------------------------------------------------------- #
def generate_w_operators_string(max_order, s1=75, s2=28):
    """Return a string containing the python code to generate w operators up to (and including) `max_order`.
    Requires the following header: `"import numpy as np\nfrom math import factorial"`.
    """

    spacing_line = "# " + "-"*s1 + " #\n"

    def named_line(name, width):
        """ x """
        return "# " + "-"*width + f" {name} " + "-"*width + " #\n"

    # ------------------------------------------------------------------------------------------- #
    # header for default functions (as opposed to the optimized functions)
    string = spacing_line + named_line("DEFAULT FUNCTIONS", s2) + spacing_line
    # ----------------------------------------------------------------------- #
    # header for VECI/CC (MIXED) functions
    string += '\n' + named_line("VECI/CC CONTRIBUTIONS", s2)
    # generate the VECI/CC (MIXED) contributions to W
    string += "".join([_generate_vemx_contributions(order=order) for order in range(1, max_order+1)])
    # ----------------------------------------------------------------------- #
    # header for VECC contributions
    string += '\n' + named_line("VECC CONTRIBUTIONS", s2)
    # generate the VECC contributions to W
    string += "".join([_generate_vecc_contributions(order=order) for order in range(1, max_order+1)])
    # ----------------------------------------------------------------------- #
    # header for w operator functions
    string += '\n' + named_line("W OPERATOR FUNCTIONS", s2)
    # generate the w operator function
    string += "".join([_write_w_function_strings(order=order) for order in range(1, max_order+1)])
    # ----------------------------------------------------------------------- #
    # write the master function
    string += _write_master_w_compute_function(max_order) + "\n"
    # ------------------------------------------------------------------------------------------- #
    #
    # ------------------------------------------------------------------------------------------- #
    # header for optimized functions
    string += spacing_line + named_line("OPTIMIZED FUNCTIONS", s2-1) + spacing_line
    # ----------------------------------------------------------------------- #
    # header for VECI/CC (MIXED) functions
    string += '\n' + named_line("VECI/CC CONTRIBUTIONS", s2)
    # generate the VECI/CC (MIXED) contributions to W
    string += "".join([_generate_vemx_contributions(order=order, opt_einsum=True) for order in range(1, max_order+1)])
    # ----------------------------------------------------------------------- #
    # header for VECC contributions
    string += '\n' + named_line("VECC CONTRIBUTIONS", s2)
    # generate the VECC contributions to W
    string += "".join([_generate_vecc_contributions(order=order, opt_einsum=True) for order in range(1, max_order+1)])
    # ----------------------------------------------------------------------- #
    # header for w operator functions
    string += '\n' + named_line("W OPERATOR FUNCTIONS", s2)
    # generate the w operator function
    string += "".join([_write_w_function_strings(order=order, opt_einsum=True) for order in range(1, max_order+1)])
    # ----------------------------------------------------------------------- #
    # write the master function
    string += _write_master_w_compute_function(max_order, opt_einsum=True) + "\n"
    # ------------------------------------------------------------------------------------------- #
    #
    # ------------------------------------------------------------------------------------------- #
    # header for optimized paths function
    string += '\n' + named_line("OPTIMIZED PATHS FUNCTION", s2)
    # write the code for generating optimized paths for VECI/CC (mixed) contributions
    string += _write_optimized_vemx_paths_function(max_order) + '\n'
    # write the code for generating optimized paths for VECC contributions
    string += _write_optimized_vecc_paths_function(max_order) + '\n'
    # ------------------------------------------------------------------------------------------- #
    return string


def generate_w_operator_equations_file(max_w_order, path="./w_operator_equations.py"):
    """Generates and saves to a file the code to calculate the w operator equations for the CC approach."""

    # start with the import statements
    file_data = (
        "# system imports\n"
        "from math import factorial\n"
        "\n"
        "# third party imports\n"
        "import numpy as np\n"
        "import opt_einsum as oe\n"
        "\n"
        "# local imports\n"
        "from .symmetrize import symmetrize_tensor\n"
        "from ..log_conf import log\n"
        "\n"
    )

    # write the functions to calculate the W operators
    file_data += generate_w_operators_string(max_order=max_w_order)

    # save data
    with open(path, 'w') as fp:
        fp.write(file_data)

    return

# ----------------------------------------------------------------------------------------------- #
# ---------------------  GENERATING T AMPLITUDE DERIVATIVE EQUATIONS  --------------------------- #
# ----------------------------------------------------------------------------------------------- #


dt_terms = [
    None,
    t_term_namedtuple("dt_i", 1, "(A, A, N)"),
    t_term_namedtuple("dt_ij", 2, "(A, A, N, N)"),
    t_term_namedtuple("dt_ijk", 3, "(A, A, N, N, N)"),
    t_term_namedtuple("dt_ijkl", 4, "(A, A, N, N, N, N)"),
    t_term_namedtuple("dt_ijklm", 5, "(A, A, N, N, N, N, N)"),
    t_term_namedtuple("dt_ijklmn", 6, "(A, A, N, N, N, N, N, N)"),
    t_term_namedtuple("dt_ijklmno", 7, "(A, A, N, N, N, N, N, N, N)"),
    t_term_namedtuple("dt_ijklmnop", 8, "(A, A, N, N, N, N, N, N, N, N)"),
]


# ----------------------------------------------------------------------------------------------- #
def _generate_disconnected_einsum_operands_list(dt_index, tupl):
    """Generate a string representing the list of operands for an einsum call.
    the names of the arguments to the einsum call are stored in the lists `t_terms`, `dt_terms`
    and the objects are accessed using each integer in the tuple (2, 1) -> ("t_ij", "dt_i")
    """
    term_list = []
    for t_index, num in enumerate(tupl):
        # old_print_wrapper(f"{dt_index=} {t_index=} {num=} {tupl=}")
        if dt_index == t_index:
            term_list.append(dt_terms[num].string)
        else:
            term_list.append(t_terms[num].string)

    einsum_term_list = ", ".join(term_list)
    return einsum_term_list


def _generate_disconnected_einsum_function_call_list(partition, order):
    """Returns a list of strings. Each string represents a call to np.einsum.
    The list is filled relative to the specific `partition` that is being calculated.
    here we want to do all permutations like so:
    2, 1, 1
    1, 2, 1
    1, 1, 2
    but in addition we want to permute which of them is dt term
    and because dt and t don't commute we need to count all perms
    d2, 1, 1
    1, d2, 1
    1, 1, d2
    --------
    d1, 1, 2
    d1, 2, 1
    --------
    1, d1, 2
    2, d1, 1
    --------
    1, 2, d1
    2, 1, d1
    """

    # the unique permutations of the `partition` of the integer `order`
    combinations = unique_permutations(partition)
    # the einsum indices for the surface dimensions
    surf_index = _generate_surface_index(partition)
    # the einsum indices for the normal mode dimensions this is different than the w generators
    mode_index = _generate_mode_index(partition, order)

    return_list = []  # store return values here

    # `combinations` is a list of tuples such as (2, 2, 1, ) or (5,)
    for i, tupl in enumerate(combinations):
        # the input dimensions are two character strings representing the surface dimensions
        # plus 1 or more characters representing the normal mode dimensions
        in_dims = ", ".join([surf_index[a]+mode_index[i][a] for a in range(len(surf_index))])
        # the output dimension is the same for all einsum calls for a given `partition` argument
        out_dims = f"ab{tag_str[0:order]}"

        # the names of the arguments to the einsum call are stored in the lists `t_terms`, `dt_terms`
        # and the objects are accessed using each integer in the tuple (2, 1) -> ("t_ij", "dt_i")
        for dt_index in range(len(partition)):
            operands = _generate_disconnected_einsum_operands_list(dt_index, tupl)
            return_list.append(f"np.einsum('{in_dims}->{out_dims}', {operands})")

    return return_list


def _generate_optimized_disconnected_einsum_function_call_list(partition, order, iterator_name='optimized_einsum'):
    """Returns a list of strings. Each string represents a call to np.einsum.
    The list is filled relative to the specific `partition` that is being calculated.

    This function, `_construct_linked_disconnected_definition`, and `_construct_un_linked_disconnected_definition`
    need to agree on the name of the iterator containing the optimized paths. At the moment it is named `optimized_einsum`.
    """

    # the unique permutations of the `partition` of the integer `order`
    combinations = unique_permutations(partition)

    return_list = []  # store return values here

    # `combinations` is a list of tuples such as (2, 2, 1, ) or (5,)
    for i, tupl in enumerate(combinations):
        # the names of the arguments to the einsum call are stored in the lists `t_terms`, `dt_terms`
        # and the objects are accessed using each integer in the tuple (2, 1) -> ("t_ij", "dt_i")
        for dt_index in range(len(partition)):
            operands = _generate_disconnected_einsum_operands_list(dt_index, tupl)
            return_list.append(f"next({iterator_name})({operands})")

    return return_list


# ----------------------------------------------------------------------------------------------- #
def _construct_linked_disconnected_definition(return_string, order, opt_einsum=False, iterator_name='optimized_einsum'):
    """Return the string containing the python code to prepare the function definition and unpack the t_args.
    This function and `_optimized_linked_disconnected_einsum_list` need to agree on the name of the
    iterator containing the optimized paths. At the moment it is named `optimized_einsum`.
    """

    # this line of python code labels the terms in the `t_args` tuple
    t_arg_string = ", ".join([t_terms[n].string for n in range(1, order)] + ["*unusedargs"])
    # this line of python code labels the terms in the `dt_args` tuple
    dt_arg_string = ", ".join([dt_terms[n].string for n in range(1, order)] + ["*unusedargs"])

    if not opt_einsum:
        return_string += (
            f"\ndef _order_{order}_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n"
            f'{tab}"""Calculate all uniquely linked disconnected terms generated from the wave operator ansatz.\n'
            f'{tab}This means for order 5 we include terms such as (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'
            f'{tab}But not terms (5), (3, 2), (2, 2, 1)\n'
            f'{tab}"""\n'
            f"{tab}# unpack the `t_args` and 'dt_args'\n"
            f"{tab}{t_arg_string} = t_args\n"
            f"{tab}{dt_arg_string} = dt_args\n"
        )
    else:
        return_string += (
            f"\ndef _order_{order}_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list):\n"
            f'{tab}"""Calculate all uniquely linked disconnected terms generated from the wave operator ansatz.\n'
            f'{tab}This means for order 5 we include terms such as (4, 1), (3, 1, 1), (2, 1, 1, 1)\n'
            f'{tab}But not terms (5), (3, 2), (2, 2, 1), (1, 1, 1, 1, 1)\n'
            f'{tab}Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'
            f'{tab}"""\n'
            f"{tab}# unpack the `t_args` and 'dt_args'\n"
            f"{tab}{t_arg_string} = t_args\n"
            f"{tab}{dt_arg_string} = dt_args\n"
        )
        return_string += (
            f"{tab}# make an iterable out of the `opt_path_list`\n"
            f"{tab}{iterator_name} = iter(opt_path_list)\n"
        )

    return return_string


def _write_linked_disconnected_strings(order, opt_einsum=False):
    """Output the python code which generates all CI/CC terms to subtract from dt."""

    # may want to switch to numerical arguments for einsum
    assert order <= 6, "Can only handle up to 6th order due to `einsum_surface_tags` limit"
    return_string = ""  # we store the output in this string

    # prepare the function definition, unpacking of arguments, and initialization of the W array
    return_string += _construct_linked_disconnected_definition(return_string, order, opt_einsum)

    # special exception for order less than 2
    if order < 2:
        lst = return_string.split('"""')
        exception = (
            "\n"
            f"{tab}raise Exception(\n"
            f'{tab}{tab}"the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"\n'
            f'{tab}{tab}"which requires a residual of at least 2nd order"\n'
            f"{tab})\n"
        )
        return '\"\"\"'.join([lst[0], "Exists for error checking.", exception])

    # name of the return array
    array = "linked_disconnected_terms"

    # initialize the return array
    return_string += (
        f"{tab}# Creating the {num_tag[order]} order return array\n"
        f"{tab}{array} = np.zeros(({', '.join(['A','A',] + ['N',]*order)}), dtype=complex)\n"
    )

    for partition in generate_linked_disconnected_partitions_of_n(order):

        # Label the order of this partition's contribution
        return_string += f"{tab}# the {partition} term\n"

        # make the prefactor
        prefactor = _generate_w_operator_prefactor(partition)

        # compile a list of the einsum calls
        if opt_einsum:
            einsum_list = _generate_optimized_disconnected_einsum_function_call_list(partition, order)
        else:
            einsum_list = _generate_disconnected_einsum_function_call_list(partition, order)

        # join them together
        sum_of_einsums = f" +\n{tab}{tab}".join(einsum_list)

        # place the einsums between the parentheses
        return_string += (
            f"{tab}{array} += {prefactor} * (\n"
            f"{tab}{tab}{sum_of_einsums}\n"
            f"{tab})\n"
        )

    return_string += f"\n{tab}return {array}\n"

    return return_string


# ----------------------------------------------------------------------------------------------- #
def _construct_un_linked_disconnected_definition(return_string, order, opt_einsum=False, iterator_name='optimized_einsum'):
    """Return the string containing the python code to prepare the function definition and unpack the t_args.
    This function and `_optimized_un_linked_disconnected_einsum_list` need to agree on the name of the
    iterator containing the optimized paths. At the moment it is named `optimized_einsum`.
    """

    # this line of python code labels the terms in the `t_args` tuple
    t_arg_string = ", ".join([t_terms[n].string for n in range(1, order)] + ["*unusedargs"])
    # this line of python code labels the terms in the `dt_args` tuple
    dt_arg_string = ", ".join([dt_terms[n].string for n in range(1, order)] + ["*unusedargs"])

    if not opt_einsum:
        return_string += (
            f"\ndef _order_{order}_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n"
            f'{tab}"""Calculate all uniquely un-linked disconnected terms generated from the wave operator ansatz.\n'
            f'{tab}This means for order 5 we include terms such as (3, 2), (2, 2, 1)\n'
            f'{tab}But not terms (5), (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'
            f'{tab}"""\n'
            f"{tab}# unpack the `t_args` and 'dt_args'\n"
            f"{tab}{t_arg_string} = t_args\n"
            f"{tab}{dt_arg_string} = dt_args\n"
        )
    else:
        return_string += (
            f"\ndef _order_{order}_un_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list):\n"
            f'{tab}"""Calculate all uniquely un-linked disconnected terms generated from the wave operator ansatz.\n'
            f'{tab}This means for order 5 we include terms such as (3, 2), (2, 2, 1)\n'
            f'{tab}But not terms (5), (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'
            f'{tab}Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'
            f'{tab}"""\n'
            f"{tab}# unpack the `t_args` and 'dt_args'\n"
            f"{tab}{t_arg_string} = t_args\n"
            f"{tab}{dt_arg_string} = dt_args\n"
        )
        return_string += (
            f"{tab}# make an iterable out of the `opt_path_list`\n"
            f"{tab}{iterator_name} = iter(opt_path_list)\n"
        )

    return return_string


def _write_un_linked_disconnected_strings(order, opt_einsum=False):
    """Output the python code which generates all CC terms to subtract from dt."""

    # may want to switch to numerical arguments for einsum
    assert order <= 6, "Can only handle up to 6th order due to `einsum_surface_tags` limit"
    return_string = ""  # we store the output in this string

    # prepare the function definition, unpacking of arguments, and initialization of the W array
    return_string += _construct_un_linked_disconnected_definition(return_string, order, opt_einsum)

    # special exception for order less than 2
    if order < 4:
        lst = return_string.split('"""')
        exception = (
            "\n"
            f"{tab}raise Exception(\n"
            f'{tab}{tab}"the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n'
            f'{tab}{tab}"which requires a residual of at least 4th order"\n'
            f"{tab})\n"
        )
        return '\"\"\"'.join([lst[0], "Exists for error checking.", exception])

    # name of the return array
    array = "un_linked_disconnected_terms"

    # initialize the return array
    return_string += (
        f"{tab}# Creating the {num_tag[order]} order return array\n"
        f"{tab}{array} = np.zeros(({', '.join(['A','A',] + ['N',]*order)}), dtype=complex)\n"
    )

    for partition in generate_un_linked_disconnected_partitions_of_n(order):

        # Label the order of this partition's contribution
        return_string += f"{tab}# the {partition} term\n"

        # make the prefactor
        prefactor = _generate_w_operator_prefactor(partition)

        # compile a list of the einsum calls
        if opt_einsum:
            einsum_list = _generate_optimized_disconnected_einsum_function_call_list(partition, order)
        else:
            einsum_list = _generate_disconnected_einsum_function_call_list(partition, order)

        # join them together
        sum_of_einsums = f" +\n{tab}{tab}".join(einsum_list)

        # place the einsums between the parentheses
        return_string += (
            f"{tab}{array} += {prefactor} * (\n"
            f"{tab}{tab}{sum_of_einsums}\n"
            f"{tab})\n"
        )

    return_string += f"\n{tab}return {array}\n"

    return return_string


# ----------------------------------------------------------------------------------------------- #
def _construct_dt_amplitude_definition(return_string, order, opt_einsum=False, iterator_name='optimized_einsum'):
    """Return the string containing the python code to prepare the function definition,
    unpack the t_args, and initialize the W array.

    This function and `_construct_dt_amplitude_definition` need to agree on the name of the
    iterator containing the optimized paths. At the moment it is named `optimized_einsum`.
    """

    # this line of python code labels the terms in the `t_args` tuple
    w_arg_string = ", ".join([w_dict[n] for n in range(1, order+1)] + ["*unusedargs"])

    if not opt_einsum:
        return_string += (
            f"\ndef _calculate_order_{order}_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):\n"
            f'{tab}"""Calculate the derivative of the {order} t-amplitude for use in the calculation of the residuals."""\n'
            f"{tab}# unpack the `w_args`\n"
            f"{tab}{w_arg_string} = w_args\n"
        )
    else:
        return_string += (
            f"\ndef _calculate_order_{order}_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_epsilon, opt_path_list):\n"
            f'{tab}"""Calculate the derivative of the {order} t-amplitude for use in the calculation of the residuals.\n'
            f'{tab}Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'
            f'{tab}"""\n'
            f"{tab}# unpack the `w_args`\n"
            f"{tab}{w_arg_string} = w_args\n"
        )
        if order >= 2:
            return_string += (
                f"{tab}# make an iterable out of the `opt_path_list`\n"
                f"{tab}{iterator_name} = iter(opt_path_list)\n"
            )

    return return_string


def _write_dt_amplitude_strings(order, opt_einsum=False):
    """ output the string of the function `compute_CI_from_CC` which generates all the W's """

    # may want to switch to numerical arguments for einsum
    assert order <= 6, "Can only handle up to 6th order due to `einsum_surface_tags` limit"
    return_string = ""  # we store the output in this string

    # prepare the function definition, unpacking of arguments, and initialization of the W array
    return_string += _construct_dt_amplitude_definition(return_string, order, opt_einsum)

    mode_subscripts = f"{tag_str[0:order]}"

    # initialize the W array
    return_string += (
        f"{tab}# Calculate the {num_tag[order]} order residual\n"
        f"{tab}residual = residual_equations.calculate_order_{order}_residual(A, N, trunc, h_args, w_args)\n"
    )

    # generate the appropriate strings `vemx_operations` and `vecc_operations
    # which are used in a f-string about 50 lines below
    if not opt_einsum:
        prefactor = f"1/factorial({order})"

        epsilon_term = f"{prefactor} * np.einsum('ac{mode_subscripts},cb->ab{mode_subscripts}', w_{mode_subscripts}, epsilon)"

        if order < 2:
            vemx_operations = f"{tab}{tab}pass  # no linked disconnected terms for order < 2\n"
            vecc_operations = f"{tab}{tab}pass  # no un-linked disconnected terms for order < 4\n"
        else:
            vemx_operations = (
                f"{tab}{tab}residual -= _order_{order}_linked_disconnected_terms(A, N, trunc, t_args, dt_args)\n"
            )
            if order < 4:
                vecc_operations = (
                    f"{tab}{tab}residual -= _order_{order}_linked_disconnected_terms(A, N, trunc, t_args, dt_args)\n"
                    f"{tab}{tab}pass  # no un-linked disconnected terms for order < 4\n"
                )
            else:
                vecc_operations = (
                    f"{tab}{tab}residual -= _order_{order}_linked_disconnected_terms(A, N, trunc, t_args, dt_args)\n"
                    f"{tab}{tab}residual -= _order_{order}_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args)\n"
                )
    else:
        prefactor = f"1/factorial({order})"

        epsilon_term = f"{prefactor} * opt_epsilon(w_{mode_subscripts}, epsilon)"

        if order < 2:
            vemx_operations = f"{tab}{tab}pass  # no linked disconnected terms for order < 2\n"
            vecc_operations = f"{tab}{tab}pass  # no un-linked disconnected terms for order < 4\n"
        else:
            vemx_operations = (
                f"{tab}{tab}residual -= _order_{order}_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list)\n"
            )
            if order < 4:
                vecc_operations = (
                    f"{tab}{tab}residual -= _order_{order}_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list)\n"
                    f"{tab}{tab}pass  # no un-linked disconnected terms for order < 4\n"
                )
            else:
                vecc_operations = (
                    f"{tab}{tab}residual -= _order_{order}_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list)\n"
                    f"{tab}{tab}residual -= _order_{order}_un_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list)\n"
                )

    return_string += (
        # subtract epsilon term
        f"{tab}# subtract the epsilon term (which is R_0)\n"
        f"{tab}residual -= {epsilon_term}\n"
        "\n"
        # disconnected terms
        f"{tab}# subtract the disconnected terms\n"
        f"{tab}if ansatz.VECI:\n"
        f"{tab}{tab}pass  # veci does not include any disconnected terms\n"
        f"{tab}elif ansatz.VE_MIXED:\n"
        f"{vemx_operations}"
        f"{tab}elif ansatz.VECC:\n"
        f"{vecc_operations}\n"
    )

    return_string += (
        f"{tab}# Symmetrize the residual operator\n"
        f"{tab}dt_{mode_subscripts} = symmetrize_tensor(N, residual, order={order})\n"
        f"{tab}return dt_{mode_subscripts}\n"
    )

    return return_string


def _write_master_dt_amplitude_function(order, opt_einsum=False):
    """ x """
    t_term = t_terms[order].string
    dt_term = dt_terms[order].string
    name = taylor_series_order_tag[order]

    if not opt_einsum:
        string = f'''
            def solve_{name}_equations(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):
                """Compute the change in the {t_term} term ({name})"""

                if not trunc.{name}:
                    raise Exception(
                        "It appears that {name} is not true, this cannot be."
                        "Something went terribly wrong!!!"
                    )
                {dt_term} = _calculate_order_{order}_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args)
                return {dt_term}
        '''
    else:
        string = f'''
            def solve_{name}_equations_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list):
                """Compute the change in the {t_term} term ({name})"""

                if not trunc.{name}:
                    raise Exception(
                        "It appears that {name} is not true, this cannot be."
                        "Something went terribly wrong!!!"
                    )
                {dt_term} = _calculate_order_{order}_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list)
                return {dt_term}
        '''

    # remove three indents from string block
    lines = string.splitlines()
    # 3 consecutive tabs that we are ignoring/skipping
    trimmed_string = "\n".join([line[tab_length*3:] for line in lines])

    return trimmed_string


# ----------------------------------------------------------------------------------------------- #
def _write_optimized_dt_amplitude_paths_function(max_order):
    """Return strings to write all the `oe.contract_expression` calls.
    Unfortunately the code got a lot messier when I had to add in the truncation if statements.
    It should get a rework/factorization at some point
    """
    assert max_order < 7, "optimized paths only implemented up to 6th order"

    string = (
        f"\ndef compute_optimized_paths(A, N, truncation):\n"
        f'{tab}"""Calculate optimized paths for the einsum calls up to `highest_order`."""\n'
    )
    # we need three optimized lists:
    # 1 - optimized epsilon call
    # 2 - optimized linked disconnected calls
    # 3 - optimized un-linked disconnected calls

    optimized_orders = list(range(2, max_order+1))

    for order in optimized_orders:
        pass

    string += (
        "\n"
        f"{tab}order_1_list, order_2_list, order_3_list = [], [], []\n"
        f"{tab}order_4_list, order_5_list, order_6_list = [], [], []\n"
        "\n"
    )

    # return_list = ', '.join(['order_1_list',] + [f'order_{order}_list' for order in optimized_orders])
    return_list = None
    string += f"{tab}return [{return_list}]\n"

    return string


def generate_dt_amplitude_string(max_order, s1=75, s2=28):
    """Return a string containing the python code to generate dt up to (and including) `max_order`.
    Requires the following header: `"import numpy as np\nfrom math import factorial"`.
    """

    spacing_line = "# " + "-"*s1 + " #\n"

    def named_line(name, width):
        """ x """
        return "# " + "-"*width + f" {name} " + "-"*width + " #\n"

    # ------------------------------------------------------------------------------------------- #
    # header for default functions
    string = spacing_line + named_line("DEFAULT FUNCTIONS", s2) + spacing_line
    # ----------------------------------------------------------------------- #    # header for VECI functions
    string += '\n' + named_line("DISCONNECTED TERMS", s2)
    # generate the linked disconnected function
    string += "".join([_write_linked_disconnected_strings(order=order) for order in range(1, max_order+1)])
    # generate the un-linked disconnected function
    string += "".join([_write_un_linked_disconnected_strings(order=order) for order in range(1, max_order+1)])
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    # header for VECI functions
    string += '\n' + named_line("dt AMPLITUDES", s2)
    # generate the default VECI functions
    string += "".join([_write_dt_amplitude_strings(order=order) for order in range(1, max_order+1)])
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    # header for master functions
    string += '\n' + named_line("WRAPPER FUNCTIONS", s2)
    # write the master functions
    string += "".join([_write_master_dt_amplitude_function(order) for order in range(1, max_order+1)])

    string += '\n'
    # ------------------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------------------- #
    # header for optimized functions
    string += spacing_line + named_line("OPTIMIZED FUNCTIONS", s2-1) + spacing_line
    # ----------------------------------------------------------------------- #    # header for VECI functions
    string += '\n' + named_line("DISCONNECTED TERMS", s2)
    # generate the linked disconnected function
    string += "".join([_write_linked_disconnected_strings(order=order, opt_einsum=True) for order in range(1, max_order+1)])
    # generate the un-linked disconnected function
    string += "".join([_write_un_linked_disconnected_strings(order=order, opt_einsum=True) for order in range(1, max_order+1)])
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    # header for VECI functions
    string += '\n' + named_line("dt AMPLITUDES", s2)
    # generate the optimized functions
    string += "".join([_write_dt_amplitude_strings(order=order, opt_einsum=True) for order in range(1, max_order+1)])
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    # header for master functions
    string += '\n' + named_line("WRAPPER FUNCTIONS", s2)
    # write the master functions
    string += "".join([_write_master_dt_amplitude_function(order, opt_einsum=True) for order in range(1, max_order+1)])
    # ----------------------------------------------------------------------- #
    # header for optimized paths function
    string += '\n' + named_line("OPTIMIZED PATHS FUNCTION", s2)
    # write the code for generating the optimized paths
    string += _write_optimized_dt_amplitude_paths_function(max_order)

    string += '\n'
    # ------------------------------------------------------------------------------------------- #
    return string


def generate_dt_amplitude_equations_file(max_w_order, path="./dt_amplitude_equations.py"):
    """Generates and saves to a file the code to calculate the t-amplitude derivative equations for the CC approach."""

    # start with the import statements
    file_data = (
        "# system imports\n"
        "from math import factorial\n"
        "\n"
        "# third party imports\n"
        "import numpy as np\n"
        "import opt_einsum as oe\n"
        "\n"
        "# local imports\n"
        "from ..log_conf import log\n"
        "from .symmetrize import symmetrize_tensor\n"
        "from . import residual_equations\n"
        "\n"
    )

    # write the functions to calculate the derivatives of the t-amplitudes
    file_data += generate_dt_amplitude_string(max_order=max_w_order)

    # save data
    with open(path, 'w') as fp:
        fp.write(file_data)

    return


# ----------------------------------------------------------------------------------------------- #
# -------------------------  GENERATING FULL Z T symmetrized LATEX  ----------------------------- #
# ----------------------------------------------------------------------------------------------- #


# this serves the same purpose as `general_operator_namedtuple`
# we changed the typename from "operator" to "h_operator" to distinguish the specific role and improve debugging
h_operator_namedtuple = namedtuple('h_operator', ['rank', 'm', 'n'])


# these serve the same purpose as `hamiltonian_namedtuple`
# we changed the typename from "operator" to distinguish the specific role and improve debugging
z_operator_namedtuple = namedtuple('z_operator', ['maximum_rank', 'operator_list'])

"""
Note that the attributes for these `namedtuples` operate in a specific manner.
The general rule that can be followed is the m/n refers to the object it is an attribute of:
`lhs.m_h` refers to the `m` value of `lhs`, contracting with `h`, which implies that it contracts with the `n` of `h`.
`m_h` counts how many of the LHS's or z's m indices contract _omega_joining_with_itself the n indices of the h operator.
So a h_1 couples with z^1 but the values inside the z^1 `namedtuple` object would be:
    `m_l`=1, `n_l`=0, `m_lhs`=1, `n_lhs`=0, `m_r`=0, `n_r`=0.
"""

# the `m_t` and `n_t` are lists of integers whose length is == number of t's
# so LHS^2_1 z_1 h^1 z_1 would mean m_l = [1, ] and m_r = [1, ] and n_h = [1, ]
connected_lhs_operator_namedtuple = namedtuple(
    'connected_LHS',
    ['rank', 'm', 'n', 'm_l', 'n_l', 'm_h', 'n_h', 'm_r', 'n_r']
)
connected_h_z_operator_namedtuple = namedtuple(
    'connected_h_z',
    ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_l', 'n_l', 'm_r', 'n_r']
)
# ------------------------------------------------------------------------ #
connected_z_left_operator_namedtuple = namedtuple(
    'connected_z_left',
    ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_h', 'n_h', 'm_r', 'n_r']
)
connected_z_right_operator_namedtuple = namedtuple(
    'connected_z_right',
    ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_h', 'n_h', 'm_l', 'n_l']
)
# ------------------------------------------------------------------------ #
disconnected_z_left_operator_namedtuple = namedtuple(
    'disconnected_z_left',
    ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_h', 'n_h', 'm_r', 'n_r']
)
disconnected_z_right_operator_namedtuple = namedtuple(
    'disconnected_z_right',
    ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_h', 'n_h', 'm_l', 'n_l']
)

""" These are the indices used to label the h and z's in the generated latex"""
z_summation_indices = 'klmno'
z_unlinked_indices = 'yxwuv'


# ------------------------------------------------------------------------ #


def generate_z_operator(maximum_cc_rank, only_ground_state):
    """Return an `z_operator_namedtuple` whose attributes are determined by `maximum_cc_rank`.

    The `operator_list` contains all permutations of (`m`,`n`) for `m` & `n` in `range(maximum_cc_rank + 1)`.
    The name is a string of the chars `d` and `b` according to `m` and `n` respectively.
    `m` is associated with creation operators (d) and `n` is associated with annihilation operators (b).
    There is no ground state restriction on the operators since they act from both sides.
    Filter out of excited state terms happens later in the process.
    """
    return_list = []

    for m in range(maximum_cc_rank + 1):          # m is the upper label (creation operators)
        for n in range(maximum_cc_rank + 1 - m):  # n is the lower label (annihilation operators)

            if only_ground_state and (n > 0):
                continue

            name = "z"
            name += f"^{m}" if m > 0 else ""
            name += f"_{n}" if n > 0 else ""

            return_list.append(general_operator_namedtuple(name, m+n, m, n))

    return z_operator_namedtuple(maximum_cc_rank, return_list)
# ------------------------------------------------------------------------ #


def _write_t_symmetric_latex_from_lists(rank, fully):
    """Return the latex commands to write the provided terms.
    We use `join` to insert two backward's slashes \\ BETWEEN each line
    rather then adding them to end and having extra trailing slashes on the last line.
    The user is expected to manually copy the relevant lines from the text file into a latex file
    and generate the pdf themselves.
    """
    return_string = ""

    # special case for zero order equation
    if rank == 0:
        return_string = _make_latex(rank, fully)
        return return_string.replace("^{}", "").replace("_{}", "")

    # no ____ terms
    no_fully = ' '*4 + r'\textit{no fully connected terms}'

    return_string += _make_latex(rank, fully) if fully != [] else no_fully

    # remove all empty ^{}/_{} terms that are no longer needed
    return return_string.replace("^{}", "").replace("_{}", "")


def _generate_t_symmetric_latex_equations(omega, H, s_taylor_expansion, remove_f_terms=True):
    """Return a string containing latex code to be placed into a .tex file.
    For a given set of input arguments: (`omega`, `H`, `s_taylor_expansion`) we generate
    all possible and valid CC terms. Note that:
        - `omega` is an `omega_namedtuple` object
        - `H` is a `hamiltonian_namedtuple` object
        - `s_taylor_expansion` is one of :
            - a single `general_operator_namedtuple`
            - a list of `general_operator_namedtuple`s
            - a list of lists of `general_operator_namedtuple`s

    One possible input could be:
        - `omega` is the creation operator d
        - `H` is a Hamiltonian of rank two
        - `s_taylor_expansion` is the S^1 Taylor expansion term
    """

    simple_repr_list = []  # old list, not so important anymore, might remove in future
    valid_term_list = []   # store all valid Omega * h * (s*s*...s) terms here

    """ First we want to generate a list of valid terms.
    We start with the list of lists `s_taylor_expansion` which is processed by `_filter_out_valid_s_terms`.
    This function identifies valid pairings AND places those pairings in the `valid_term_list`.
    Specifically we replace the `general_operator_namedtuple`s with `connected_namedtuple`s and/or
    `disconnected_namedtuple`s.
    """
    for count, s_series_term in enumerate(s_taylor_expansion):

        if False:  # debugging
            old_print_wrapper(s_series_term, "-"*100, "\n\n")

        _filter_out_valid_s_terms(omega, H, s_series_term, simple_repr_list, valid_term_list, remove_f_terms=remove_f_terms)

    """ Next we take all terms and separate them into their respective groups """
    fully, linked, unlinked = _seperate_s_terms_by_connection(valid_term_list)

    if False:  # extra heavy debugging
        _debug_print_valid_term_list(valid_term_list)
        _debug_print_different_types_of_terms(fully, linked, unlinked)

    # write and return the latex code
    return _write_t_symmetric_latex_from_lists(omega.rank, fully)


def _generate_t_symmetric_left_hand_side(omega):
    """ Generate the latex code for the LHS (left hand side) of the CC equation.
    The order of the `omega` operator determines all terms on the LHS.
    """

    omega_order = omega.m + omega.n

    if omega_order == 0:
        return r'''i\left(\dv{\bs_{0,\gamma}}{\tau}\right)'''

    # generate all possible tuples (m, n) representing t terms t^m_n
    single_t_list = [[m, n] for m in range(0, omega_order+1) for n in range(0, omega_order+1) if ((n == m != 0) or (n != m))]

    all_combinations_list = []

    # generate all possible combinations of t^m_n
    # such as t_1, t^2, t^1 * t^1, t^1 * t^2_3, ... etc
    for length in range(1, omega_order+1):
        all_combinations_list.append(list(it.product(single_t_list, repeat=length)))

    """ Next we filter out the combinations that don't match omega.
    Suppose omega is o^2_1:
        - it can match with  (t^1 * t^1 * t_1) or (t^2_1) and so forth
        - it cant match with (t^1 * t^1 * t^1) or (t^1_1) and so forth
    """
    matched_set = set()
    for list_of_t_terms in all_combinations_list:
        for t_terms in list_of_t_terms:
            upper_sum = sum([t[0] for t in t_terms])  # the sum over all m superscripts (t^m)
            lower_sum = sum([t[1] for t in t_terms])  # the sum over all n superscripts (t_n)

            # remember that a t_1 contracts with o^1
            # so the `lower_sum` needs to be compared to o^n
            if omega.m == lower_sum and omega.n == upper_sum:
                """ The "filtering" is accomplished by adding sorted tuples of tuples to a set.
                We cannot use lists because sets require hashable elements (immutable) such as tuples.
                However with tuples, we can end up with duplicates like ((2, 0), (0, 1)) and ((0, 1), (2, 0)).
                So we have to:
                    - generate lists of tuples:             [(0, 1), (0, 2)]
                    - sort those lists in reverse order:    [(0, 2), (0, 1)]
                    - make a tuple from the list:           ((0, 2), (0, 1))
                    - add the tuple to `matched_set`
                """
                matched_set.add(tuple(sorted([tuple(x) for x in t_terms], reverse=True)))

    """ Transform the set into a list of lists sort by increasing length
    Two examples:
        - omega is `operator(name='bb', m=0, n=2)`
        then `sorted_list` is [[(2, 0)], [(1, 0), (1, 0)]]

        - omega is `operator(name='ddd', m=3, n=0)`
        then `sorted_list` is [[(0, 3)], [(0, 2), (0, 1)], [(0, 1), (0, 1), (0, 1)]]
    """
    sorted_list = sorted([[b for b in a] for a in matched_set], key=len)

    """ Next we generate the latex code for each t term represented by the (m, n) tuples
    One possible `t_group` could be:
        - [[(0, 2)], [(0, 1), (0, 1)]]
    and its corresponding `group_list` would be:
        - [['\\bt^{}_{ij}'], ['\\bt^{}_{i}', '\\bt^{}_{j}']]
    """
    latex_t_terms_list = []  # store the latex code for each valid t term here
    for t_group in sorted_list:
        count = 0  # keep track of what index labels have been used
        group_list = []

        for t in t_group:
            upper_label = summation_indices[count:count+t[0]]
            count += t[0]

            lower_label = summation_indices[count:count+t[1]]
            count += t[1]

            lower_label += r'\gamma' if lower_label == "" else r',\gamma'

            group_list.append(f"{bold_t_latex}^{{{upper_label}}}_{{{lower_label}}}")

        latex_t_terms_list.append(group_list)

    # create derivative terms
    derivative_list = []
    for t_group in latex_t_terms_list:
        # loop over each t in the group and take the derivative of that specific t
        for i in range(len(t_group)):
            string = ""

            # if there are t terms to the left of the current index
            if len(t_group[0:i]) > 0:
                string += f"{''.join(t_group[0:i])}"

            # the t term we are currently taking the derivative of
            string += rf'\dv{{{t_group[i]}}}{{\tau}}'

            # if there are t terms to the right of the current index
            if len(t_group[i:]) > 0:
                string += f"{''.join(t_group[i+1:])}"

            derivative_list.append(string)

    # order the derivative terms before the epsilon terms
    return_string = ' + '.join(derivative_list)
    return rf'''i\left({return_string}\right)'''
# ------------------------------------------------------------------------ #


def _make_z_symmetric_latex(rank, term_list, linked_condense=False, unlinked_condense=False, print_prefactors=True):
    """Return the latex commands to write the provided terms."""

    return_list = []  # store output here

    # prepare the common factors for condensing
    if unlinked_condense:
        common_unlinked_factor = prepare_condensed_terms(term_list, unlinked_condense=True)

    elif linked_condense:
        common_linked_factor_list = prepare_condensed_terms(term_list, linked_condense=True)
        # where we store the lists of latex code
        linked_return_list = [[] for i in range(len(common_linked_factor_list))]

    # prepare all the latex strings
    for term in term_list:
        # extract elements of list `term`
        LHS, h, t_list = term[0], term[1], term[2]
        old_print_wrapper(type(t_list), t_list)

        # make sure all s_terms are valid objects
        _validate_s_terms(t_list)

        term_string = ''

        # if needed add f prefactors
        # if h has unpaired lower terms this implies it would contract with LHS in a `db` fashion
        if _creates_f_prefactor(LHS, h):
            if LHS.m_h == 1:
                term_string += "f"
            else:
                term_string += f"f^{{{LHS.m_h}}}"

        # added by shanmei, which of f and fbar should go first?
        # if needed add fbar prefactors
        # if h has unpaired upper terms this implies it would contract with LHS in a `bd` fashion
        if _creates_fbar_prefactor(LHS, h):
            if LHS.n_h == 1:
                term_string += "\\bar{f}"
            else:
                term_string += f"\\bar{{f}}^{{{LHS.n_h}}}"

        # add any prefactors if they exist
        if print_prefactors:
            term_string += _build_latex_prefactor(h, t_list)

        # special treatment for condensing the linked disconnected terms
        if linked_condense:
            linked_list_index, new_t_list, t_offset_dict = _linked_condensed_adjust_t_terms(common_linked_factor_list, h, t_list)

            # prepare the t-amplitude terms
            t_string_list = _build_t_term_latex_group(new_t_list, h, t_offset_dict)

            # build the latex code representing this term in the sum
            h_offset = sum([t.m_o + t.n_o for t in common_linked_factor_list[linked_list_index]])
            term_string += _build_h_term_latex_labels(h, h_offset) + ''.join(t_string_list)

            # store the result
            linked_return_list[linked_list_index].append(term_string)

        else:
            # prepare the t-amplitude terms
            t_string_list = _build_t_term_latex_group(t_list, h=h)

            # build the latex code representing this term in the sum
            term_string += _build_h_term_latex_labels(h) + ''.join(t_string_list)

            # store the result
            return_list.append(term_string)

    # glue all the latex code together!!

    # special treatment to condense the unlinked disconnected terms
    if unlinked_condense:
        log.warning("This has not been tested for Hamiltonians of rank >= 3")

        # remove the common factor from each term
        common_latex = _build_t_term_latex(common_unlinked_factor)
        return_list = [term.replace(common_latex, '') for term in return_list]

        # old_print_wrapper(return_list)

        return f"({' + '.join(return_list)}){common_latex}"

    # special treatment to condense the linked disconnected terms
    if linked_condense:
        log.warning("This has not been tested for Hamiltonians of rank >= 3")

        return_strings = []

        for i, factor in enumerate(common_linked_factor_list):
            common_latex = ''.join(_build_t_term_latex_group(factor))
            # old_print_wrapper('z', i, linked_return_list[i], '\n\n')
            # old_print_wrapper(factor)

            # if common_latex == '\\bt^{z}_{}\\bt^{y}_{}':
            #     old_print_wrapper('z', common_latex)
            #     for term in linked_return_list[i]:
            #         if common_latex in term:
            #             old_print_wrapper('\n\n', term)
            #             sys.exit(0)

            #             if "\\bh^{z}_{}\\bt^{}_{y}\\bt^{x}_{}" in term:
            #                 old_print_wrapper('\n\n', term)
            #                 sys.exit(0)

            # linked_return_list[i] = [term.replace(common_latex, '') for term in linked_return_list[i]]
            # old_print_wrapper('z', i, linked_return_list[i])
            return_strings.append(f"({' + '.join(linked_return_list[i])}){common_latex}")

        # join the lists with the equation splitting string
        splitting_string = r'\\  &+  % split long equation'
        final_string = f"\n{tab}{splitting_string}\n".join(return_strings)

        return final_string

        # if rank == 1:
        #     return _glue_linked_disconnected_latex_together(rank, term_list, common_linked_factor_list, linked_return_list)
        # if rank == 2:
        #     return _glue_linked_disconnected_latex_together(rank, term_list, common_linked_factor_list, linked_return_list)
        # if rank == 3:
        #     return _glue_linked_disconnected_latex_together(rank, term_list, common_linked_factor_list, linked_return_list)

    # otherwise we simply glue everything together
    else:
        # the maximum number of terms on 1 horizontal line (in latex)
        # change this as needed to fit equations on page
        split_number = 7

        # if the line is so short we don't need to split
        if len(return_list) < split_number*2:
            return f"({' + '.join(return_list)})"

        # make a list of each line
        split_equation_list = []
        for i in range(0, len(return_list) // split_number):
            split_equation_list.append(' + '.join(return_list[i*split_number:(i+1)*split_number]))

        # join the lists with the equation splitting string
        splitting_string = r'\\  &+  % split long equation'
        final_string = f"\n{tab}{splitting_string}\n".join(split_equation_list)

        # and we're done!
        return f"(\n{final_string}\n)"


def _z_joining_with_z_terms(LHS, h, left_z, right_z):
    """Remove terms like `z^1_3 h^2` which require the z^1_3 to join with itself.

    We count the number of annihilation operators `b` and creation operators `d`
    provided for the left Z by the LHS, H and right Z operators.
    We count the number of annihilation operators `b` and creation operators `d`
    provided for the right Z by the LHS, H and left Z operators.

    If a left/right Z operator requires more `b`/`d`s than provided by the other
    operators this implies that left/right Z operator is contracting/joining with itself.
    Theoretically this doesn't exist, and therefore we reject this term.
    """

    available_b_for_left_z = LHS.n + h.n + right_z.n
    available_d_for_left_z = LHS.m + h.m + right_z.m

    available_b_for_right_z = LHS.n + h.n + left_z.n
    available_d_for_right_z = LHS.m + h.m + left_z.m

    required_b_for_left_z = left_z.m
    required_d_for_left_z = left_z.n

    required_b_for_right_z = right_z.m
    required_d_for_right_z = right_z.n

    if (required_b_for_left_z > available_b_for_left_z):
        return True

    if (required_d_for_left_z > available_d_for_left_z):
        return True

    if (required_b_for_right_z > available_b_for_right_z):
        return True

    if (required_d_for_right_z > available_d_for_right_z):
        return True

    return False


def _generate_valid_z_n_operator_permutations(LHS, h, all_z_permutations):
    """ Remove s permutations whose b/d operators don't add up (theoretically can't exist)
    For example LHS^1 h_1 is allowed but not LHS^1 h_1 t^1 because we have 2 d operators but only 1 b operator.
    Additionally we need to make sure the t operators are not joining with themselves.
    This means that the b/d operators from Omega/LHS and h need to be sufficient to balance the b/d's from the t's.
    So LHS^1_1, h_2, t^1_1, t^2 is allowed but not LHS_1, h_1, t^1_1, t^2 because the t^1_1 term has to pair with the
    t^2, or in other words the only sources of d operators are t terms so the b operator from t^1_1 has to pair with
    a d from a t term. This is not allowed.
    """

    valid_permutations = []

    # generate all the possible valid permutations
    for perm in all_z_permutations:
        left_z, right_z = perm

        nof_creation_ops = LHS.m + h.m + left_z.m + right_z.m
        nof_annhiliation_ops = LHS.n + h.n + left_z.n + right_z.n
        cannot_pair_off_b_d_operators = bool(nof_creation_ops != nof_annhiliation_ops)

        # only terms which can pair off all operators are non zero
        if cannot_pair_off_b_d_operators:
            log.debug('Bad Permutation (b and d are not balanced):', LHS, h, left_z, right_z)
            continue

        # Omega/LHS and H need to satisfy all b/d requirements of the z terms
        # z terms can join with each other, but a single z term should not be able to join to itself!!
        if _z_joining_with_z_terms(LHS, h, left_z, right_z):
            log.debug('Bad Permutation (t joins with itself):', LHS, h, left_z, right_z)
            continue

        # Omega/LHS must be able to connect with at least 1 b/d operator from h or a z_term otherwise it 'joins' with itself
        if _omega_joining_with_itself(LHS, h, perm):
            log.debug('Bad Permutation (LHS joins with itself):', LHS, h, left_z, right_z)
            continue

        # h must connect with at least 1 b/d operator from Omega/LHS or a z_term otherwise it 'joins' with itself
        if _h_joining_with_itself(LHS, h, perm):
            log.debug('Bad Permutation (h joins with itself):', LHS, h, left_z, right_z)
            continue

        # record a valid permutation
        valid_permutations.append(perm)
        log.debug(('Good Permutation', LHS, h, perm))

    return valid_permutations


def _generate_all_valid_z_connection_permutations(LHS, h, z_term_list, log_invalid=True):
    """ Generate all possible valid combinations of z terms
    with LHS and h over all index distributions.
    By convention the tuples are (lhs, h, other_z).
    """

    valid_upper_perm_combinations = []
    valid_lower_perm_combinations = []

    m_perms, n_perms = [], []

    # there is are two z's: a left z and a right z
    assert len(z_term_list) == 2
    left_z, right_z = z_term_list

    # generate all possible individual t assignments
    for s_term in z_term_list:
        M, N = s_term.m, s_term.n

        temp_list = []
        for a, b in it.product(range(M+1), repeat=2):
            if a+b <= M:
                temp_list.append((a, b, M-(a+b)))

        m_perms.append(temp_list)

        temp_list = []
        for a, b in it.product(range(N+1), repeat=2):
            if a+b <= N:
                temp_list.append((a, b, N-(a+b)))

        n_perms.append(temp_list)

    # old_print_wrapper(f"{m_perms=}")
    # old_print_wrapper(f"{n_perms=}")

    # validate upper pairing
    combined_m_perms = list(it.product(*m_perms))
    for m_perm in combined_m_perms:

        total_lhs_m = sum([t[0] for t in m_perm])
        total_h_m = sum([t[1] for t in m_perm])
        old_print_wrapper(f"{m_perm=}")
        old_print_wrapper(LHS)
        old_print_wrapper(f"{total_lhs_m=}")
        old_print_wrapper(h)
        old_print_wrapper(f"{total_h_m=}")

        total_lhs_balanced = bool(total_lhs_m <= LHS.n)
        total_h_balanced = bool(total_h_m <= h.n)
        left_z_balanced_right = bool(m_perm[0][2] <= right_z.n)
        right_z_balanced_left = bool(m_perm[1][2] <= left_z.n)

        if total_h_balanced and total_lhs_balanced and left_z_balanced_right and right_z_balanced_left:
            log.debug(f"Valid upper perm:   LHS={total_lhs_m}, zL={m_perm[0]}, h={total_h_m}, zR={m_perm[1]}")
            valid_upper_perm_combinations.append(m_perm)

        elif log_invalid:
            log.debug(
                "Invalid upper perm: "
                f"h={total_h_m} <= {h.n}"
                " or "
                f"LHS={total_lhs_m} <= {LHS.n}"
                " or "
                f"zL={m_perm[0][2]} <= {right_z.n}"
                " or "
                f"zR={m_perm[1][2]} <= {left_z.n}"
                f"{'': >6s}"
                f"{m_perm}"
            )

    # validate lower pairing
    combined_n_perms = list(it.product(*n_perms))
    for n_perm in combined_n_perms:

        total_lhs_n = sum([t[0] for t in n_perm])
        total_h_n = sum([t[1] for t in n_perm])
        old_print_wrapper(f"{n_perm=}")
        old_print_wrapper(LHS)
        old_print_wrapper(f"{total_lhs_n=}")
        old_print_wrapper(h)
        old_print_wrapper(f"{total_h_n=}")

        total_lhs_balanced = bool(total_lhs_n <= LHS.m)
        total_h_balanced = bool(total_h_n <= h.m)
        left_z_balanced_right = bool(n_perm[0][2] <= right_z.m)
        right_z_balanced_left = bool(n_perm[1][2] <= left_z.m)
        old_print_wrapper(f"{total_lhs_balanced=}")
        old_print_wrapper(f"{total_h_balanced=}")
        old_print_wrapper(f"{left_z_balanced_right=}")
        old_print_wrapper(f"{right_z_balanced_left=}")

        if total_h_balanced and total_lhs_balanced and left_z_balanced_right and right_z_balanced_left:
            log.debug(f"Valid lower perm:   LHS={total_lhs_n}, zL={n_perm[0]}, h={total_h_n}, zR={n_perm[1]}")
            valid_lower_perm_combinations.append(n_perm)

        elif log_invalid:
            log.debug(
                "Invalid lower perm: "
                f"h={total_h_n} <= {h.m}"
                " or "
                f"LHS={total_lhs_n} <= {LHS.m}"
                " or "
                f"zL={n_perm[0][2]} <= {right_z.m}"
                " or "
                f"zR={n_perm[1][2]} <= {left_z.m}"
                f"{'': >6s}"
                f"{n_perm}"
            )

    return valid_upper_perm_combinations, valid_lower_perm_combinations


def _generate_all_o_h_z_connection_permutations(LHS, h, valid_z_permutations, found_it_bool=False):
    """ Generate all possible permutations of matching with LHS and h for t_terms """

    annotated_permutations = []  # store output here
    log_conf.setLevelDebug(log)
    old_print_wrapper('-'*30 + 'here' + '-'*30)
    i = 0
    for perm in valid_z_permutations:
        left_z, right_z = perm
        log.debug(f'\n{left_z=}\n{right_z=}\n')
        upper_perms, lower_perms = _generate_all_valid_z_connection_permutations(LHS, h, perm)
        log.debug(f"{upper_perms=}")
        log.debug(f"{lower_perms=}")

        for upper in upper_perms:
            for lower in lower_perms:
                assert len(upper) == len(lower)
                log.debug(f"{upper=}")
                log.debug(f"{lower=}")
                left_z_upper, right_z_upper = upper
                left_z_lower, right_z_lower = lower
                z_left_kwargs, z_right_kwargs = {}, {}

                z_list = []
                # for each Z operator we make a `connected_namedtuple` or a `disconnected_namedtuple`
                if left_z.name is None:
                    assert left_z_upper == left_z_lower == (0, 0, 0)  # make sure this permutation is okay for no z left
                    z_list.append(None)
                else:
                    z_left_kwargs = {
                        'rank': left_z.rank,
                        'm': left_z.m,
                        'm_lhs': left_z_upper[0],
                        'm_h':   left_z_upper[1],
                        'm_r':   left_z_upper[2],
                        'n': left_z.n,
                        'n_lhs': left_z_lower[0],
                        'n_h':   left_z_lower[1],
                        'n_r':   left_z_lower[2],
                    }
                    # if the Z operator is disconnected (meaning no connections to H)
                    if z_left_kwargs['m_h'] == z_left_kwargs['n_h'] == 0:
                        z_list.append(disconnected_z_left_operator_namedtuple(**z_left_kwargs))
                    # if the Z operator is connected (at least 1 connection to H)
                    else:
                        z_list.append(connected_z_left_operator_namedtuple(**z_left_kwargs))

                if right_z.name is None:
                    assert right_z_upper == right_z_lower == (0, 0, 0)  # make sure this permutation is okay for no z right
                    z_list.append(None)
                else:
                    z_right_kwargs = {
                        'rank': right_z.rank,
                        'm': right_z.m,
                        'm_lhs': right_z_upper[0],
                        'm_h':   right_z_upper[1],
                        'm_l':   right_z_upper[2],
                        'n': right_z.n,
                        'n_lhs': right_z_lower[0],
                        'n_h':   right_z_lower[1],
                        'n_l':   right_z_lower[2],
                    }
                    # if the Z operator is disconnected (meaning no connections to H)
                    if z_right_kwargs['m_h'] == z_right_kwargs['n_h'] == 0:
                        z_list.append(disconnected_z_right_operator_namedtuple(**z_right_kwargs))
                    # if the Z operator is connected (at least 1 connection to H)
                    else:
                        z_list.append(connected_z_right_operator_namedtuple(**z_right_kwargs))

                # if we have the ZHZ terms then we need to check that the Z <-> Z
                # contractions are correct
                if (z_left_kwargs != {}) and (z_right_kwargs != {}):
                    # if these contractions are not equal
                    if z_left_kwargs['m_r'] != z_right_kwargs['n_l']:
                        term_string = f"{tab}{LHS}, {h}, {perm}\n{tab}{z_left_kwargs=}\n{tab}{z_right_kwargs=}\n"
                        log.debug(f"Found an invalid term (z_left.m_r != z_right.n_l)\n{term_string}")
                        continue

                    # if these contractions are not equal
                    if z_left_kwargs['n_r'] != z_right_kwargs['m_l']:
                        term_string = f"{tab}{LHS}, {h}, {perm}\n{tab}{z_left_kwargs=}\n{tab}{z_right_kwargs=}\n"
                        log.debug(f"Found an invalid term (z_left.n_r != z_right.m_l)\n{term_string}")
                        continue

                log.debug(f"{z_list}")
                annotated_permutations.append(z_list)

        old_print_wrapper(annotated_permutations)
        old_print_wrapper(f'{i=}\n\n')

        # if i == 3:
        #     sys.exit(0)
        i += 1
    log_conf.setLevelInfo(log)

    return annotated_permutations


def _generate_explicit_z_connections(LHS, h, unique_s_permutations):
    """ Generate new namedtuples for LHS and h explicitly labeling how they connect with each other and t.
    We make `connected_lhs_operator_namedtuple` and `connected_h_z_operator_namedtuple`.
    The output `labeled_permutations` is a list where each element is `[new_LHS, new_h, z_left, z_right]`.
    We also check to make sure each term is valid.
    """

    labeled_permutations = []  # store output here

    for z_list in unique_s_permutations:
        z_left, z_right = z_list
        lhs_kwargs, h_kwargs = {}, {}

        assert len(z_list) == 2

        # bool declarations for readability
        z_left_exists = isinstance(z_left, (connected_z_left_operator_namedtuple, disconnected_z_left_operator_namedtuple))
        z_right_exists = isinstance(z_right, (connected_z_right_operator_namedtuple, disconnected_z_right_operator_namedtuple))

        # sanity checks
        if z_left is None:
            assert z_right_exists
        elif z_right is None:
            assert z_left_exists
        else:
            assert z_right_exists
            assert z_left_exists

        lhs_kwargs = {
            'm_l': z_left.n_lhs if z_left_exists else 0,
            'n_l': z_left.m_lhs if z_left_exists else 0,
            'm_r': z_right.n_lhs if z_right_exists else 0,
            'n_r': z_right.m_lhs if z_right_exists else 0
        }

        old_print_wrapper(z_list)
        old_print_wrapper(z_left)
        old_print_wrapper(z_right)
        old_print_wrapper(f"{lhs_kwargs=}")

        h_kwargs = {
            'm_l': z_left.n_h if z_left_exists else 0,
            'n_l': z_left.m_h if z_left_exists else 0,
            'm_r': z_right.n_h if z_right_exists else 0,
            'n_r': z_right.m_h if z_right_exists else 0
        }

        lhs_kwargs.update({'rank': LHS.m + LHS.n, 'm': LHS.m, 'n': LHS.n})
        h_kwargs.update({'rank': h.m + h.n, 'm': h.m, 'n': h.n})

        # calculate the contractions as the remainder after all other contractions
        lhs_kwargs['m_h'] = lhs_kwargs['m'] - (lhs_kwargs['m_l'] + lhs_kwargs['m_r'])
        lhs_kwargs['n_h'] = lhs_kwargs['n'] - (lhs_kwargs['n_l'] + lhs_kwargs['n_r'])
        h_kwargs['m_lhs'] = h_kwargs['m'] - (h_kwargs['m_l'] + h_kwargs['m_r'])
        h_kwargs['n_lhs'] = h_kwargs['n'] - (h_kwargs['n_l'] + h_kwargs['n_r'])

        # make sure these values are not negative
        assert lhs_kwargs['m_h'] >= 0 and lhs_kwargs['n_h'] >= 0
        assert h_kwargs['m_lhs'] >= 0 and h_kwargs['n_lhs'] >= 0

        # if these contractions are not equal
        if h_kwargs['m_lhs'] != lhs_kwargs['n_h']:
            term_string = f"{tab}{LHS}, {h}, {z_list}\n{tab}{lhs_kwargs=}\n{tab}{h_kwargs=}\n"
            log.debug(f"Found an invalid term (h.m_lhs != LHS.n_h)\n{term_string}")
            continue

        # if these contractions are not equal
        elif h_kwargs['n_lhs'] != lhs_kwargs['m_h']:
            term_string = f"{tab}{LHS}, {h}, {z_list}\n{tab}{lhs_kwargs=}\n{tab}{h_kwargs=}\n"
            log.debug(f"Found an invalid term (h.n_lhs != LHS.m_h)\n{term_string}")
            continue

        new_LHS = connected_lhs_operator_namedtuple(**lhs_kwargs)
        new_h = connected_h_z_operator_namedtuple(**h_kwargs)

        labeled_permutations.append([new_LHS, new_h, z_list])
        for p in labeled_permutations:
            old_print_wrapper('\n\np')
            for x in p:
                old_print_wrapper(x)
            old_print_wrapper('\n\n')
        # sys.exit(0)

    return labeled_permutations


def _filter_out_valid_z_terms(LHS, H, Z_left, Z_right, total_list):
    """ fill up the `term_list` and `total_list` for the Z^n term
    first we find out what term (in the taylor expansion of e^Z) `z_series_term` represents
    set a boolean flag, and wrap the lower order terms in lists so that they have the same
    structure as the z_n case (a list of lists of `general_operator_namedtuple`s)
    """

    zhz_debug = False
    # H*Z terms, straightforward
    if Z_left is None:
        log.info("Z on the right\n")
        z_left_terms = [general_operator_namedtuple(None, 0, 0, 0), ]
        z_right_terms = Z_right.operator_list
        assert isinstance(z_right_terms, list) and isinstance(z_right_terms[0], general_operator_namedtuple)

    # Z*H terms, straightforward
    elif Z_right is None:
        log.info("Z on the left\n")
        z_left_terms = Z_left.operator_list
        z_right_terms = [general_operator_namedtuple(None, 0, 0, 0), ]
        # z_right_terms = Z_left.operator_list
        assert isinstance(z_left_terms, list) and isinstance(z_left_terms[0], general_operator_namedtuple)

        # valid_lower_perms = [list(it.dropwhile(lambda y: y == 0, x)) for x in unique_permutations if (maximum >= sum(x))]
        # valid_lower_perms[valid_lower_perms.index([])] = [0]

    # Z*H*Z terms, most complicated
    else:
        zhz_debug = True
        log.info("Z on both sides\n")
        z_left_terms = Z_left.operator_list
        z_right_terms = Z_right.operator_list
        assert isinstance(z_right_terms, list) and isinstance(z_right_terms[0], general_operator_namedtuple)
        assert isinstance(z_left_terms, list) and isinstance(z_left_terms[0], general_operator_namedtuple)

    all_z_permutations = [(z_left, z_right) for z_left, z_right in it.product(z_left_terms, z_right_terms)]

    if zhz_debug or False:  # debug prints
        # print all possible pairings
        for a in all_z_permutations:
            old_print_wrapper('Z PAIRING', a)

        # we may not need the unique permutations... unclear at this moment
        if zhz_debug or False:
            unique_z_permutation_list = sorted(list(set(all_z_permutations)))
            for a in unique_z_permutation_list:
                old_print_wrapper('Z TERM1', a)

    # next we process the z operators inside z_term_list
    for h in H.operator_list:

        # valid pairings of s operators given a specific `LHS` and `h`
        valid_permutations = _generate_valid_z_n_operator_permutations(LHS, h, all_z_permutations)

        # if no valid operators continue to the next h
        if valid_permutations == []:
            continue

        if zhz_debug or False:  # debug prints
            for a in valid_permutations:
                old_print_wrapper('VALID TERM', LHS, h, a)

        # we need to generate all possible combinations of each z with the LHS and h operators and the other z
        s_connection_permutations = _generate_all_o_h_z_connection_permutations(LHS, h, valid_permutations)

        if zhz_debug or False:  # debug prints
            for s in s_connection_permutations:
                old_print_wrapper('CONNECTED TERMS', LHS, h, s)

        # NOTE - at the moment I don't believe the Z term logic generates any permutations
        # (remove all duplicate permutations)
        # unique_s_permutations = _remove_duplicate_z_permutations(s_connection_permutations)
        unique_s_permutations = s_connection_permutations

        if True:  # debug prints
            for s in unique_s_permutations:
                old_print_wrapper('UNIQUE TERMS', LHS, h, s)

        # generate all the explicit connections
        # this also removes all invalid terms
        labeled_permutations = _generate_explicit_z_connections(LHS, h, unique_s_permutations)

        old_print_wrapper('labeled', labeled_permutations)

        # we record
        for term in labeled_permutations:
            log.debug(f"{term=}")
            if term[2] != set():
                # if it is not an empty set
                total_list.append(term)
            else:
                old_print_wrapper('exit?')
                sys.exit(0)

    return


# --------------- assigning of upper/lower latex indices ------------------------- #
def _build_left_z_term(z_left, h, color=True):
    """ Builds latex code for labeling a `connected_z_left_operator_namedtuple`.

    The `condense_offset` is an optional argument which is needed when creating latex code
    for linked disconnected terms in a condensed format.
    """
    if z_left.rank == 0:
        return f"{bold_z_latex}_0"

    upper_indices, lower_indices = '', ''

    # do the upper indices first
    if z_left.m > 0:
        # contract with h
        upper_indices += z_summation_indices[0:z_left.m_h]

        if not color:
            # contract with right z
            offset = z_left.m_h
            upper_indices += z_summation_indices[offset:offset + z_left.m_r]

            # pair with left hand side (LHS)
            upper_indices += z_unlinked_indices[0:z_left.m_lhs]
        else:
            # contract with right z
            offset = z_left.m_h
            upper_indices += r'\blue{' + z_summation_indices[offset:offset + z_left.m_r] + '}'

            # pair with left hand side (LHS)
            upper_indices += r'\red{' + z_unlinked_indices[0:z_left.m_lhs] + '}'

    # now do the lower indices
    if z_left.n > 0:
        # contract with h
        offset = z_left.m_h + z_left.m_r
        lower_indices += z_summation_indices[offset:offset + z_left.n_h]

        if not color:
            # contract with right z
            offset += z_left.n_h
            lower_indices += z_summation_indices[offset:offset + z_left.n_r]

            # pair with left hand side (LHS)
            lhs_offset = z_left.m_lhs
            lower_indices += z_unlinked_indices[lhs_offset:lhs_offset + z_left.n_lhs]
        else:
            # contract with right z
            offset += z_left.n_h
            lower_indices += r'\blue{' + z_summation_indices[offset:offset + z_left.n_r] + '}'

            # pair with left hand side (LHS)
            lhs_offset = z_left.m_lhs
            lower_indices += r'\red{' + z_unlinked_indices[lhs_offset:lhs_offset + z_left.n_lhs] + '}'

    return f"{bold_z_latex}^{{{upper_indices}}}_{{{lower_indices}}}"


def _build_hz_term_latex_labels(h, offset_dict, color=True):
    """ Builds latex code for labeling a `connected_h_operator_namedtuple`."""

    if h.rank == 0:
        return f"{bold_h_latex}_0"

    upper_indices, lower_indices = '', ''

    # subscript indices
    if h.n > 0:
        # contract with left z
        lower_indices += r'\blue{' + z_summation_indices[0:h.n_l] + '}'

        # contract with right z
        s = offset_dict['summation_index']
        lower_indices += r'\blue{' + z_summation_indices[s:s + h.n_r] + '}'
        offset_dict['summation_index'] += h.n_r

        # pair with left hand side (LHS)
        u = offset_dict['unlinked_index']
        lower_indices += r'\red{' + z_unlinked_indices[u:u + h.n_lhs] + '}'
        offset_dict['unlinked_index'] += h.n_lhs

    # superscript indices
    if h.m > 0:
        # contract with left z
        a = offset_dict['left_upper']
        upper_indices += r'\blue{' + z_summation_indices[a:a + h.m_l] + '}'

        # contract with right z
        s = offset_dict['summation_index']
        upper_indices += r'\blue{' + z_summation_indices[s:s + h.m_r] + '}'
        offset_dict['summation_index'] += h.m_r

        # pair with left hand side (LHS)
        u = offset_dict['unlinked_index']
        upper_indices += r'\red{' + z_unlinked_indices[u:u + h.m_lhs] + '}'
        offset_dict['unlinked_index'] += h.m_lhs

    return f"{bold_h_latex}^{{{upper_indices}}}_{{{lower_indices}}}"


def _build_right_z_term(h, z_right, offset_dict, color=True):
    """ Builds latex code for labeling a `connected_z_right_operator_namedtuple`.

    The `condense_offset` is an optional argument which is needed when creating latex code
    for linked disconnected terms in a condensed format.
    """
    if z_right.rank == 0:
        return f"{bold_z_latex}_0"

    a, b = 0, 0
    upper_indices, lower_indices = '', ''

    # subscript indices
    if z_right.n > 0:
        # contract with left z
        a = offset_dict['left_lower']
        lower_indices += r'\magenta{' + z_summation_indices[a:a + z_right.n_l] + '}'

        # contract with h
        b = offset_dict['h_lower']
        lower_indices += r'\blue{' + z_summation_indices[b:b + z_right.n_h] + '}'

        # pair with left hand side (LHS)
        u = offset_dict['unlinked_index']
        lower_indices += r'\red{' + z_unlinked_indices[u:u + z_right.n_lhs] + '}'
        offset_dict['unlinked_index'] += z_right.n_lhs

    # superscript indices
    if z_right.m > 0:
        # contract with left z
        a = offset_dict['left_upper']
        upper_indices += r'\magenta{' + z_summation_indices[a:a + z_right.m_l] + '}'

        # contract with h
        b = offset_dict['h_upper']
        upper_indices += r'\blue{' + z_summation_indices[b:b + z_right.m_h] + '}'

        # pair with left hand side (LHS)
        u = offset_dict['unlinked_index']
        upper_indices += r'\red{' + z_unlinked_indices[u:u + z_right.m_lhs] + '}'
        offset_dict['unlinked_index'] += z_right.m_lhs

    return f"{bold_z_latex}^{{{upper_indices}}}_{{{lower_indices}}}"


# -------------------------------------------------------------------------------- #
def _build_z_latex_prefactor(h, t_list, simplify_flag=True):
    """Attempt to return latex code representing appropriate prefactor term.

    All prefactors begin with 1/n! where n is the number of t amplitudes in given term.
    This comes from the prefactors of the taylor expansion of e^S = 1 + S^1 + S^2 ....
    We also need to account for the prefactors of the Hamiltonian as defined in the theory.
    There are also permutations of indices to take into account.
    Overall the entire affair is quite complicated.

    In general the rules are:
        1 - 1/n! where n is the number of t terms (taylor series)
        1 - All t terms are balanced/neutral (they contribute m!n!/m!n! = 1)
        1 - h contributes x!/m!n! where x is the number of UNIQUE t terms that h contracts with

    A single h with no t list is a special case where the prefactor is always 1.
    """

    # special case, single h
    if len(t_list) == 1 and t_list[0] == disconnected_namedtuple(0, 0, 0, 0):
        return ''

    numerator_list = []
    denominator_list = []

    # account for prefactor from Hamiltonian term `h`
    connected_ts = [t for t in t_list if t.m_h > 0 or t.n_h > 0]
    x = len(set(connected_ts))

    debug_flag = bool(
        h.n == 2
        and h.m == 0
        and len(t_list) == 2
        and t_list[0].m_h == 1
        and t_list[0].m_o == 1
        and t_list[1].m_h == 1
        and t_list[1].m_o == 1
    )

    if False and debug_flag:
        old_print_wrapper('\n\n\nzzzzzzzzz')
        old_print_wrapper(connected_ts)
        old_print_wrapper(set(connected_ts))
        old_print_wrapper(len(set(connected_ts)))
        old_print_wrapper('zzzzzzzzz\n\n\n')

    if x > 1:
        numerator_list.append(f'{x}!')
    if h.m > 1:
        denominator_list.append(f'{h.m}!')
    if h.n > 1:
        denominator_list.append(f'{h.n}!')

    # account for the number of permutations of all t-amplitudes
    if len(t_list) > 1:
        denominator_list.append(f'{len(t_list)}!')

    # simplify
    if simplify_flag:
        numerator_list, denominator_list = _simplify_full_cc_python_prefactor(numerator_list, denominator_list)

    # glue the numerator and denominator together
    numerator = '1' if (numerator_list == []) else f"{''.join(numerator_list)}"
    denominator = '1' if (denominator_list == []) else f"{''.join(denominator_list)}"

    if numerator == '1' and denominator == '1':
        return ''
    else:
        return f"\\frac{{{numerator}}}{{{denominator}}}"


def _f_zL_h_contributions(z_left, h):
    """ x """
    if isinstance(z_left, disconnected_z_left_operator_namedtuple):
        assert 0 == h.n_l
        return 0
    else:
        assert z_left.m_h == h.n_l
        return z_left.m_h


def _fbar_zL_h_contributions(z_left, h):
    """ x """
    if isinstance(z_left, disconnected_z_left_operator_namedtuple):
        assert 0 == h.m_l
        return 0
    else:
        assert z_left.n_h == h.m_l
        return z_left.n_h


def _f_h_zR_contributions(h, z_right):
    """ x """
    if isinstance(z_right, disconnected_z_right_operator_namedtuple):
        assert h.m_r == 0
        return 0
    else:
        assert h.m_r == z_right.n_h
        return z_right.n_h


def _fbar_h_zR_contributions(h, z_right):
    """ x """
    if isinstance(z_right, disconnected_z_right_operator_namedtuple):
        assert h.n_r == 0
        return 0
    else:
        assert h.n_r == z_right.m_h
        return z_right.m_h


def _f_zL_zR_contributions(z_left, z_right):
    """ x """
    assert z_left.m_r == z_right.n_l
    return z_left.m_r


def _fbar_zL_zR_contributions(z_left, z_right):
    """ x """
    assert z_left.n_r == z_right.m_l
    return z_left.n_r


# -------------------------------------------------------------------------------- #
def _prepare_second_z_latex(term_list, split_width=7, remove_f_terms=False, print_prefactors=False):
    """Return the latex commands to write the provided terms.

    The `split_width` is the maximum number of terms on 1 horizontal line (in latex) and should
    be changed as needed to fit equations on page.
    If `remove_f_terms` is true terms then terms where nof_fs > 0 will are not written to latex.
    If `print_prefactors` is true then we add the prefactor string generated by `_build_z_latex_prefactor`.
    """

    return_list = []  # store output here

    # prepare all the latex strings
    for term in term_list:
        term_string = ''

        # extract elements of list `term`, for now we don't use `LHS` or `z_right`
        LHS, h, z_left, z_right = term[0], term[1], *term[2]

        # if needed add f prefactors
        nof_fs = _f_zL_h_contributions(z_left, h)
        if remove_f_terms and (nof_fs > 0):
            continue
        if nof_fs > 0:
            term_string += "f" if (nof_fs == 1) else f"f^{{{nof_fs}}}"

        # if needed add fbar prefactors
        nof_fbars = _fbar_zL_h_contributions(z_left, h)
        if nof_fbars > 0:
            term_string += "\\bar{f}" if (nof_fbars == 1) else f"\\bar{{f}}^{{{nof_fbars}}}"

        # add any prefactors if they exist
        if print_prefactors:
            raise Exception("prefactor code for z stuff is not done")
            term_string += _build_z_latex_prefactor(h, z_left)

        # prepare the z terms
        h_offset_dict = {
            'left_upper': z_left.m - z_left.m_lhs,
            'summation_index': z_left.rank - z_left.m_lhs - z_left.n_lhs,
            'unlinked_index': z_left.m_lhs + z_left.n_lhs
        }

        left_z = _build_left_z_term(z_left, h)
        h_term = _build_hz_term_latex_labels(h, h_offset_dict)
        right_z = ''

        # build the latex code representing this term in the sum
        term_string += left_z + h_term + right_z

        # store the result
        return_list.append(term_string)

    # if the line is so short we don't need to split
    if len(return_list) < split_width*2:
        return f"({' + '.join(return_list)})"

    split_equation_list = []
    for i in range(0, len(return_list) // split_width):
        split_equation_list.append(' + '.join(return_list[i*split_width:(i+1)*split_width]))

    # make sure we pickup the last few terms
    last_few_terms = (len(return_list) % split_width)-split_width+1
    split_equation_list.append(' + '.join(return_list[last_few_terms:]))

    # join the lists with the equation splitting string
    splitting_string = r'\\  &+  % split long equation'
    final_string = f"\n{tab}{splitting_string}\n".join(split_equation_list)

    # and we're done!
    return f"(\n{final_string}\n)"


def _prepare_third_z_latex(term_list, split_width=7, remove_f_terms=False, print_prefactors=False):
    """Return the latex commands to write the provided terms.

    The `split_width` is the maximum number of terms on 1 horizontal line (in latex) and should
    be changed as needed to fit equations on page.
    If `remove_f_terms` is true terms then terms where nof_fs > 0 will are not written to latex.
    If `print_prefactors` is true then we add the prefactor string generated by `_build_z_latex_prefactor`.
    """

    return_list = []  # store output here

    # prepare all the latex strings
    for term in term_list:
        term_string = ''

        # old_print_wrapper("TERM", term)
        # extract elements of list `term`
        LHS, h, z_left, z_right = term[0], term[1], *term[2]

        # if needed add f prefactors
        nof_fs = _f_h_zR_contributions(h, z_right)
        if remove_f_terms and (nof_fs > 0):
            continue
        if nof_fs > 0:
            term_string += "f" if (nof_fs == 1) else f"f^{{{nof_fs}}}"

        # if needed add fbar prefactors
        nof_fbars = _fbar_h_zR_contributions(h, z_right)
        if nof_fbars > 0:
            term_string += "\\bar{f}" if (nof_fbars == 1) else f"\\bar{{f}}^{{{nof_fbars}}}"

        # add any prefactors if they exist
        if print_prefactors:
            raise Exception("prefactor code for z stuff is not done")
            term_string += _build_z_latex_prefactor(h, z_right)

        # prepare the z terms
        h_offset_dict = {
            'left_upper': 0,
            'summation_index': 0,
            'unlinked_index': 0
        }

        right_z_offset_dict = {
            'left_lower': 0,
            'left_upper': 0,
            'h_lower': h.n_r,
            'h_upper': 0,
            'unlinked_index': h.m_lhs + h.n_lhs
        }

        left_z = ''
        h_term = _build_hz_term_latex_labels(h, h_offset_dict)
        right_z = _build_right_z_term(h, z_right, right_z_offset_dict)

        # build the latex code representing this term in the sum
        term_string += left_z + h_term + right_z

        # store the result
        return_list.append(term_string)

    # if the line is so short we don't need to split
    if len(return_list) < split_width*2:
        return f"({' + '.join(return_list)})"

    split_equation_list = []
    for i in range(0, len(return_list) // split_width):
        split_equation_list.append(' + '.join(return_list[i*split_width:(i+1)*split_width]))

    # make sure we pickup the last few terms
    last_few_terms = (len(return_list) % split_width)-split_width+1
    split_equation_list.append(' + '.join(return_list[last_few_terms:]))

    # join the lists with the equation splitting string
    splitting_string = r'\\  &+  % split long equation'
    final_string = f"\n{tab}{splitting_string}\n".join(split_equation_list)

    # and we're done!
    return f"(\n{final_string}\n"


def _prepare_fourth_z_latex(term_list, split_width=7, remove_f_terms=False, print_prefactors=False):
    """Return the latex commands to write the provided terms.

    The `split_width` is the maximum number of terms on 1 horizontal line (in latex) and should
    be changed as needed to fit equations on page.
    If `remove_f_terms` is true terms then terms where nof_fs > 0 will are not written to latex.
    If `print_prefactors` is true then we add the prefactor string generated by `_build_z_latex_prefactor`.
    """

    return_list = []  # store output here

    # prepare all the latex strings
    for term in term_list:
        term_string = ''

        # old_print_wrapper("TERM", term)
        # extract elements of list `term`
        LHS, h, z_left, z_right = term[0], term[1], *term[2]

        assert z_left is not None
        assert z_right is not None

        # if needed add f prefactors
        nof_fs = _f_zL_h_contributions(z_left, h)
        nof_fs += _f_h_zR_contributions(h, z_right)
        nof_fs += _f_zL_zR_contributions(z_left, z_right)
        if remove_f_terms and (nof_fs > 0):
            continue
        if nof_fs > 0:
            term_string += "f" if (nof_fs == 1) else f"f^{{{nof_fs}}}"

        # if needed add fbar prefactors
        nof_fbars = _fbar_zL_h_contributions(z_left, h)
        nof_fbars += _fbar_h_zR_contributions(h, z_right)
        nof_fbars += _fbar_zL_zR_contributions(z_left, z_right)
        if nof_fbars > 0:
            term_string += "\\bar{f}" if (nof_fbars == 1) else f"\\bar{{f}}^{{{nof_fbars}}}"

        # add any prefactors if they exist
        if print_prefactors:
            raise Exception("prefactor code for z stuff is not done")
            term_string += _build_z_latex_prefactor(h, z_left)

        # prepare the z terms
        h_offset_dict = {
            'left_upper': z_left.m - z_left.m_lhs,
            'summation_index': z_left.rank - z_left.m_lhs - z_left.n_lhs,
            'unlinked_index': z_left.m_lhs + z_left.n_lhs
        }

        right_z_offset_dict = {
            'left_lower': z_left.m_h,
            'left_upper': h_offset_dict['left_upper'] + z_left.n_h,
            'h_lower': h_offset_dict['summation_index'] + h.n_r,
            'h_upper': h_offset_dict['summation_index'],
            'unlinked_index': h_offset_dict['unlinked_index'] + h.m_lhs + h.n_lhs
        }

        left_z = _build_left_z_term(z_left, h)
        h_term = _build_hz_term_latex_labels(h, h_offset_dict)
        right_z = _build_right_z_term(h, z_right, right_z_offset_dict)

        # build the latex code representing this term in the sum
        term_string += left_z + h_term + right_z

        disconnected_term = bool(
            isinstance(z_left, disconnected_z_left_operator_namedtuple)
            or isinstance(z_right, disconnected_z_right_operator_namedtuple)
        )

        if disconnected_term:
            term_string = r'\disconnected{' + term_string + r'}'

        # store the result
        return_list.append(term_string)

    # if the line is so short we don't need to split
    if len(return_list) < split_width*2:
        return f"({' + '.join(return_list)})"

    split_equation_list = []
    for i in range(0, len(return_list) // split_width):
        split_equation_list.append(' + '.join(return_list[i*split_width:(i+1)*split_width]))

    # make sure we pickup the last few terms
    last_few_terms = (len(return_list) % split_width)-split_width+1
    split_equation_list.append(' + '.join(return_list[last_few_terms:]))

    # join the lists with the equation splitting string
    splitting_string = r'\\  &+  % split long equation'
    final_string = f"\n{tab}{splitting_string}\n".join(split_equation_list)
    # if LHS.n == 2:
    #     old_print_wrapper(len(return_list))
    #     sys.exit(0)

    # and we're done!
    return f"(\n{final_string}\n)"


# ------------------------------------------------------------------------ #
def _build_first_z_term(LHS):
    if LHS.rank == 0:
        return f"{bold_h_latex}" + r'_{0,xb}(1-\delta_{x\gamma})'
    else:
        first_term = f"{bold_h_latex}"

        # do the upper indices first
        upper_indices = summation_indices[0:LHS.n]
        first_term += f"^{{{upper_indices}}}"

        h_offset = LHS.n

        # now do the lower indices
        lower_indices = summation_indices[h_offset:h_offset+LHS.m]
        first_term += f"_{{{lower_indices}}}"

        # label the off diagonality
        first_term += r'(1-\delta_{x\gamma})'

        return first_term


def _build_second_z_term(LHS, H, Z, remove_f_terms=False):
    """
    This one basically needs to be like the t term stuff EXCEPT:
        - there is a single z term
        - it is always on the left side
        - always bond to projection operator in same dimension (^i ^i)
    """
    valid_term_list = []

    # generate all valid combinations
    _filter_out_valid_z_terms(LHS, H, Z, None, valid_term_list)

    if valid_term_list == []:
        return ""

    return _prepare_second_z_latex(valid_term_list, remove_f_terms=remove_f_terms)


def _build_third_z_term(LHS, H, Z, remove_f_terms=False):
    """
    This one basically needs to be like the t term stuff EXCEPT:
        - there is a single z term
        - it is always on the right side
        - always bond to projection operator in opposite dimension (^i _i)
    """
    valid_term_list = []

    # generate all valid combinations
    _filter_out_valid_z_terms(LHS, H, None, Z, valid_term_list)

    if valid_term_list == []:
        return ""

    return _prepare_third_z_latex(valid_term_list, remove_f_terms=remove_f_terms)


def _build_fourth_z_term(LHS, H, Z, remove_f_terms=False):
    """
    This one basically needs to be like the t term stuff EXCEPT:
        - there is always two z terms
        - one on each side
        - all combinations are considered here
    """
    valid_term_list = []

    # generate all valid combinations
    _filter_out_valid_z_terms(LHS, H, Z, Z, valid_term_list)

    if valid_term_list == []:
        return ""

    return _prepare_fourth_z_latex(valid_term_list, remove_f_terms=remove_f_terms)


def _build_fifth_z_term(LHS, Z):
    """ This is all permutations of the two operators:
    `dt/dtau` and `z` at over different numbers of creation operators.
    """

    term_list = []

    for z_m in range(LHS.n+1):

        # build dt/dtau first

        if (LHS.m == 0) and (LHS.n-z_m == 0):
            t_term_latex = r'\hat{' + bold_t_latex + r'}_{0, \gamma}'
            dv_t_term_latex = f'\\dv{{{t_term_latex}}}{{\\tau}}'
        else:
            t_upper_label = f"{summation_indices[0:LHS.n-z_m]}" if LHS.n - z_m > 0 else ""
            t_lower_label = f"{summation_indices[LHS.n:LHS.n+LHS.m]}" if LHS.m > 0 else ""

            t_term_latex = f"\\hat{{{bold_t_latex}}}"
            t_term_latex += f"^{{{t_upper_label}}}" if t_upper_label != "" else ""
            t_term_latex += f"_{{{t_lower_label}, \\gamma}}" if t_lower_label != "" else r'_{\gamma}'

            # the t term we are taking the derivative of
            dv_t_term_latex = f'\\dv{{{t_term_latex}}}{{\\tau}}'

        # build z second
        if z_m == 0:
            z_term_latex = f"\\hat{{{bold_z_latex}}}_{{0, \\gamma}}"
        else:
            z_upper_label = f"{summation_indices[LHS.rank-z_m:LHS.rank]}"
            z_term_latex = f"\\hat{{{bold_z_latex}}}^{{{z_upper_label}}}_{{\\gamma}}"

        # glue them together
        term_list.append(dv_t_term_latex + r'\,' + z_term_latex)

    return " + ".join(term_list)


# ------------------------------------------------------------------------ #
def _generate_z_symmetric_latex_equations(LHS, H, Z, only_ground_state=True, remove_f_terms=False):
    """Return a string containing latex code to be placed into a .tex file.
    For a given set of input arguments: (`LHS`, `H`, `Z`) we generate
    all possible and valid CC terms. Note that:
        - `LHS` is an `LHS_namedtuple` object
        - `H` is a `hamiltonian_namedtuple` object
        - `Z` is a `z_operator_namedtuple` object

    One possible input could be:
        - `LHS` is the creation operator d
        - `H` is a Hamiltonian of rank two
        - `Z` is the Z operator
    """

    """ First we want to generate a list of valid terms.
    We start with the list of lists `s_taylor_expansion` which is processed by `_filter_out_valid_s_terms`.
    This function identifies valid pairings AND places those pairings in the `valid_term_list`.
    Specifically we replace the `general_operator_namedtuple`s with `connected_namedtuple`s and/or
    `disconnected_namedtuple`s.
    """
    return_string = ""

    # the first H term
    return_string += _build_first_z_term(LHS)

    # the second (subtraction) term
    if not only_ground_state:  # If we are acting on the vaccum state then these terms don't exist
        raise Exception('The excited state ZT terms for the 5th ansatz has not been properly implemented')
        return_string += r'\\&-\Big(' + _build_second_z_term(LHS, H, Z, remove_f_terms) + r'\Big)'

    # the third (addition) term
    return_string += r'\\&+\sum\Big(' + _build_third_z_term(LHS, H, Z, remove_f_terms) + r'\Big)(1-\delta_{cb})'

    # the fourth (subtraction) term
    if not only_ground_state:  # If we are acting on the vaccum state then these terms don't exist
        raise Exception('The excited state ZT terms for the 5th ansatz has not been properly implemented')
        return_string += r'\\&-\sum\Big(' + _build_fourth_z_term(LHS, H, Z, remove_f_terms) + r'\Big)(1-\delta_{db})'

    if only_ground_state:  # If we are acting on the vacuum state then we add these extra terms
        temporary_string = r"\text{all permutations of }\dv{\hat{t}_{\gamma}}{\tau}\hat{z}"
        return_string += r'\\&-i\sum\Big(' + _build_fifth_z_term(LHS, Z) + r'\Big)'

    # remove all empty ^{}/_{} terms that are no longer needed
    return return_string.replace("^{}", "").replace("_{}", "")
    # return r'%'


def generate_z_t_symmetric_latex(truncations, only_ground_state=True, remove_f_terms=False, path="./generated_latex.tex"):
    """Generates and saves to a file the latex equations for full CC expansion."""

    assert len(truncations) == 4, "truncations argument needs to be tuple of four integers!!"
    maximum_h_rank, maximum_cc_rank, s_taylor_max_order, omega_max_order = truncations

    master_omega = generate_omega_operator(maximum_cc_rank, omega_max_order)
    raw_H = generate_full_cc_hamiltonian_operator(maximum_h_rank)
    Z = generate_z_operator(maximum_cc_rank, only_ground_state)

    # aug 4th songhao says this is because of theory
    pruned_list = [term for term in raw_H.operator_list if (term.rank < 3) or (term.m == 0)]
    H = hamiltonian_namedtuple(raw_H.maximum_rank, pruned_list)

    latex_code = ""  # store result in here

    rank_name_list = [
        "0 order", "LINEAR", "QUADRATIC", "CUBIC", "QUARTIC", "QUINTIC", "SEXTIC", "SEPTIC", "OCTIC"
    ]

    for i, omega_term in enumerate(master_omega.operator_list):

        # for debugging purposes
        # if you only want to generate the linear terms for example; change the False to True
        if False and omega_term.rank not in [1, ]:
            continue

        # for the new ansatz v5 we only print the annihilation operator projections
        if omega_term.m > 0:
            continue

        rank_name = rank_name_list[omega_term.rank]

        if omega_term.rank > master_omega.operator_list[i-1].rank:
            # latex_code += '\\newpage\n' if omega_term.rank >= 2 else ''
            latex_code += f'\\paragraph{{{rank_name.capitalize()} Equations}}\n\n'
        else:
            latex_code += r'\vspace{2cm}'

        def _generate_t_lhs(omega):

            upper_label = summation_indices[0:omega.n]
            lower_label = summation_indices[omega.n:omega.rank]
            lower_label += r'\gamma' if lower_label == "" else r',\gamma'

            t_term_latex = f"\\hat{{{bold_t_latex}}}^{{{upper_label}}}_{{{lower_label}}}"

            # the t term we are taking the derivative of
            derivative_latex = rf'i\dv{{{t_term_latex}}}{{\tau}}'

            return derivative_latex

        def _wrap_t_latex(omega, lhs):
            """ Latex commands to wrap around the generated terms `lhs`
            so that the *.tex file compiles correctly.
            """
            omega_string = ""
            for i, char in enumerate(omega.name):
                if char == "d":
                    omega_string += f'\\up{{{summation_indices[i]}}}'
                elif char == "b":
                    omega_string += f'\\down{{{summation_indices[i]}}}'

            omega_string = "1" if omega_string == "" else omega_string

            g_upper_label = f"{summation_indices[0:omega.n]}" if omega.n > 0 else ""
            g_lower_label = f"{summation_indices[omega.n:omega.rank]}" if omega.m > 0 else ""
            g_lower_label += r'\gamma\gamma' if g_lower_label == "" else r', \gamma\gamma'
            g_string = f"\\hat{{G}}^{{{g_upper_label}}}_{{{g_lower_label}}}"

            return (
                '\\begin{equation}\n'
                f'{tab}\\hat{{\\Omega}} = {omega_string}\n'
                r'\qquad\qquad'
                f"\n{tab}{lhs} = {g_string}"
                "\n"
                r'\end{equation}'
                '\n\n'
            )

        # generate latex for t terms
        if True:
            # generate the i(dt/dtau + t*epsilon) latex
            lhs_string = _generate_t_lhs(omega_term)

            # where we do all the work of generating the latex
            # header for the sub section
            latex_code += '%\n%\n%\n%\n%\n\n'
            latex_code += _wrap_t_latex(omega_term, lhs_string)

        def _wrap_z_align_environment(lhs, eqns):
            """ Latex commands to wrap around the generated terms `lhs` and `eqns`
            so that the *.tex file compiles correctly.
            """
            string = (
                '\\begin{align}\\begin{split}\n'
                r'LHS &='
                '\n'
                f"{tab}{lhs}\n"
                r'\\ RHS &='
                '\n%\n%\n'
                f'{eqns}\n'
                r'\end{split}\end{align}'
                '\n\n'
            )
            return string

        def _generate_z_lhs(omega):
            """ quick fix """
            upper_label = summation_indices[0:omega.rank]
            z_term_latex = f"\\hat{{{bold_z_latex}}}^{{{upper_label}}}_{{\\gamma}}"

            # the t term we are taking the derivative of
            derivative_latex = rf'i\Big(\dv{{{z_term_latex}}}{{\tau}}\Big)'
            return derivative_latex

        # generate latex for z terms
        # the omega term is the LHS
        if True:
            # generate the i(dt/dtau + t*epsilon) latex
            lhs_string = _generate_z_lhs(omega_term)

            # where we do all the work of generating the latex
            equations_string = _generate_z_symmetric_latex_equations(omega_term, H, Z, only_ground_state, remove_f_terms)

            # header for the sub section
            latex_code += '%\n%\n%\n%\n%\n\n'
            latex_code += _wrap_z_align_environment(lhs_string, equations_string)

    # write the latex to file
    if only_ground_state:
        # use the predefined header in `reference_latex_headers.py`
        header = headers.ground_state_z_t_symmetric_latex_header
    else:
        # use the predefined header in `reference_latex_headers.py`
        header = headers.full_z_t_symmetric_latex_header

    header += '\\textbf{Note that all terms with a $f$ prefactor have been removed}\n' if remove_f_terms else ''

    # write the new header with latex code attached
    with open(path, 'w') as fp:
        fp.write(header + latex_code + r'\end{document}')

    return


# ----------------------------------------------------------------------------------------------- #
# -------------------------  GENERATING FULL Omega e^T Z H Z symmetrized LATEX  ----------------------------- #
# ----------------------------------------------------------------------------------------------- #

eT_operator_namedtuple = namedtuple('eT_operator', ['maximum_rank', 'operator_list'])


# -------------------------------------------------------------------------------- #
def generate_eT_operator(maximum_eT_rank=2):
    """Return an `eT_operator_namedtuple` whose attributes are determined by `maximum_cc_rank`.

    The `operator_list` contains all permutations of (`m`,`n`) for `m` & `n` in `range(maximum_cc_rank + 1)`.
    The name is a string of the chars `d` and `b` according to `m` and `n` respectively.
    `m` is associated with creation operators (d) and `n` is associated with annihilation operators (b).
    """
    return_list = []

    for m in [0]:                                    # m is the upper label (creation operators)
        for n in range(0, maximum_eT_rank + 1 - m):  # n is the lower label (annihilation operators)

            # we account for the zero order S operator in `_generate_s_taylor_expansion`
            if m == n == 0:
                continue

            name = "s"
            name += f"^{m}" if m > 0 else ""
            name += f"_{n}" if n > 0 else ""

            return_list.append(general_operator_namedtuple(name, m+n, m, n))

    return eT_operator_namedtuple(maximum_cc_rank, return_list)


def generate_eT_taylor_expansion(maximum_eT_rank=2, eT_taylor_max_order=3):
    """Return a list of lists of `eT_operator_namedtuple`s.

    Expanding e^{T} by Taylor series gives you 1 + T + T^2 + T^3 ... etc.
    Each of the terms in that sum (1, T, T^2, ...) is represented by a list inside the returned list.
    So `eT_taylor_expansion[0]` is the 1 term and has a single `eT_operator_namedtuple`.
    Then `eT_taylor_expansion[1]` represents the `T` term and is a list of `eT_operator_namedtuple`s
    generated by `generate_eT_operator` from eT_1 all the way to eT_n as determined by `maximum_eT_rank`.
    The third list `eT_taylor_expansion[2]` is all the T^2 terms, and so on.

    For terms T^2 and higher we compute all possible products, including non unique ones.
    This means for T^3 we will compute T^1 * T^1 * T^1 three times, however later on we will remove the duplicate terms.
    The duplicate terms are used to account for multiple possible index label orders.

    Only creation operators can act on a system in the ground state, => `n` is required to be 0.
    """

    # The eT_operator_namedtuple's
    T = generate_eT_operator(maximum_eT_rank)

    # create the list
    eT_taylor_expansion = [None, ]*(eT_taylor_max_order+1)

    eT_taylor_expansion[0] = general_operator_namedtuple("1", 0, 0, 0)  # 1 term
    eT_taylor_expansion[1] = T.operator_list                            # T term

    """ We compute all combinations including non unique ones ON PURPOSE!!
    The products of T operators do not have indices mapping them to omega and H.
    Therefore T^2 * T^1 === T^1 * T^2.
    Later when we add indices, the non unique combinations will become unique,
    due to the nature of the process of assigning the indices.
    For example, if omega = b and h_{ij}:
     - T^2 * T^1 can become T^{ij} * T^{z}, which would be a disconnected term.
     - T^1 * T^2 can become T^{i} * T^{jz}, which would be a connected term.
    """
    for n in range(2, eT_taylor_max_order+1):
        eT_taylor_expansion[n] = [list(tup) for tup in it.product(T.operator_list, repeat=n)]

    return eT_taylor_expansion


# --------------------- Validating operator pairings ------------------------ #

def _z_joining_with_z_terms_eT(LHS, t_list, h, left_z, right_z):
    """Remove terms like `z^1_3 h^2` which require the z^1_3 to join with itself.

    We count the number of annihilation operators `b` and creation operators `d`
    provided for the left Z by the LHS, H and right Z operators.
    We count the number of annihilation operators `b` and creation operators `d`
    provided for the right Z by the LHS, H and left Z operators.

    If a left/right Z operator requires more `b`/`d`s than provided by the other
    operators this implies that left/right Z operator is contracting/joining with itself.
    Theoretically this doesn't exist, and therefore we reject this term.
    """
    available_b_for_both = LHS.n + h.n + sum([t.n for t in t_list])
    available_d_for_both = LHS.m + h.m + sum([t.m for t in t_list])

    available_b_for_left_z = right_z.n + available_b_for_both
    available_d_for_left_z = right_z.m + available_d_for_both

    available_b_for_right_z = left_z.n + available_b_for_both
    available_d_for_right_z = left_z.m + available_d_for_both

    required_b_for_left_z = left_z.m
    required_d_for_left_z = left_z.n

    required_b_for_right_z = right_z.m
    required_d_for_right_z = right_z.n

    if (required_b_for_left_z > available_b_for_left_z):
        return True

    if (required_d_for_left_z > available_d_for_left_z):
        return True

    if (required_b_for_right_z > available_b_for_right_z):
        return True

    if (required_d_for_right_z > available_d_for_right_z):
        return True

    return False


def _t_joining_with_t_terms_eT(omega, t_list, h, z_left, z_right):
    """Remove terms like `b h^1 t^2 t_2` which require the t^2 to join with t_2.

    We count the number of annihilation operators `b` and creation operators `d`
    provided by the Omega and H operators. Next we count the number of operators (`b`,`d`)
    required by all the t operators. If the t operators require more operators than
    Omega or H provide this implies that they would be contracting/joining with each other.
    Theoretically this doesn't exist, and therefore we reject this term.
    """
    available_d = omega.m + h.m + z_left.m + z_right.m
    available_b = omega.n + h.n + z_left.n + z_right.n

    required_b = sum([t.m for t in t_list])
    required_d = sum([t.n for t in t_list])

    if (required_b > available_b) or (required_d > available_d):
        return True

    return False


def _omega_joining_with_itself_eT(omega, t_list, h, z_left, z_right):
    """Remove terms like `bd h_0` which require Omega to join with itself.

    We already know that the number of operators is balanced, as we check
    that before calling this function. So here we check if the s or h terms have any b/d
    operators for omega to join with. If all these terms are h^0_0 and/or s^0_0 then omega
    must be joining with itself. Theoretically this doesn't exist, and therefore we reject this term.
    """

    # omega can't join with itself unless it has both creation and annihilation operators
    if (omega.m == 0) or (omega.n == 0):
        return False

    if omega.n > 0:
        if (h.m > 0) or (z_left.m > 0) or (z_right.m > 0):
            return False

        for t in t_list:
            if t.m > 0:
                return False

    if omega.m > 0:
        if (h.n > 0) or (z_left.n > 0) or (z_right.n > 0):
            return False

        for t in t_list:
            if t.n > 0:
                return False

    return True


def _h_joining_with_itself_eT(omega, t_list, h, z_left, z_right):
    """Remove terms like `h^1_1` which require h to join with itself.

    We already know that the number of operators is balanced, as we check
    that before calling this function. So here we check if the s or omega terms have any b/d
    operators for h to join with. If all these terms are o^0_0 and/or s^0_0 then h
    must be joining with itself. Theoretically this doesn't exist, and therefore we reject this term.
    """

    # h can't join with itself unless it has both creation and annihilation operators
    if (h.m == 0) or (h.n == 0):
        return False

    if h.n > 0:
        if (omega.m > 0) or (z_left.m > 0) or (z_right.m > 0):
            return False

        for t in t_list:
            if t.m > 0:
                return False

    if h.m > 0:
        if (omega.n > 0) or (z_left.n > 0) or (z_right.n > 0):
            return False

        for t in t_list:
            if t.n > 0:
                return False

    return True


# -------------------------------------------------------------------------------- #
# the `m_t` and `n_t` are lists of integers whose length is == number of t's
# so LHS^2_1 z_1 h^1 z_1 would mean m_l = [1, ] and m_r = [1, ] and n_h = [1, ]
connected_eT_lhs_operator_namedtuple = namedtuple(
    'connected_LHS',
    ['rank', 'm', 'n', 'm_l', 'n_l', 'm_t', 'n_t', 'm_h', 'n_h', 'm_r', 'n_r']
)
connected_eT_h_z_operator_namedtuple = namedtuple(
    'connected_h_z',
    ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_t', 'n_t', 'm_l', 'n_l', 'm_r', 'n_r']
)
# ------------------------------------------------------------------------ #
connected_eT_z_left_operator_namedtuple = namedtuple(
    'connected_z_left',
    ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_t', 'n_t', 'm_h', 'n_h', 'm_r', 'n_r']
)
connected_eT_z_right_operator_namedtuple = namedtuple(
    'connected_z_right',
    ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_t', 'n_t', 'm_h', 'n_h', 'm_l', 'n_l']
)
connected_t_operator_namedtuple = namedtuple(
    'connected_t',
    ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_l', 'n_l', 'm_h', 'n_h', 'm_r', 'n_r']
)
# ------------------------------------------------------------------------ #
disconnected_eT_z_left_operator_namedtuple = namedtuple(
    'disconnected_z_left',
    ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_t', 'n_t', 'm_h', 'n_h', 'm_r', 'n_r']
)
disconnected_eT_z_right_operator_namedtuple = namedtuple(
    'disconnected_z_right',
    ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_t', 'n_t', 'm_h', 'n_h', 'm_l', 'n_l']
)
disconnected_t_operator_namedtuple = namedtuple(
    'disconnected_t',
    ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_l', 'n_l', 'm_h', 'n_h', 'm_r', 'n_r']
)
# -------------------------------------------------------------------------------- #


def _generate_valid_eT_z_n_operator_permutations(LHS, eT, h, all_z_permutations):
    """ Remove s permutations whose b/d operators don't add up (theoretically can't exist)
    For example LHS^1 h_1 is allowed but not LHS^1 h_1 t^1 because we have 2 d operators but only 1 b operator.
    Additionally we need to make sure the t operators are not joining with themselves.
    This means that the b/d operators from Omega/LHS and h need to be sufficient to balance the b/d's from the t's.
    So LHS^1_1, h_2, t^1_1, t^2 is allowed but not LHS_1, h_1, t^1_1, t^2 because the t^1_1 term has to pair with the
    t^2, or in other words the only sources of d operators are t terms so the b operator from t^1_1 has to pair with
    a d from a t term. This is not allowed.
    """

    valid_permutations = []

    log.info(f"\n{tab}eT=" + f'\n{tab}{tab}'.join(['']+[str(t) for t in eT]))
    # generate all the possible valid permutations
    for perm_T in eT:
        subheader_log.info(f"{perm_T = }")
        for perm_Z in all_z_permutations:
            left_z, right_z = perm_Z

            str_group = f'\n{tab}'.join([
                '',
                f'{"LHS":<12s}{LHS}',
                f'{"T":<12s}{perm_T}',
                f'{"Z_left":<12s}{left_z}',
                f'{"h":<12s}{h}',
                f'{"Z_right":<12s}{right_z}',
            ])

            nof_creation_ops = LHS.m + h.m + left_z.m + right_z.m + sum([t.m for t in perm_T])
            nof_annhiliation_ops = LHS.n + h.n + left_z.n + right_z.n + sum([t.n for t in perm_T])
            cannot_pair_off_b_d_operators = bool(nof_creation_ops != nof_annhiliation_ops)

            # only terms which can pair off all operators are non zero
            if cannot_pair_off_b_d_operators:
                log.debug(f'Bad Permutation (b={nof_creation_ops} and d={nof_annhiliation_ops} are not balanced)' + str_group)
                continue

            # Omega/LHS and H need to satisfy all b/d requirements of the z terms
            # z terms can join with each other, but a single z term should not be able to join to itself!!
            if _z_joining_with_z_terms_eT(LHS, perm_T, h, left_z, right_z):
                log.debug('Bad Permutation (z joins with itself)' + str_group)
                continue

            # Omega/LHS and H need to satisfy all b/d requirements of the t terms
            # t terms can join with each other, but a single t term should not be able to join to itself!!
            if _t_joining_with_t_terms_eT(LHS, perm_T, h, left_z, right_z):
                log.debug('Bad Permutation (t joins with itself)' + str_group)
                continue

            # Omega/LHS must be able to connect with at least 1 b/d operator from h or a z_term otherwise it 'joins' with itself
            if _omega_joining_with_itself_eT(LHS, perm_T, h, left_z, right_z):
                log.debug('Bad Permutation (LHS joins with itself)' + str_group)
                continue

            # h must connect with at least 1 b/d operator from Omega/LHS or a z_term otherwise it 'joins' with itself
            if _h_joining_with_itself_eT(LHS, perm_T, h, left_z, right_z):
                log.debug('Bad Permutation (h joins with itself)' + str_group)
                continue

            if len(perm_T) == 1:
                # record a valid permutation
                log.debug('Good Permutation' + str_group)
                valid_permutations.append((tuple(perm_T), left_z, right_z))
            else:
                # record a valid permutation
                log.debug('Good Permutation' + str_group)
                valid_permutations.append((tuple(perm_T), left_z, right_z))

    return valid_permutations


def _generate_all_valid_eT_z_connection_permutations(LHS, t_list, h, left_z, right_z, log_invalid=True):
    """ Generate all possible valid combinations of z terms
    with LHS, t_list, and h over all index distributions.
    By convention the tuples are (lhs, eT, h, other_z).
    First we generate lists of all possible permutations whose sum is < the respective power (m/n)
    For example if left_z has m=4, n=2 then we generate an m_list
    which can be [0,0,0,0], [0,4,0,0], [1,1,1,1] but not things like [0,0,5,0] [2,1,1,1] etc..
    and the n_list would be [0,0,0,0], [0,0,0,2], [0,1,0,1] but not things like [1,1,1,0] [3,0,0,0] etc..
    """

    valid_upper_perm_combinations = []
    valid_lower_perm_combinations = []

    m_perms, n_perms = [], []

    # we need to dynamically adjust the temporary lists if we add more t terms
    nof_t_terms = len(t_list)

    """ generate all possible individual t assignments

    Here `a` represents LHS and `b` represents h so that we don't clobber the argument
    definitions for the top function
    """
    for s_term in [left_z, right_z]:
        M, N = s_term.m, s_term.n

        temp_list = []
        for a, b, *t in it.product(range(M+1), repeat=2+nof_t_terms):
            total = a+b+sum(t)
            if total <= M:
                temp_list.append((a, tuple(t), b, M-total))

        m_perms.append(temp_list)

        temp_list = []
        for a, b, *t in it.product(range(N+1), repeat=2+nof_t_terms):
            total = a+b+sum(t)
            if total <= N:
                temp_list.append((a, tuple(t), b, N-total))

        n_perms.append(temp_list)

    expander = lambda lst: f" {lst}" if len(lst) < 2 else ''.join([f'\n{tab}{tab}{m}' for m in lst])
    log.debug(''.join([
        "Uncombined permutations:",
        f"\n{tab}Z_left M perms:{expander(m_perms[0])}",
        f"\n{tab}Z_left N perms:{expander(n_perms[0])}",
        f"\n{tab}Z_right M perms:{expander(m_perms[1])}",
        f"\n{tab}Z_right N perms:{expander(n_perms[1])}",
    ]))

    # validate upper pairing
    combined_m_perms = list(it.product(*m_perms))
    for m_perm in combined_m_perms:
        # and `m_perm` is [Z_left, Z_right]
        # where each Z is [LHS, t_list, h, other_Z]
        # with numbers: [0, [0, ...], 0, 0]

        total_lhs_m = sum([t[0] for t in m_perm])
        total_eT_m = sum([sum(t[1]) for t in m_perm])
        total_h_m = sum([t[2] for t in m_perm])

        total_lhs_balanced = bool(total_lhs_m <= LHS.n)
        total_eT_balanced = bool(total_eT_m <= sum([t.n for t in t_list]))
        total_h_balanced = bool(total_h_m <= h.n)
        left_z_balanced_right = bool(m_perm[0][-1] <= right_z.n)
        right_z_balanced_left = bool(m_perm[1][-1] <= left_z.n)

        each_eT_balanced = all([
            bool(t.n <= (LHS.m + h.m + m_perm[0][1][i] + m_perm[1][1][i]))
            for i, t in enumerate(t_list)
        ])

        print(f"{each_eT_balanced=}")
        print([(t.n, LHS.m, h.m, m_perm[0][1][i], m_perm[1][1][i]) for i, t in enumerate(t_list)])
        # import pdb; pdb.set_trace()

        dense_output = f"\n{tab}".join([
            '',
            f"{'Z_left  perm '}{m_perm[0]}",
            f"{'Z_right perm '}{m_perm[1]}",
            f"{'LHS':<4}{total_lhs_m:>2d} <= {LHS.n:>2d}  {total_lhs_balanced}",
            f"{'eT':<4}{total_eT_m:>2d} <= {sum([t.n for t in t_list]):>2d}  {total_eT_balanced}",
            f"{each_eT_balanced=}",
            f"{'h':<4}{total_h_m:>2d} <= {h.n:>2d}  {total_h_balanced}",
            f"{'zL':<4}{m_perm[0][-1]:>2d} <= {right_z.n:>2d}  {left_z_balanced_right}",
            f"{'zR':<4}{m_perm[1][-1]:>2d} <= {left_z.n:>2d}  {right_z_balanced_left}",
        ])

        bool_list = [
            total_h_balanced,
            total_lhs_balanced,
            total_eT_balanced,
            each_eT_balanced,
            left_z_balanced_right,
            right_z_balanced_left
        ]

        if all(bool_list):
            log.debug("  Valid upper perm" + dense_output)
            valid_upper_perm_combinations.append(m_perm)

        elif log_invalid:
            log.debug("Invalid upper perm" + dense_output)

    # validate lower pairing
    combined_n_perms = list(it.product(*n_perms))
    for n_perm in combined_n_perms:
        # and `n_perm` is [Z_left, Z_right]
        # where each Z is [LHS, t_list, h, other_Z]
        # with numbers: [0, [0, ...], 0, 0]

        total_lhs_n = sum([t[0] for t in n_perm])
        total_eT_n = sum([sum(t[1]) for t in n_perm])
        total_h_n = sum([t[2] for t in n_perm])

        total_lhs_balanced = bool(total_lhs_n <= LHS.m)
        total_eT_balanced = bool(total_eT_n <= sum([t.m for t in t_list]))
        total_h_balanced = bool(total_h_n <= h.m)
        left_z_balanced_right = bool(n_perm[0][-1] <= right_z.m)
        right_z_balanced_left = bool(n_perm[1][-1] <= left_z.m)

        each_eT_balanced = all([
            bool(t.m <= (LHS.n + h.n + n_perm[0][1][i] + n_perm[1][1][i]))
            for i, t in enumerate(t_list)
        ])

        dense_output = f"\n{tab}".join([
            '',
            f"{'Z_left  perm '}{n_perm[0]}",
            f"{'Z_right perm '}{n_perm[1]}",
            f"{'LHS':<4}{total_lhs_n:>2d} <= {LHS.m:>2d}  {total_lhs_balanced}",
            f"{'eT':<4}{total_eT_n:>2d} <= {sum([t.m for t in t_list]):>2d}  {total_eT_balanced}",
            f"{each_eT_balanced=}",
            f"{'h':<4}{total_h_n:>2d} <= {h.m:>2d}  {total_h_balanced}",
            f"{'zL':<4}{n_perm[0][-1]:>2d} <= {right_z.m:>2d}  {left_z_balanced_right}",
            f"{'zR':<4}{n_perm[1][-1]:>2d} <= {left_z.m:>2d}  {right_z_balanced_left}",
        ])

        bool_list = [
            total_h_balanced,
            total_lhs_balanced,
            total_eT_balanced,
            each_eT_balanced,
            left_z_balanced_right,
            right_z_balanced_left
        ]

        if all(bool_list):
            log.debug("  Valid lower perm" + dense_output)
            valid_lower_perm_combinations.append(n_perm)

        elif log_invalid:
            log.debug("Invalid lower perm" + dense_output)

    return valid_upper_perm_combinations, valid_lower_perm_combinations


def _generate_all_valid_eT_connection_permutations(LHS, t_list, h, z_pair, log_invalid=True):
    """ Generate all possible valid combinations of t terms
    with LHS, z_left, h, and z_right over all index distributions.
    By convention the tuples are (lhs, eT, h, other_z).
    """

    z_left, z_right = z_pair

    if z_left is None:
        args = [0, 0, 0, 0, 0, [0, ]*len(t_list), [0, ]*len(t_list), 0, 0, 0, 0]
        z_left = disconnected_eT_z_left_operator_namedtuple(*args)

    if z_right is None:
        args = [0, 0, 0, 0, 0, [0, ]*len(t_list), [0, ]*len(t_list), 0, 0, 0, 0]
        z_right = disconnected_eT_z_right_operator_namedtuple(*args)

    valid_upper_perm_combinations = []
    valid_lower_perm_combinations = []
    m_perms, n_perms = [], []

    remaining_m = sum([t.m for t in t_list])
    remaining_m -= sum(z_left.n_t)
    remaining_m -= sum(z_right.n_t)

    remaining_n = sum([t.n for t in t_list])
    remaining_n -= sum(z_left.m_t)
    remaining_n -= sum(z_right.m_t)

    print('\nKK', remaining_m, remaining_n, t_list)

    if remaining_n == remaining_m == 0:

        temp_upper, temp_lower = [], []
        print('VV1', z_left.n_t, z_right.n_t)
        print('VV2', z_left.m_t, z_right.m_t)
        # print(z_left)
        # print(z_right)
        # print(t_list)

        for i, t in enumerate(t_list):
            temp_upper.append([0, z_left.n_t[i], 0, z_right.n_t[i]])
            temp_lower.append([0, z_left.m_t[i], 0, z_right.m_t[i]])

        valid_upper_perm_combinations.append(temp_upper)
        valid_lower_perm_combinations.append(temp_lower)

    elif remaining_m == 0:
        print(f"\n{remaining_n=}")
        print(t_list)

        # ----------------------------------------------------------
        for i, t in enumerate(t_list):
            M = t.m - z_left.n_t[i] - z_right.n_t[i]
            N = t.n - z_left.m_t[i] - z_right.m_t[i]

            temp_list = []
            print(f"{M=} {N=}")
            for a in range(M+1):
                temp_list.append((a, z_left.n_t[i], M-a, z_right.n_t[i]))

            m_perms.append(temp_list)

            temp_list = []
            for a in range(N+1):
                temp_list.append((a, z_left.m_t[i], N-a, z_right.m_t[i]))

            n_perms.append(temp_list)

        print(f"original {n_perms=}")
        # ----------------------------------------------------------
        combined_n_perms = list(it.product(*n_perms))
        for n_perm in combined_n_perms:
            print(f"{n_perm=}")

            total_lhs_n = sum([t[0] for t in n_perm])
            total_h_n = sum([t[2] for t in n_perm])

            total_lhs_balanced = bool(total_lhs_n <= LHS.m)
            total_h_balanced = bool(total_h_n <= h.m)

            dense_output = f"\n{tab}".join([
                '',
                f"{'LHS':<4}{total_lhs_n:>2d} >= {LHS.m:>2d}  {total_lhs_balanced}",
                f"{'h':<4}{total_h_n:>2d} >= {h.m:>2d}  {total_h_balanced}",
                f"{'T':<4}{n_perm}",
                f"{'zL':<4}{z_left}",
                f"{'zR':<4}{z_right}",
            ])

            if total_h_balanced and total_lhs_balanced:
                log.debug("  Valid lower perm" + dense_output)
                valid_lower_perm_combinations.append(n_perm)

            elif log_invalid:
                log.debug("Invalid lower perm" + dense_output)

        print(f"original {m_perms=}")
        # ----------------------------------------------------------
        combined_m_perms = list(it.product(*m_perms))
        for m_perm in combined_m_perms:
            print(f"{m_perm=}")

            total_lhs_m = sum([t[0] for t in m_perm])
            total_h_m = sum([t[2] for t in m_perm])

            total_lhs_balanced = bool(total_lhs_m <= LHS.n)
            total_h_balanced = bool(total_h_m <= h.n)

            dense_output = f"\n{tab}".join([
                '',
                f"{'LHS':<4}{total_lhs_m:>2d} >= {LHS.n:>2d}  {total_lhs_balanced}",
                f"{'h':<4}{total_h_m:>2d} >= {h.n:>2d}  {total_h_balanced}",
                f"{'T':<4}{m_perm}",
                f"{'zL':<4}{z_left}",
                f"{'zR':<4}{z_right}",
            ])

            if total_h_balanced and total_lhs_balanced:
                log.debug("  Valid upper perm" + dense_output)
                valid_upper_perm_combinations.append(m_perm)

            elif log_invalid:
                log.debug("Invalid upper perm" + dense_output)

    elif remaining_n == 0:
        raise Exception("have not coded this yet")

    else:
        raise Exception("have not coded this yet")

    return valid_upper_perm_combinations, valid_lower_perm_combinations


def _generate_all_o_eT_h_z_connection_permutations(LHS, h, valid_permutations, found_it_bool=False):
    """ Generate all possible permutations of matching with LHS, and h for e^T and z_terms """

    annotated_permutations = []  # store output here
    annotated_z_permutations = []  # store intermediate output here

    def print_triplet(i, p):
        et_string = ''.join([
            f"{'(':<8s}",
            f'\n{tab}{tab}{tab}{tab}{tab}'.join([str(t) for t in p[0]]),
            f"\n{tab}{tab})"
        ])

        return str(
            f'\n{tab}{i+1:<4}{"eT":<4s}{et_string}'
            f'\n{tab}{tab}{"Z left":<12s}{p[1]}'
            f'\n{tab}{tab}{"Z right":<12s}{p[2]}'
        )

    log.info(f"\n{tab}valid_permutations=" + ''.join([
        print_triplet(i, p)
        for i, p in enumerate(valid_permutations)
    ]))

    for i, triplet in enumerate(valid_permutations):

        subheader_log.debug(f"PERMUTATION{i+1:>4}")
        log.debug(f"\n{tab}Processing permutation" + print_triplet(i, triplet))

        # unpack the triplet
        t_list, left_z, right_z = triplet

        upper_perms, lower_perms = _generate_all_valid_eT_z_connection_permutations(LHS, t_list, h, left_z, right_z)

        expander = lambda lst: f" {lst}" if len(lst) < 2 else ''.join([f'\n{tab}{tab}{m}' for m in lst])
        log.debug(''.join([
            f"\n{tab}Connected permutations (Z left / Z right):",
            f"\n{tab}M perms:{expander(upper_perms)}",
            f"\n{tab}N perms:{expander(lower_perms)}",
        ]))

        # compute all the permutations for z
        for upper in upper_perms:
            for lower in lower_perms:
                assert len(upper) == len(lower)

                left_z_upper, right_z_upper = upper
                left_z_lower, right_z_lower = lower
                z_left_kwargs, z_right_kwargs = {}, {}
                z_pair = []

                # for each Z operator we make a `connected_namedtuple` or a `disconnected_namedtuple`
                if left_z.name is None:
                    # make sure this permutation is okay for no z left
                    assert left_z_upper == left_z_lower
                    assert left_z_upper[0] == 0 and left_z_upper[2:] == (0, 0)
                    assert all([x == 0 for x in left_z_upper[1]])
                    z_pair.append(None)
                else:
                    z_left_kwargs = {
                        'rank': left_z.rank,
                        'm': left_z.m,
                        'm_lhs': left_z_upper[0],
                        'm_t':   left_z_upper[1],
                        'm_h':   left_z_upper[2],
                        'm_r':   left_z_upper[-1],
                        'n': left_z.n,
                        'n_lhs': left_z_lower[0],
                        'n_t':   left_z_lower[1],
                        'n_h':   left_z_lower[2],
                        'n_r':   left_z_lower[-1],
                    }
                    # if the Z operator is disconnected (meaning no connections to H)
                    if z_left_kwargs['m_h'] == z_left_kwargs['n_h'] == 0:
                        z_pair.append(disconnected_eT_z_left_operator_namedtuple(**z_left_kwargs))
                    # if the Z operator is connected (at least 1 connection to H)
                    else:
                        z_pair.append(connected_eT_z_left_operator_namedtuple(**z_left_kwargs))

                if right_z.name is None:
                    # make sure this permutation is okay for no z right
                    assert right_z_upper == right_z_lower
                    assert right_z_upper[0] == 0 and right_z_upper[2:] == (0, 0)
                    assert all([x == 0 for x in right_z_upper[1]])
                    z_pair.append(None)
                else:
                    z_right_kwargs = {
                        'rank': right_z.rank,
                        'm': right_z.m,
                        'm_lhs': right_z_upper[0],
                        'm_t':   right_z_upper[1],
                        'm_h':   right_z_upper[2],
                        'm_l':   right_z_upper[-1],
                        'n': right_z.n,
                        'n_lhs': right_z_lower[0],
                        'n_t':   right_z_lower[1],
                        'n_h':   right_z_lower[2],
                        'n_l':   right_z_lower[-1],
                    }
                    # if the Z operator is disconnected (meaning no connections to H)
                    if z_right_kwargs['m_h'] == z_right_kwargs['n_h'] == 0:
                        z_pair.append(disconnected_eT_z_right_operator_namedtuple(**z_right_kwargs))
                    # if the Z operator is connected (at least 1 connection to H)
                    else:
                        z_pair.append(connected_eT_z_right_operator_namedtuple(**z_right_kwargs))

                # if we have the ZHZ terms then we need to check that the Z <-> Z contractions are correct
                if (z_left_kwargs != {}) and (z_right_kwargs != {}):

                    # if these contractions are not equal
                    if z_left_kwargs['m_r'] != z_right_kwargs['n_l']:
                        term_string = f"{tab}{LHS}, {h}, {triplet}\n{tab}{z_left_kwargs=}\n{tab}{z_right_kwargs=}\n"
                        log.debug(
                            f"Invalid term (z_left.m_r({z_left_kwargs['m_r']}) != z_right.n_l({z_right_kwargs['n_l']}))\n"
                            f"{term_string}"
                        )
                        assert False
                        continue

                    # if these contractions are not equal
                    if z_left_kwargs['n_r'] != z_right_kwargs['m_l']:
                        term_string = f"{tab}{LHS}, {h}, {triplet}\n{tab}{z_left_kwargs=}\n{tab}{z_right_kwargs=}\n"
                        log.debug(
                            f"Invalid term (z_left.n_r({z_left_kwargs['n_r']}) != z_right.m_l({z_right_kwargs['m_l']}))\n"
                            f"{term_string}"
                        )
                        assert False
                        continue

                log.debug(
                    f"\n{tab}Z left  {z_pair[0]}"
                    f"\n{tab}Z right {z_pair[1]}"
                )
                annotated_z_permutations.append(tuple(z_pair))

        for z_pair in annotated_z_permutations:

            print('t_list', t_list)
            print('z_pair', z_pair)
            # now we have to compute all the permutations for t
            if len(t_list) == 1 and t_list[0].rank == 0:
                upper_perms = [[[0, ] * 4, ], ]
                lower_perms = [[[0, ] * 4, ], ]
            else:
                upper_perms, lower_perms = _generate_all_valid_eT_connection_permutations(LHS, t_list, h, z_pair)

            log.debug(f"{upper_perms=}")
            log.debug(f"{lower_perms=}")

            # compute all the permutations for t
            for upper in upper_perms:   # for now the upper_perms only have 1 permutation (all zeros)
                for lower in lower_perms:

                    perm_list = []
                    print('XX', upper, lower, t_list)
                    # for each t operator we make a `connected_namedtuple` or a `disconnected_namedtuple`
                    for i, t in enumerate(t_list):
                        print(f"{i=}", t)
                        t_upper = upper[i]
                        t_lower = lower[i]
                        print('YY', t_upper, t_lower)
                        assert list(t_upper) == [0, 0, 0, 0]  # for now t's can't have upper components

                        if t.m != sum(t_upper):
                            log.debug(f"Bad t perms + z perms: {t.m = } {t_upper = }")
                            raise Exception('sds')

                        if t.n != sum(t_lower):
                            log.debug(f"Bad t perms + z perms: {t.n = } {t_lower = }")
                            raise Exception('sds')

                        t_kwargs = {
                            'rank': t.rank,
                            'm': t.m,
                            'm_lhs': t_upper[0],
                            'm_l':   t_upper[1],
                            'm_h':   t_upper[2],
                            'm_r':   t_upper[-1],
                            'n': t.n,
                            'n_lhs': t_lower[0],
                            'n_l':   t_lower[1],
                            'n_h':   t_lower[2],
                            'n_r':   t_lower[-1],
                        }

                        # if the t operator is disconnected (meaning no connections to H)
                        if t_kwargs['m_h'] == t_kwargs['n_h'] == 0:
                            perm_list.append(disconnected_t_operator_namedtuple(**t_kwargs))
                        # if the t operator is connected (at least 1 connection to H)
                        else:
                            perm_list.append(connected_t_operator_namedtuple(**t_kwargs))

                    # after looping over t_list we append the list of operators
                    # for this specific permutation

            annotated_permutations.append((tuple(perm_list), z_pair))

        # print(annotated_permutations)
        # import pdb; pdb.set_trace()

        splitperm = lambda array: f'\n{tab}{tab}'.join(['']+[str(t) for t in array[0]])

        for perm in annotated_permutations:
            log.debug(
                f"\n{tab}Accepted Permutation ({splitperm(perm)}"
                f"\n{tab})"
                f"\n{tab}{perm[1]}"
            )

    return annotated_permutations


def _remove_duplicate_eT_z_permutations(LHS, h, eT_connection_permutations):
    """ x """
    duplicate_set = set()

    # print('\n\n', eT_connection_permutations)
    for i, perm in enumerate(eT_connection_permutations):
        t_tuple, z_pair = perm

        print('\n', perm)
        print('\n', i, t_tuple, z_pair)
        # a.sort()
        # a = tuple(a)

        splitperm = lambda array: f'\n{tab}{tab}'.join(['']+[str(t) for t in array[0]])

        if perm not in duplicate_set:
            log.debug(f"\n{tab}Added unique: ({splitperm(perm)}\n{tab})\n{tab}{perm[1]}")
            duplicate_set.add(perm)
        else:
            log.debug(f"\n{tab}Removed duplicate: {splitperm(perm)}\n{tab})\n{tab}{perm[1]}")
            pass

    return duplicate_set


def _generate_explicit_eT_z_connections(LHS, h, unique_permutations):
    """ Generate new namedtuples for LHS and h explicitly labeling how they connect with each other and t.
    We make `connected_lhs_operator_namedtuple` and `connected_h_z_operator_namedtuple`.
    The output `labeled_permutations` is a list where each element is `[new_LHS, new_eT, new_h, z_left, z_right]`.
    We also check to make sure each term is valid.
    """

    labeled_permutations = []  # store output here

    for perm in unique_permutations:
        t_list, z_pair = perm
        z_left, z_right = z_pair
        lhs_kwargs, h_kwargs = {}, {}

        assert len(perm) == 2

        # bool declarations for readability
        z_left_exists = isinstance(z_left, (connected_eT_z_left_operator_namedtuple, disconnected_eT_z_left_operator_namedtuple))
        z_right_exists = isinstance(z_right, (connected_eT_z_right_operator_namedtuple, disconnected_eT_z_right_operator_namedtuple))

        # sanity checks
        if z_left is None:
            assert z_right_exists
        elif z_right is None:
            assert z_left_exists
        else:
            assert z_right_exists
            assert z_left_exists

        lhs_kwargs = {
            'm_l': z_left.n_lhs if z_left_exists else 0,
            'n_l': z_left.m_lhs if z_left_exists else 0,
            'm_r': z_right.n_lhs if z_right_exists else 0,
            'n_r': z_right.m_lhs if z_right_exists else 0,
            'm_t': [t.n_lhs for t in t_list],
            'n_t': [t.m_lhs for t in t_list]
        }

        h_kwargs = {
            'm_l': z_left.n_h if z_left_exists else 0,
            'n_l': z_left.m_h if z_left_exists else 0,
            'm_r': z_right.n_h if z_right_exists else 0,
            'n_r': z_right.m_h if z_right_exists else 0,
            'm_t': [t.n_h for t in t_list],
            'n_t': [t.m_h for t in t_list]
        }

        lhs_kwargs.update({'rank': LHS.m + LHS.n, 'm': LHS.m, 'n': LHS.n})
        h_kwargs.update({'rank': h.m + h.n, 'm': h.m, 'n': h.n})

        # calculate the contractions as the remainder after all other contractions
        lhs_kwargs['m_h'] = lhs_kwargs['m'] - (sum(lhs_kwargs['m_t']) + lhs_kwargs['m_l'] + lhs_kwargs['m_r'])
        lhs_kwargs['n_h'] = lhs_kwargs['n'] - (sum(lhs_kwargs['n_t']) + lhs_kwargs['n_l'] + lhs_kwargs['n_r'])
        h_kwargs['m_lhs'] = h_kwargs['m'] - (sum(h_kwargs['m_t']) + h_kwargs['m_l'] + h_kwargs['m_r'])
        h_kwargs['n_lhs'] = h_kwargs['n'] - (sum(h_kwargs['n_t']) + h_kwargs['n_l'] + h_kwargs['n_r'])

        # make sure these values are not negative
        # otherwise our math went horribly wrong somewhere and we over counted?
        # the balancing math should have been already worked out before we get to this function
        assert lhs_kwargs['m_h'] >= 0 and lhs_kwargs['n_h'] >= 0
        assert h_kwargs['m_lhs'] >= 0 and h_kwargs['n_lhs'] >= 0

        # if these contractions are not equal
        if h_kwargs['m_lhs'] != lhs_kwargs['n_h']:
            term_string = f"{tab}{LHS}, {t_list}, {h}, {perm}\n{tab}{lhs_kwargs=}\n{tab}{h_kwargs=}\n"
            log.debug(f"Found an invalid term (h.m_lhs != LHS.n_h)\n{term_string}")
            continue

        # if these contractions are not equal
        elif h_kwargs['n_lhs'] != lhs_kwargs['m_h']:
            term_string = f"{tab}{LHS}, {t_list}, {h}, {perm}\n{tab}{lhs_kwargs=}\n{tab}{h_kwargs=}\n"
            log.debug(f"Found an invalid term (h.n_lhs != LHS.m_h)\n{term_string}")
            continue

        # cheating for the moment

        new_LHS = connected_eT_lhs_operator_namedtuple(**lhs_kwargs)
        new_h = connected_eT_h_z_operator_namedtuple(**h_kwargs)

        labeled_permutations.append([new_LHS, t_list, new_h, z_pair])
        for p in labeled_permutations:
            old_print_wrapper('\n\np')
            for x in p:
                old_print_wrapper(x)
            old_print_wrapper('\n\n')
        # sys.exit(0)

    return labeled_permutations


# -------------------------------------------------------------------------------- #

def _f_t_h_contributions(t_list, h):
    """ x """
    return_list = []

    for i, t in enumerate(t_list):
        if isinstance(t, disconnected_t_operator_namedtuple):
            assert 0 == t.m_h == h.n_t[i]
            return_list.append(0)
        else:
            assert t.m_h == h.n_t[i]
            return_list.append(t.m_h)

    return return_list


def _fbar_t_h_contributions(t_list, h):
    """ x """
    return_list = []

    for i, t in enumerate(t_list):
        if isinstance(t, disconnected_t_operator_namedtuple):
            assert 0 == t.n_h == h.m_t[i]
            return_list.append(0)
        else:
            assert t.n_h == h.m_t[i] >= 1
            return_list.append(t.n_h)

    return return_list


def _f_t_zR_contributions(t_list, z_right):
    """ x """
    return_list = []

    for i, t in enumerate(t_list):
        assert t.m_r == z_right.n_t[i]
        return_list.append(t.m_r)

    return return_list


def _fbar_t_zR_contributions(t_list, z_right):
    """ x """
    return_list = []

    for i, t in enumerate(t_list):
        print(t, z_right)
        assert t.n_r == z_right.m_t[i]
        return_list.append(t.n_r)

    return return_list


def _f_t_zL_contributions(t_list, z_left):
    """ x """
    return_list = []

    for i, t in enumerate(t_list):
        assert t.m_l == z_left.n_t[i]
        return_list.append(t.m_l)

    return return_list


def _fbar_t_zL_contributions(t_list, z_left):
    """ x """
    return_list = []

    for i, t in enumerate(t_list):
        assert t.n_l == z_left.m_t[i]
        return_list.append(t.n_l)

    return return_list


def _build_eT_term_latex_labels(t_list, offset_dict, color=True):
    """ Builds latex code for labeling a `connected_t_operator_namedtuple`."""

    return_list = []

    print("t_list\n", t_list, '\n')
    for t in t_list:
        if t.rank == 0:
            return f"{bold_t_latex}_0"

        upper_indices, lower_indices = '', ''

        # subscript indices
        if t.n > 0:
            # contract with left z
            lower_indices += r'\magenta{' + z_summation_indices[0:t.n_l] + '}'
            offset_dict['summation_index'] += t.n_l

            # contract with h
            b = offset_dict['summation_index']
            lower_indices += r'\blue{' + z_summation_indices[b:b + t.n_h] + '}'
            offset_dict['summation_index'] += t.n_h

            # contract with right z
            s = offset_dict['summation_index']
            lower_indices += r'\magenta{' + z_summation_indices[s:s + t.n_r] + '}'
            offset_dict['summation_index'] += t.n_r

            # pair with left hand side (LHS)
            u = offset_dict['unlinked_index']
            lower_indices += r'\red{' + z_unlinked_indices[u:u + t.n_lhs] + '}'
            offset_dict['unlinked_index'] += t.n_lhs

        # superscript indices
        if t.m > 0:
            # contract with left z
            a = offset_dict['summation_index']
            upper_indices += r'\magenta{' + z_summation_indices[a:a + t.m_l] + '}'
            offset_dict['summation_index'] += t.m_l

            # contract with h
            b = offset_dict['summation_index']
            lower_indices += r'\blue{' + z_summation_indices[b:b + t.n_h] + '}'
            offset_dict['summation_index'] += t.n_h

            # contract with right z
            s = offset_dict['summation_index']
            upper_indices += r'\magenta{' + z_summation_indices[s:s + t.m_r] + '}'
            offset_dict['summation_index'] += t.m_r

            # pair with left hand side (LHS)
            u = offset_dict['unlinked_index']
            upper_indices += r'\red{' + z_unlinked_indices[u:u + t.m_lhs] + '}'
            offset_dict['unlinked_index'] += t.m_lhs

        return_list.append(f"{bold_t_latex}^{{{upper_indices}}}_{{{lower_indices}}}")

    return ''.join(return_list)


def _build_eT_hz_term_latex_labels(h, offset_dict, color=True):
    """ Builds latex code for labeling a `connected_h_operator_namedtuple`."""

    if h.rank == 0:
        return f"{bold_h_latex}_0"

    upper_indices, lower_indices = '', ''

    # subscript indices
    if h.n > 0:
        # contract with left t terms
        lower_indices += r'\blue{' + z_summation_indices[0:sum(h.n_t)] + '}'

        # contract with left z
        s = offset_dict['summation_index']
        lower_indices += r'\blue{' + z_summation_indices[s:s + h.n_l] + '}'
        offset_dict['summation_index'] += h.n_l

        # contract with right z
        s = offset_dict['summation_index']
        lower_indices += r'\blue{' + z_summation_indices[s:s + h.n_r] + '}'
        offset_dict['summation_index'] += h.n_r

        # pair with left hand side (LHS)
        u = offset_dict['unlinked_index']
        lower_indices += r'\red{' + z_unlinked_indices[u:u + h.n_lhs] + '}'
        offset_dict['unlinked_index'] += h.n_lhs

    # superscript indices
    if h.m > 0:
        # contract with right t terms
        a = offset_dict['t_upper']
        upper_indices += r'\blue{' + z_summation_indices[a:a + sum(h.m_t)] + '}'

        # contract with left z
        a = offset_dict['left_upper']
        upper_indices += r'\blue{' + z_summation_indices[a:a + h.m_l] + '}'

        # contract with right z
        s = offset_dict['summation_index']
        upper_indices += r'\blue{' + z_summation_indices[s:s + h.m_r] + '}'
        offset_dict['summation_index'] += h.m_r

        # pair with left hand side (LHS)
        u = offset_dict['unlinked_index']
        upper_indices += r'\red{' + z_unlinked_indices[u:u + h.m_lhs] + '}'
        offset_dict['unlinked_index'] += h.m_lhs

    return f"{bold_h_latex}^{{{upper_indices}}}_{{{lower_indices}}}"


def _build_eT_right_z_term(h, z_right, offset_dict, color=True):
    """ Builds latex code for labeling a `connected_z_right_operator_namedtuple`.

    The `condense_offset` is an optional argument which is needed when creating latex code
    for linked disconnected terms in a condensed format.
    """
    if z_right.rank == 0:
        return f"{bold_z_latex}_0"

    a, b = 0, 0
    upper_indices, lower_indices = '', ''

    # subscript indices
    if z_right.n > 0:
        # contract with left t terms
        a = offset_dict['t_lower']
        lower_indices += r'\magenta{' + z_summation_indices[a:a + sum(z_right.n_t)] + '}'

        # contract with left z
        b = offset_dict['left_lower']
        lower_indices += r'\magenta{' + z_summation_indices[a+b:a+b + z_right.n_l] + '}'

        # contract with h
        c = offset_dict['h_lower']
        lower_indices += r'\blue{' + z_summation_indices[c:c + z_right.n_h] + '}'

        # pair with left hand side (LHS)
        u = offset_dict['unlinked_index']
        lower_indices += r'\red{' + z_unlinked_indices[u:u + z_right.n_lhs] + '}'
        offset_dict['unlinked_index'] += z_right.n_lhs

    # superscript indices
    if z_right.m > 0:
        # contract with right t terms
        a = offset_dict['t_upper']
        upper_indices += r'\magenta{' + z_summation_indices[a:a + sum(z_right.m_t)] + '}'

        # contract with left z
        b = offset_dict['left_upper']
        upper_indices += r'\magenta{' + z_summation_indices[b:b + z_right.m_l] + '}'

        # contract with h
        c = offset_dict['h_upper']
        upper_indices += r'\blue{' + z_summation_indices[c:c + z_right.m_h] + '}'

        # pair with left hand side (LHS)
        u = offset_dict['unlinked_index']
        upper_indices += r'\red{' + z_unlinked_indices[u:u + z_right.m_lhs] + '}'
        offset_dict['unlinked_index'] += z_right.m_lhs

    return f"{bold_z_latex}^{{{upper_indices}}}_{{{lower_indices}}}"


def _prepare_third_eTz_latex(term_list, split_width=7, remove_f_terms=False, print_prefactors=False):
    """Return the latex commands to write the provided terms.

    The `split_width` is the maximum number of terms on 1 horizontal line (in latex) and should
    be changed as needed to fit equations on page.
    If `remove_f_terms` is true terms then terms where nof_fs > 0 will are not written to latex.
    If `print_prefactors` is true then we add the prefactor string generated by `_build_z_latex_prefactor`.
    """
    print('\n\n')
    return_list = []  # store output here

    # for term in term_list:
    #     LHS, t_list, h, z_left, z_right = term[0], term[1], term[2], *term[3]
    #     print(z_right)
    # print('\n\n')

    # for term in term_list:
    #     LHS, t_list, h, z_left, z_right = term[0], term[1], term[2], *term[3]
    #     print(h)

    # import pdb; pdb.set_trace()

    # prepare all the latex strings
    for term in term_list:
        term_string = ''

        # old_print_wrapper("TERM", term)

        # extract elements of list `term`
        LHS, t_list, h, z_left, z_right = term[0], term[1], term[2], *term[3]

        # if needed add f prefactors
        nof_fs = _f_h_zR_contributions(h, z_right)
        nof_fs += sum(_f_t_h_contributions(t_list, h))
        nof_fs += sum(_f_t_zR_contributions(t_list, z_right))

        if remove_f_terms and (nof_fs > 0):
            continue
        if nof_fs > 0:
            term_string += "f" if (nof_fs == 1) else f"f^{{{nof_fs}}}"

        # if needed add fbar prefactors
        nof_fbars = _fbar_h_zR_contributions(h, z_right)
        nof_fbars += sum(_fbar_t_h_contributions(t_list, h))
        nof_fbars += sum(_fbar_t_zR_contributions(t_list, z_right))
        if nof_fbars > 0:
            term_string += "\\bar{f}" if (nof_fbars == 1) else f"\\bar{{f}}^{{{nof_fbars}}}"

        # add any prefactors if they exist
        if print_prefactors:
            raise Exception("prefactor code for z stuff is not done")
            term_string += _build_z_latex_prefactor(h, z_right)

        # prepare the z terms
        h_offset_dict = {
            't_upper': sum([t.m - t.m_lhs for t in t_list]),
            'left_upper': 0,
            'summation_index': sum([t.rank - t.m_lhs - t.n_lhs for t in t_list]),
            'unlinked_index': sum([t.m_lhs + t.n_lhs for t in t_list])
        }

        right_z_offset_dict = {
            't_lower': sum([t.m_h for t in t_list]),
            't_upper': h_offset_dict['left_upper'] + sum([t.n_h for t in t_list]),
            'left_lower': 0,
            'left_upper': 0,
            'h_lower': h_offset_dict['summation_index'] + h.n_r,
            'h_upper': h_offset_dict['summation_index'],
            'unlinked_index': h_offset_dict['unlinked_index'] + h.m_lhs + h.n_lhs
        }

        t_terms = _build_eT_term_latex_labels(
            t_list,
            {
                'left_upper': 0,
                'summation_index': 0,
                'unlinked_index': 0
            }
        )

        # print(t_terms)
        # import pdb; pdb.set_trace()

        left_z = ''
        h_term = _build_eT_hz_term_latex_labels(h, h_offset_dict)
        right_z = _build_eT_right_z_term(h, z_right, right_z_offset_dict)

        # build the latex code representing this term in the sum
        term_string += t_terms + left_z + h_term + right_z

        # store the result
        return_list.append(term_string)

    unique_list = list(set(return_list))

    for i, a in enumerate(return_list):
        print(i+1, a)

    assert len(unique_list) == len(return_list), "Duplicate terms, logic is incorrect"

    log.info(f'\n{tab}Prepared:\n{tab}' + f'\n{tab}'.join([s for s in return_list]))

    # print(len(return_list), split_width*2)
    # import pdb; pdb.set_trace()

    # if the line is so short we don't need to split
    if len(return_list) <= split_width+2:
        return f"({' + '.join(return_list)})"

    split_equation_list = []
    # print('a', len(return_list) // split_width)

    for i in range(0, len(return_list) // split_width):
        split_equation_list.append(' + '.join(return_list[i*split_width:(i+1)*split_width]))

    # make sure we pickup the last few terms
    last_few_terms = (len(return_list) % split_width)-(split_width+1)
    assert last_few_terms < 0, 'whoops, math error'
    split_equation_list.append(' + '.join(return_list[last_few_terms:]))

    # join the lists with the equation splitting string
    splitting_string = r'\\  &+  % split long equation'
    final_string = f"\n{tab}{splitting_string}\n".join(split_equation_list)

    # and we're done!
    return f"(\n{final_string}\n"

# -------------------------------------------------------------------------------- #


def _prepare_eTz_z_terms(Z_left, Z_right, zhz_debug=False):
    """ Factor out z preparation from `_filter_out_valid_eTz_terms` """

    # H*Z terms, straightforward
    if Z_left is None:
        header_log.info("Z is on the right")
        z_left_terms = [general_operator_namedtuple(None, 0, 0, 0), ]
        z_right_terms = Z_right.operator_list
        assert isinstance(z_right_terms, list) and isinstance(z_right_terms[0], general_operator_namedtuple)

    # Z*H terms, straightforward
    elif Z_right is None:
        header_log.info("Z is on the left")
        z_left_terms = Z_left.operator_list
        z_right_terms = [general_operator_namedtuple(None, 0, 0, 0), ]
        # z_right_terms = Z_left.operator_list
        assert isinstance(z_left_terms, list) and isinstance(z_left_terms[0], general_operator_namedtuple)

        # valid_lower_perms = [list(it.dropwhile(lambda y: y == 0, x)) for x in unique_permutations if (maximum >= sum(x))]
        # valid_lower_perms[valid_lower_perms.index([])] = [0]

    # Z*H*Z terms, most complicated
    else:
        zhz_debug = True
        header_log.info("Z is on both sides")
        z_left_terms = Z_left.operator_list
        z_right_terms = Z_right.operator_list
        assert isinstance(z_right_terms, list) and isinstance(z_right_terms[0], general_operator_namedtuple)
        assert isinstance(z_left_terms, list) and isinstance(z_left_terms[0], general_operator_namedtuple)

    all_z_permutations = [(z_left, z_right) for z_left, z_right in it.product(z_left_terms, z_right_terms)]

    if zhz_debug or False:  # debug prints
        # print all possible pairings
        for a in all_z_permutations:
            old_print_wrapper('Z PAIRING', a)

        # we may not need the unique permutations... unclear at this moment
        if zhz_debug or False:
            unique_z_permutation_list = sorted(list(set(all_z_permutations)))
            for a in unique_z_permutation_list:
                old_print_wrapper('Z TERM1', a)

    return all_z_permutations


def _prepare_eTz_T_terms(eT_series_term):
    """ Factor out T preparation from `_filter_out_valid_eTz_terms`.

    find out what term (in the taylor expansion of e^T) `T_series_term` represents
    set a boolean flag, and wrap the lower order terms in lists so that they have the same
    structure as the s_n case (a list of lists of `general_operator_namedtuple`s)
    """

    # T^0 operator is simply 1 in this case
    if isinstance(eT_series_term, general_operator_namedtuple):
        subheader_log.info("PROCESSING T^0")
        order = 0
        eT_series_term = [[eT_series_term, ], ]  # wrap in a list of lists

    # T^1 operator, straightforward
    elif isinstance(eT_series_term, list) and isinstance(eT_series_term[0], general_operator_namedtuple):
        subheader_log.info("PROCESSING T^1")
        order = 1
        eT_series_term = [[term, ] for term in eT_series_term]  # wrap in a list

    # T^n operator, most complicated
    elif isinstance(eT_series_term, list) and isinstance(eT_series_term[0], list) and len(eT_series_term[0]) >= 2:
        subheader_log.info("PROCESSING T^n")
        order = 'n'
        # no wrapping necessary

    # this shouldn't happen
    else:
        raise Exception(
            'Check `generate_eT_taylor_expansion` and see if the return value'
            'propagates correctly to this location.'
        )

    return eT_series_term, order


def _filter_out_valid_eTz_terms(LHS, eT, H, Z_left, Z_right, total_list, zhz_debug=False):
    """ fill up the `term_list` and `total_list` for the Z^n term
    first we find out what term (in the taylor expansion of e^Z) `z_series_term` represents
    set a boolean flag, and wrap the lower order terms in lists so that they have the same
    structure as the z_n case (a list of lists of `general_operator_namedtuple`s)
    """
    all_z_permutations = _prepare_eTz_z_terms(Z_left, Z_right, zhz_debug)
    eT, eT_order = _prepare_eTz_T_terms(eT)

    # next we process the z operators inside z_term_list
    for h in H.operator_list:

        if True:  # debug
            nof_terms = sum([len(x) for x in eT])
            subheader_log.debug(f"Checking the T^{eT_order} term:")

            if eT_order == 0:
                log.debug(f'\n{tab}'.join([
                    '',
                    f'{"LHS":<12s}{LHS}',
                    f'{"eT":<12s}{eT}',
                    f'{"Z_left":<12s}None',
                    f'{"h":<12s}{h}',
                    f'{"Z_right":<12s}None',
                ]))

            elif nof_terms < 60:
                spread_string = f'\n{tab}{tab}{tab}{tab}'.join([str(x) for x in eT])

                log.debug(f'\n{tab}'.join([
                    '',
                    f'{"LHS":<12s}{LHS}',
                    f'{"eT":<12s}{spread_string}',
                    f'{"Z_left":<12s}None',
                    f'{"h":<12s}{h}',
                    f'{"Z_right":<12s}None',
                ]))
            else:
                log.debug(f'\n{tab}'.join([
                    '',
                    f'{"LHS":<12s}{LHS}',
                    f'number of s terms: {nof_terms}',
                    f'{"Z_left":<12s}None',
                    f'{"h":<12s}{h}',
                    f'{"Z_right":<12s}None',
                ]))

        # valid pairings of s operators given a specific `LHS` and `h`
        valid_permutations = _generate_valid_eT_z_n_operator_permutations(LHS, eT, h, all_z_permutations)

        # if no valid operators continue to the next h
        if valid_permutations == []:
            continue

        if zhz_debug or True:  # debug prints
            for pair in valid_permutations:
                old_print_wrapper('VALID TERM', LHS, pair[0], h, pair[1], pair[2])
                print('VALID TERM', LHS, pair[0], h, pair[1], pair[2])

        log_conf.setLevelDebug(log)
        # we need to generate all possible combinations of
        # each z with the LHS, eT, h operators and the other z
        eT_connection_permutations = _generate_all_o_eT_h_z_connection_permutations(LHS, h, valid_permutations)

        if zhz_debug or True:  # debug prints
            for p in eT_connection_permutations:
                old_print_wrapper('CONNECTED TERMS', LHS, p[0], h, p[1])
                print('CONNECTED TERMS', LHS, p[0], h, p[1])

        # import pdb; pdb.set_trace()
        # continue

        # we need to remove duplicate permutations
        unique_eT_permutations = _remove_duplicate_eT_z_permutations(LHS, h, eT_connection_permutations)
        log_conf.setLevelInfo(log)

        # assert list(eT_connection_permutations) != []

        if True:  # debug prints
            for T, z in unique_eT_permutations:
                old_print_wrapper('UNIQUE TERMS', LHS, eT, h, z)
                print('UNIQUE TERMS', LHS, T, h, z)

        # generate all the explicit connections
        # this also removes all invalid terms
        labeled_permutations = _generate_explicit_eT_z_connections(LHS, h, unique_eT_permutations)

        for i, a in enumerate(labeled_permutations):
            print(
                '-'*20 + f'labeled {i}' + '-'*20,
                a[0],
                a[1],
                a[2],
                a[3],
                sep='\n'
            )
        # we record
        for term in labeled_permutations:
            log.debug(f"{term=}")
            if term[2] != set():
                # if it is not an empty set
                total_list.append(term)
            else:
                old_print_wrapper('exit?')
                sys.exit(0)

        for i, a in enumerate(total_list):
            if i < len(total_list) - 1:
                continue
            print(
                '-'*20 + f'total_list {i}' + '-'*20,
                a[0],
                a[1],
                a[2],
                a[3],
                sep='\n'
            )

    return


def _build_third_eTz_term(LHS, eT_taylor_expansion, H, Z, remove_f_terms=False):
    """
    LHS * (t*t*t) * H * Z

    This one basically needs to be like the t term stuff EXCEPT:
        - there is a single z term
        - it is always on the right side
        - always bond to projection operator in opposite dimension (^i _i)
    """

    valid_term_list = []   # store all valid Omega * (t*t*...t) * h * Z  terms here

    """ First we want to generate a list of valid terms.
    We start with the list of lists `eT_taylor_expansion` which is processed by `_filter_out_valid_eT_terms`.
    This function identifies valid pairings AND places those pairings in the `valid_term_list`.
    Specifically we replace the `general_operator_namedtuple`s with `connected_namedtuple`s and/or
    `disconnected_namedtuple`s.
    """
    for count, eT_series_term in enumerate(eT_taylor_expansion):

        # _filter_out_valid_eT_terms(LHS, eT_series_term, H, Z, valid_term_list, remove_f_terms=remove_f_terms)

        log.setLevel('DEBUG')
        # generate all valid combinations
        _filter_out_valid_eTz_terms(LHS, eT_series_term, H, None, Z, valid_term_list)
        log.setLevel('INFO')

    print('WOPWOOW', valid_term_list)
    # sys.exit(0)

    if valid_term_list == []:
        return ""

    return _prepare_third_eTz_latex(valid_term_list, remove_f_terms=remove_f_terms)


# -------------------------------------------------------------------------------- #

def _generate_eT_z_symmetric_latex_equations(LHS, eT_taylor_expansion, H, Z, only_ground_state=True, remove_f_terms=False):
    """Return a string containing latex code to be placed into a .tex file.
    For a given set of input arguments: (`LHS`, `H`, `Z`) we generate
    all possible and valid CC terms. Note that:
        - `LHS` is an `LHS_namedtuple` object
        - `H` is a `hamiltonian_namedtuple` object
        - `Z` is a `z_operator_namedtuple` object
        - `eT_taylor_expansion` is one of :
            - a single `general_operator_namedtuple`
            - a list of `general_operator_namedtuple`s
            - a list of lists of `general_operator_namedtuple`s

    One possible input could be:
        - `LHS` is the creation operator d
        - `H` is a Hamiltonian of rank two
        - `Z` is the Z operator
        - `eT_taylor_expansion` is the S^1 Taylor expansion term
    """
    return_string = ""

    # the first H term
    return_string += _build_first_z_term(LHS)

    # the second (subtraction) term
    if not only_ground_state:  # If we are acting on the vaccum state then these terms don't exist
        raise Exception('The excited state ZT terms for the 5th ansatz has not been properly implemented')
        return_string += r'\\&-\Big(' + _build_second_z_term(LHS, H, Z, remove_f_terms) + r'\Big)'

    # the third (addition) term
    return_string += r'\\&+\sum\Big(' + _build_third_eTz_term(LHS, eT_taylor_expansion, H, Z, remove_f_terms) + r'\Big)(1-\delta_{cb})'

    # the fourth (subtraction) term
    if not only_ground_state:  # If we are acting on the vaccum state then these terms don't exist
        raise Exception('The excited state ZT terms for the 5th ansatz has not been properly implemented')
        return_string += r'\\&-\sum\Big(' + _build_fourth_z_term(LHS, H, Z, remove_f_terms) + r'\Big)(1-\delta_{db})'

    if only_ground_state:  # If we are acting on the vacuum state then we add these extra terms
        temporary_string = r"\text{all permutations of }\dv{\hat{t}_{\gamma}}{\tau}\hat{z}"
        return_string += r'\\&-i\sum\Big(' + _build_fifth_z_term(LHS, Z) + r'\Big)'

    # remove all empty ^{}/_{} terms that are no longer needed
    return return_string.replace("^{}", "").replace("_{}", "")


def generate_eT_z_t_symmetric_latex(truncations, only_ground_state=True, remove_f_terms=False, path="./generated_latex.tex"):
    """Generates and saves to a file the latex equations for full CC expansion."""

    assert len(truncations) == 5, "truncations argument needs to be tuple of five integers!!"
    maximum_h_rank, maximum_cc_rank, maximum_T_rank, eT_taylor_max_order, omega_max_order = truncations

    master_omega = generate_omega_operator(maximum_cc_rank, omega_max_order)
    raw_H = generate_full_cc_hamiltonian_operator(maximum_h_rank)
    Z = generate_z_operator(maximum_cc_rank, only_ground_state)
    eT_taylor_expansion = generate_eT_taylor_expansion(maximum_T_rank, eT_taylor_max_order)

    """ omega and e^T only generate annihilation operators
        H and Z both generate creation operators
        therefore `rank(e^T)` is restricted to `rank(H) + rank(Z)`
    """
    # assert maximum_eT_rank <= maximum_h_rank + maximum_cc_rank, 'ranks are wrong'

    # aug 4th songhao says this is because of theory
    pruned_list = [term for term in raw_H.operator_list if (term.rank < 3) or (term.m == 0)]
    H = hamiltonian_namedtuple(raw_H.maximum_rank, pruned_list)

    latex_code = ""  # store result in here

    rank_name_list = [
        "0 order", "LINEAR", "QUADRATIC", "CUBIC", "QUARTIC", "QUINTIC", "SEXTIC", "SEPTIC", "OCTIC"
    ]

    for i, omega_term in enumerate(master_omega.operator_list):

        # for debugging purposes
        # if you only want to generate the linear terms for example; change the False to True
        if False and omega_term.rank not in [1, ]:
            continue

        # for the new ansatz v5 we only print the annihilation operator projections
        if omega_term.m > 0:
            continue

        rank_name = rank_name_list[omega_term.rank]

        if omega_term.rank > master_omega.operator_list[i-1].rank:
            # latex_code += '\\newpage\n' if omega_term.rank >= 2 else ''
            latex_code += f'\\paragraph{{{rank_name.capitalize()} Equations}}\n\n'
        else:
            latex_code += r'\vspace{2cm}'

        def _generate_t_lhs(omega):

            upper_label = summation_indices[0:omega.n]
            lower_label = summation_indices[omega.n:omega.rank]
            lower_label += r'\gamma' if lower_label == "" else r',\gamma'

            t_term_latex = f"\\hat{{{bold_t_latex}}}^{{{upper_label}}}_{{{lower_label}}}"

            # the t term we are taking the derivative of
            derivative_latex = rf'i\dv{{{t_term_latex}}}{{\tau}}'

            return derivative_latex

        def _wrap_t_latex(omega, lhs):
            """ Latex commands to wrap around the generated terms `lhs`
            so that the *.tex file compiles correctly.
            """
            omega_string = ""
            for i, char in enumerate(omega.name):
                if char == "d":
                    omega_string += f'\\up{{{summation_indices[i]}}}'
                elif char == "b":
                    omega_string += f'\\down{{{summation_indices[i]}}}'

            omega_string = "1" if omega_string == "" else omega_string

            g_upper_label = f"{summation_indices[0:omega.n]}" if omega.n > 0 else ""
            g_lower_label = f"{summation_indices[omega.n:omega.rank]}" if omega.m > 0 else ""
            g_lower_label += r'\gamma\gamma' if g_lower_label == "" else r', \gamma\gamma'
            g_string = f"\\hat{{G}}^{{{g_upper_label}}}_{{{g_lower_label}}}"

            return (
                '\\begin{equation}\n'
                f'{tab}\\hat{{\\Omega}} = {omega_string}\n'
                r'\qquad\qquad'
                f"\n{tab}{lhs} = {g_string}"
                "\n"
                r'\end{equation}'
                '\n\n'
            )

        # generate latex for t terms
        if True:
            # generate the i(dt/dtau + t*epsilon) latex
            lhs_string = _generate_t_lhs(omega_term)

            # where we do all the work of generating the latex
            # header for the sub section
            latex_code += '%\n%\n%\n%\n%\n\n'
            latex_code += _wrap_t_latex(omega_term, lhs_string)

        def _wrap_z_align_environment(lhs, eqns):
            """ Latex commands to wrap around the generated terms `lhs` and `eqns`
            so that the *.tex file compiles correctly.
            """
            string = (
                '\\begin{align}\\begin{split}\n'
                r'LHS &='
                '\n'
                f"{tab}{lhs}\n"
                r'\\ RHS &='
                '\n%\n%\n'
                f'{eqns}\n'
                r'\end{split}\end{align}'
                '\n\n'
            )
            return string

        def _generate_z_lhs(omega):
            """ quick fix """
            upper_label = summation_indices[0:omega.rank]
            z_term_latex = f"\\hat{{{bold_z_latex}}}^{{{upper_label}}}_{{\\gamma}}"

            # the t term we are taking the derivative of
            derivative_latex = rf'i\Big(\dv{{{z_term_latex}}}{{\tau}}\Big)'
            return derivative_latex

        # generate latex for z terms
        # the omega term is the LHS
        if True:
            # generate the i(dt/dtau + t*epsilon) latex
            lhs_string = _generate_z_lhs(omega_term)

            # where we do all the work of generating the latex
            equations_string = _generate_eT_z_symmetric_latex_equations(
                omega_term, eT_taylor_expansion, H, Z, only_ground_state, remove_f_terms
            )

            # header for the sub section
            latex_code += '%\n%\n%\n%\n%\n\n'
            latex_code += _wrap_z_align_environment(lhs_string, equations_string)

    # write the latex to file
    if only_ground_state:
        # use the predefined header in `reference_latex_headers.py`
        header = headers.ground_state_z_t_symmetric_latex_header
    else:
        # use the predefined header in `reference_latex_headers.py`
        header = headers.full_z_t_symmetric_latex_header

    header += '\\textbf{Note that all terms with a $f$ prefactor have been removed}\n' if remove_f_terms else ''

    # write the new header with latex code attached
    with open(path, 'w') as fp:
        fp.write(header + latex_code + r'\end{document}')

    return


# ----------------------------------------------------------------------------------------------- #
# -------------------------------- Latex of W operators ----------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


# building the latex W operators
t_namedtuple_latex = namedtuple('t_namedtuple_latex', ['m', 'n'])
w_namedtuple_latex = namedtuple('w_latex', ['m', 'n'])


def remove_list_item(i, lst):
    """"""
    return list(item for item in lst if item != i)


def count_items(lis):
    """ create a dictionary recording each different item in the list and number"""
    result = {}
    while lis != []:
        result[lis[0]] = lis.count(lis[0])
        lis = remove_list_item(lis[0], lis)
    return result


def generate_w_prefactor(w_dict):
    """Generates the prefactor for each part of W terms.
    The theory from which the prefactors arise goes as follows:
        - The Taylor series contributes a `1/factorial(length(x))`.
        - Each integer `n` in the tuple contributes `1/factorial(n)`.
    We choose not to print out the 1/factorial(1) prefactors.
    """
    result = "\\frac{1}{"
    for k in w_dict.keys():
        if w_dict[k] == 1:
            continue
        else:
            result += f"{w_dict[k]}!*"
    if result[-1] == "{":
        return ""
    else:
        result = result[:-1] + "}"
    return result


def generate_labels_on_w(order, is_excited=True):
    """ x """
    start = list(range(1, order+1))

    result = []
    list_w = []
    for n in start:
        list_w.append([n, 0])
        if n == 1 or not is_excited:
            continue
        else:
            for i in range(n-1, 0, -1):
                list_w.append([i, n-i])

    list_w = sorted(list_w, key=sum)

    for two in list_w:
        result.append(w_namedtuple_latex(two[0], two[1]))

    return result


def ground_state_w_equations_latex(max_w_order, path="./ground_state_w_equations.tex"):
    """ Generate latex for W equations with ground state only,
        for example: W^1, W^2, W^3...
    """
    latex_code = f"The maximum rank of W operator is: {max_w_order}. The valid W operators are as follows:\n\n"
    w_dict = {}
    spacer = '\n' + r'\\' + '\n'

    w_dict = {n: generate_partitions_of_n(n) for n in range(1, max_w_order+1)}
    # reverse the order of each element
    for v in w_dict.values():
        v.reverse()

    latex_code = r'\begin{align}' + "\n"

    # add the zero'th case
    latex_code += rf'{tab}\bW^{{0}} &= 1'

    for key in w_dict.keys():

        # add the left hand side
        latex_code += rf'{spacer}{tab}\bW^{{{key}}} &= '

        # add the right hand side
        terms = []

        for sub in w_dict[key]:
            item_dict = count_items(list(sub))
            prefactor = generate_w_prefactor(item_dict)
            line = prefactor

            for n in item_dict.keys():
                if item_dict[n] == 1:
                    line += rf"\bt^{{{n}}}"
                else:
                    line += rf"(\bt^{{{n}}})^{item_dict[n]}"

            terms.append(line)

        line = " + ".join(terms)

        # if 2nd order or higher we need to apply a symmetrization operator
        if key > 1:
            line = r'\hat{S}(' + line + ')'

        latex_code += line

    # close the align environment
    latex_code += "\n\\end{align}\n"

    # use the predefined header in `reference_latex_headers.py`
    header = headers.w_equations_latex_header

    # write the new header with latex code attached
    with open(path, 'w') as fp:
        fp.write(header + latex_code + r'\end{document}')

    return latex_code


# ----------------------------------------------------------------------------------------------- #
def generate_t_terms_group(w_ntuple):
    """ Generate the latex code for the LHS (left hand side) of the CC equation.
    The order of the `omega` operator determines all terms on the LHS.
    """
    omega_order = w_ntuple.m + w_ntuple.n

    if omega_order == 0:
        return r'''i\left(\varepsilon\right)'''

    # generate all possible tuples (m, n) representing t terms t^m_n
    single_t_list = [[m, n] for m in range(0, omega_order+1) for n in range(0, omega_order+1) if ((n == m != 0) or (n != m))]

    all_combinations_list = []

    # generate all possible combinations of t^m_n
    # such as t_1, t^2, t^1 * t^1, t^1 * t^2_3, ... etc
    for length in range(1, omega_order+1):
        all_combinations_list.append(list(it.product(single_t_list, repeat=length)))

    """ Next we filter out the combinations that don't match omega.
    Suppose omega is o^2_1:
        - it can match with  (t^1 * t^1 * t_1) or (t^2_1) and so forth
        - it cant match with (t^1 * t^1 * t^1) or (t^1_1) and so forth
    """
    matched_set = set()
    for list_of_t_terms in all_combinations_list:
        for t_terms in list_of_t_terms:
            upper_sum = sum([t[0] for t in t_terms])  # the sum over all m superscripts (t^m)
            lower_sum = sum([t[1] for t in t_terms])  # the sum over all n superscripts (t_n)

            # remember that a t_1 contracts with o^1
            # so the `lower_sum` needs to be compared to o^n
            if w_ntuple.n == lower_sum and w_ntuple.m == upper_sum:
                """ The "filtering" is accomplished by adding sorted tuples of tuples to a set.
                We cannot use lists because sets require hashable elements (immutable) such as tuples.
                However with tuples, we can end up with duplicates like ((2, 0), (0, 1)) and ((0, 1), (2, 0)).
                So we have to:
                    - generate lists of tuples:             [(0, 1), (0, 2)]
                    - sort those lists in reverse order:    [(0, 2), (0, 1)]
                    - make a tuple from the list:           ((0, 2), (0, 1))
                    - add the tuple to `matched_set`
                """
                matched_set.add(tuple(sorted([tuple(x) for x in t_terms], reverse=True)))

    """ Transform the set into a list of lists sort by increasing length
    Two examples:
        - omega is `operator(name='bb', m=0, n=2)`
        then `sorted_list` is [[(2, 0)], [(1, 0), (1, 0)]]

        - omega is `operator(name='ddd', m=3, n=0)`
        then `sorted_list` is [[(0, 3)], [(0, 2), (0, 1)], [(0, 1), (0, 1), (0, 1)]]
    """
    sorted_list = sorted([[b for b in a] for a in matched_set], key=len)

    return sorted_list


def excited_state_w_equations_latex(max_w_order, path="./thermal_w_equations.tex"):
    """ Generate latex for W equations with excited states,
        for example: W^1_1, W^1_2...
    """
    latex_code = "These are the W operators in full VECC.\\\\\n%"  # store result in here
    w_lable = generate_labels_on_w(max_w_order, is_excited=True)
    w_dict = {}

    for w in w_lable:
        g = generate_t_terms_group(w)
        g.reverse()
        w_dict[(w.m, w.n)] = g

    latex_code += f"\nThe maximum rank of W operator is: {max_w_order}\\\\ \n\n\\begin{{equation}}\n\n"
    latex_code += f"{tab}\\textbf{{W}}^{0} = 1 \\\\"
    latex_code += "\n\n"
    for key in w_dict.keys():
        if key[0] == 1 and key[1] == 0:
            latex_code += f"{tab}\\textbf{{W}}^{1} = \\bt^{1} \\\\"
            latex_code += "\n\n"
            continue

        if key[1] == 0:
            latex_code += f"{tab}\\textbf{{W}}^{{{key[0]}}} = \\hat{{S}}("
        else:
            latex_code += f"{tab}\\textbf{{W}}^{{{key[0]}}}_{{{key[1]}}} = \\hat{{S}}("

        for sub in w_dict[key]:
            if len(sub) == 1 and sub[0][1] != 0:
                continue
            item_dict = count_items(list(sub))
            #  old_print_wrapper("------------item_dict-------------")
            #  old_print_wrapper(item_dict)
            prefactor = generate_w_prefactor(item_dict)
            latex_code += prefactor
            for n in item_dict.keys():
                power = ""
                if item_dict[n] == 1:
                    power = ""
                else:
                    latex_code += "("
                    power += f")^{item_dict[n]}"

                if n[0] == 0:
                    latex_code += f"\\bt_{{{n[1]}}}"
                elif n[1] == 0:
                    latex_code += f"\\bt^{{{n[0]}}}"
                else:
                    latex_code += f"\\bt^{{{n[0]}}}_{{{n[1]}}}"
                latex_code += power
            latex_code += " + "
        latex_code = latex_code[:-3] + ")\\\\ \n\n"
    latex_code += "\\end{equation}\n"

    # if file already exists then update it
    if os.path.isfile(path):

        # read the entire file contents
        with open(path, 'r') as fp:
            file_contents = fp.readlines()

        # keep only the header
        header = ''.join(file_contents[0:29])

        # write the new header with latex code attached
        with open(path, 'w') as fp:
            fp.write(header + latex_code + r'\end{document}')

    # otherwise write a new file
    else:
        with open(path, 'w') as fp:
            fp.write(latex_code)

    return


# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
def generate_latex_files(truncations, only_ground_state=True, remove_f_terms=False, thermal=False, file=None):
    """ only generates .tex files to be compiled into pdf files """

    if file == 'full cc':
        if only_ground_state:
            generate_full_cc_latex(truncations, only_ground_state=True, path="./ground_state_full_cc_equations.tex")
        else:
            generate_full_cc_latex(truncations, only_ground_state=False, path="./full_cc_equations.tex")

    # this doesn't care about the truncation numbers
    elif file == 'w equations':

        """
        eventually we want to merge both the
        `ground_state_w_equations_latex`
        and the
        `excited_state_w_equations_latex`
        functions
        """

        max_w_order = 5  # this is the

        if only_ground_state:
            path = "./ground_state_w_equations.tex"
            ground_state_w_equations_latex(max_w_order, path)
        else:
            path = "./excited_state_w_equations.tex"
            assert False, 'the excited_state_w_equations_latex has not been verified'
            excited_state_w_equations_latex(max_w_order, path)

    # the `s_taylor_max_order` isn't relevant for this execution pathway
    elif file == 'z_t ansatz':
        f_term_string = "_no_f_terms" if remove_f_terms else ''

        if only_ground_state:
            path = f"./ground_state_z_t_symmetric_equations{f_term_string}.tex"
        else:
            path = f"./z_t_symmetric_equations{f_term_string}.tex"

        generate_z_t_symmetric_latex(truncations, only_ground_state, remove_f_terms, path)

    # the `s_taylor_max_order` isn't relevant for this execution pathway
    elif file == 'eT_z_t ansatz':
        f_term_string = "_no_f_terms" if remove_f_terms else ''

        if only_ground_state:
            path = f"./ground_state_eT_z_t_symmetric_equations{f_term_string}.tex"
        else:
            path = f"./eT_z_t_symmetric_equations{f_term_string}.tex"

        generate_eT_z_t_symmetric_latex(truncations, only_ground_state, remove_f_terms, path)

    else:
        raise Exception(f"Wrong file type specified in {file=}")

    return


def generate_python_files(truncations, only_ground_state=True, thermal=False):
    """ generates .py files which will be used when calculating desired quantities """

    if only_ground_state:
        generate_full_cc_python(truncations, only_ground_state=True)
    else:
        generate_full_cc_python(truncations, only_ground_state=False)

    max_residual_order = 6
    generate_residual_equations_file(max_residual_order, maximum_h_rank)
    max_w_order = 6
    generate_w_operator_equations_file(max_w_order)
    dt_order = 6
    generate_dt_amplitude_equations_file(dt_order)
    return


if (__name__ == '__main__'):
    import log_conf

    # for now make a second argument the filepath for logging output
    if len(sys.argv) > 1:
        logging_output_filename = str(sys.argv[1])
        log = log_conf.get_filebased_logger(logging_output_filename)
    else:
        log = log_conf.get_stdout_logger()

    header_log = log_conf.HeaderAdapter(log, {})
    subheader_log = log_conf.SubHeaderAdapter(log, {})

    # dump_all_stdout_to_devnull()   # calling this removes all prints / logs from stdout
    # log.setLevel('CRITICAL')

    # maximum_h_rank = 4
    # maximum_cc_rank = 4
    # s_taylor_max_order = 4  # this doesn't matter for the Z ansatz
    # omega_max_order = 4

    # # Z ansatz
    # truncations = maximum_h_rank, maximum_cc_rank, s_taylor_max_order, omega_max_order

    maximum_h_rank = 2
    maximum_cc_rank = 2
    maximum_T_rank = 2
    eT_taylor_max_order = 2
    omega_max_order = 2

    # need to have truncation of e^T
    eT_z_t_truncations = maximum_h_rank, maximum_cc_rank, maximum_T_rank, eT_taylor_max_order, omega_max_order

    generate_latex_files(
        eT_z_t_truncations,
        only_ground_state=True,
        remove_f_terms=False,
        thermal=False,
        file='eT_z_t ansatz'
    )
    # generate_python_files(truncations, only_ground_state=True, thermal=False)
    print("We reached the end of main")
