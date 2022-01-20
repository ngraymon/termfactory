# system imports
from collections import namedtuple
import itertools as it
# third party imports

# local imports
from latex_defines import *
import reference_latex_headers as headers
from common_imports import tab, summation_indices, unlinked_indices, old_print_wrapper
from namedtuple_defines import (
    general_operator_namedtuple,
    hamiltonian_namedtuple,
    omega_namedtuple,
    connected_namedtuple,
    disconnected_namedtuple,
)

# temp

# temp logging fix
import log_conf

log = log_conf.get_filebased_logger(f'{__name__}.txt', submodule_name=__name__)
header_log = log_conf.HeaderAdapter(log, {})
subheader_log = log_conf.SubHeaderAdapter(log, {})


# ----------------------------------------------------------------------------------------------- #
# --------------------------------  GENERATING FULL CC LATEX  ----------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# this serves the same purpose as `general_operator_namedtuple`
# we changed the typename from "operator" to "h_operator" to distinguish the specific role and improve debugging
h_operator_namedtuple = namedtuple('h_operator', ['rank', 'm', 'n'])


# these serve the same purpose as `hamiltonian_namedtuple`
# we changed the typename from "operator" to distinguish the specific role and improve debugging
s_operator_namedtuple = namedtuple('s_operator', ['maximum_rank', 'operator_list'])
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

    if s_taylor_max_order >= 1:
        s_taylor_expansion[1] = S.operator_list                        # S term

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
    for trunc in truncations:
        assert trunc >= 1, "Truncations need to be positive integers"
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
