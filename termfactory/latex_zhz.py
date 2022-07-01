# system imports
import math
from fractions import Fraction
import itertools as it
from collections import namedtuple

# third party imports

# local imports
from latex_defines import *
from common_imports import tab, z_summation_indices, z_unlinked_indices, summation_indices, old_print_wrapper
from latex_full_cc import generate_omega_operator, generate_full_cc_hamiltonian_operator, _omega_joining_with_itself, _h_joining_with_itself, disconnected_namedtuple
from namedtuple_defines import general_operator_namedtuple, hamiltonian_namedtuple
import reference_latex_headers as headers
from truncations import _verify_fcc_truncations
from truncation_keys import TruncationsKeys as tkeys
import log_conf

log = log_conf.get_filebased_logger(f'{__name__}.txt', submodule_name=__name__)
header_log = log_conf.HeaderAdapter(log, {})
subheader_log = log_conf.SubHeaderAdapter(log, {})
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
    # TODO trigger these cases in tests
    if (required_b_for_left_z > available_b_for_left_z):  # pragma: no cover
        return True

    if (required_d_for_left_z > available_d_for_left_z):  # pragma: no cover
        return True

    if (required_b_for_right_z > available_b_for_right_z):  # pragma: no cover
        return True

    if (required_d_for_right_z > available_d_for_right_z):  # pragma: no cover
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

        elif log_invalid:  # pragma: no cover
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

        elif log_invalid:  # pragma: no cover
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
    # log_conf.setLevelDebug(log)
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
                else:  # pragma: no cover
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

                if right_z.name is None:  # pragma: no cover
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
                if (z_left_kwargs != {}) and (z_right_kwargs != {}):  # pragma: no cover
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
    # log_conf.setLevelInfo(log)

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
        elif z_right is None:  # pragma: no cover
            assert z_left_exists
        else:  # pragma: no cover
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
            else: #pragma: no cover
                old_print_wrapper('exit?')
                sys.exit(0)

    return


# --------------- assigning of upper/lower latex indices ------------------------- #
def _build_left_z_term(z_left, h, color=True):  # pragma: no cover
    # not in coverage for now, only used in unfinished excited state
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
def _build_z_latex_prefactor(h, t_list, simplify_flag=True): # pragma: no cover
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
        if print_prefactors:  # pragma: no cover
            raise NotImplementedError("prefactor code for z stuff is not done")
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

    # if the line is so short we don't need to split, so far never seems to have long lines
    if len(return_list) < split_width*2:
        return f"({' + '.join(return_list)})"

    split_equation_list = []  # pragma: no cover
    for i in range(0, len(return_list) // split_width):  # pragma: no cover
        split_equation_list.append(' + '.join(return_list[i*split_width:(i+1)*split_width]))  # pragma: no cover

    # make sure we pickup the last few terms
    last_few_terms = (len(return_list) % split_width)-split_width+1  # pragma: no cover
    split_equation_list.append(' + '.join(return_list[last_few_terms:]))  # pragma: no cover

    # join the lists with the equation splitting string
    splitting_string = r'\\  &+  % split long equation'  # pragma: no cover
    final_string = f"\n{tab}{splitting_string}\n".join(split_equation_list)  # pragma: no cover

    # and we're done!
    return f"(\n{final_string}\n)"  # pragma: no cover


def _build_hz_latex_prefactor(h, z_left, z_right, simplify_flag=False):
    """Attempt to return latex code representing appropriate prefactor term.

    In general the rules are:
        - h                contributes 1/m! 1/n!
        - the right z term contribute 1/m! 1/n!
        - multiply by (n choose n_k) where n_k are the number of internal labels of z right

    By convention we account for internal permutations using z, not h.
    What this means is that if we have h^2 z_2
    - we consult Z
        -> multiply by 2 because we have two permutations of labels

    whereas if we ALSO multiplied by 2 considering the permutations on h then we would have overcounted the permutations
    consider the following four situations
    - h^{ij} z_{ij}
    - h^{ij} z_{ji}
    - h^{ji} z_{ij}
    - h^{ji} z_{ji}
    we have now permuted 2 for h and 2 for z, however the second term and the third term are identical and the fourth and first terms are identical

    Therefore the *current* convention is that we ignore internal permutations on h.
    Note that if we want to extend this to e^T * HZ then we must rejigger the logic when we add t2 terms back in.
    Since if we had a term like
    - t^{ij} h_{ijk} z^{k}
    then we would NOT count the `ij` permutation correctly using the current logic.

    Although i think a simple solution would be to subtract the magnitude of the contracts with z right from the combinatorial in h?
    Might work?

    """

    if z_left is not None:
        raise NotImplementedError('Logic does not support ZH, or ZHZ terms at the moment')

    numerator_value = 1
    denominator_value = 1

    numerator_list, denominator_list = [], []
    # ---------------------------------------------------------------------------------------------------------

    if h.m > 1:
        # by definition
        denominator_value *= math.factorial(h.m)
        denominator_list.append(f'{h.m}!')
        # to account for the permutations of internal labels around the external labels
        number = math.comb(h.m, h.m_r)
        if number > 1:
            numerator_value *= number
            numerator_list.append(f'{number}')

    if h.n > 1:
        # by definition
        denominator_value *= math.factorial(h.n)
        denominator_list.append(f'{h.n}!')
        # to account for the permutations of internal labels around the external labels
        number = math.comb(h.n, h.n_r)
        if number > 1:
            numerator_value *= number
            numerator_list.append(f'{number}')

    if z_right.m > 1:
        # by definition
        denominator_value *= math.factorial(z_right.m)
        denominator_list.append(f'{z_right.m}!')

        # to account for the permutations of external labels
        number = math.comb(z_right.m, z_right.m_lhs)
        if number > 1:
            numerator_value *= number
            numerator_list.append(f'{number}')

        # to account for the permutations of internal labels
        if z_right.m_h > 1:
            numerator_value *= math.factorial(z_right.m_h)
            numerator_list.append(f'{z_right.m_h}!')

    if z_right.n > 1:
        # by definition
        denominator_value *= math.factorial(z_right.n)
        denominator_list.append(f'{z_right.n}!')

        # to account for the permutations of external labels
        number = math.comb(z_right.n, z_right.n_lhs)
        if number > 1:
            numerator_value *= number
            numerator_list.append(f'{number}')

        # to account for the permutations of internal labels
        if z_right.n_h > 1:
            numerator_value *= math.factorial(z_right.n_h)
            numerator_list.append(f'{z_right.n_h}!')

    # ---------------------------------------------------------------------------------------------------------

    # old_print_wrapper(numerator_value)
    # old_print_wrapper(denominator_value)

    # # simplify
    if simplify_flag:
        fraction = Fraction(numerator_value, denominator_value)
        # old_print_wrapper(f"{h = }")
        # old_print_wrapper(f"{z_right = }")
        # old_print_wrapper(f"{fraction.numerator = }")
        # old_print_wrapper(f"{fraction.denominator = }")

        # import pdb; pdb.set_trace()
        if fraction == 1:
            return ''
        else:
            return f"\\frac{{{fraction.numerator}}}{{{fraction.denominator}}}"

        # numerator_list, denominator_list = _simplify_full_cc_python_prefactor(numerator_list, denominator_list)

    # glue the numerator and denominator together
    numerator_string = '1' if (numerator_list == []) else f"{'*'.join(numerator_list)}"
    denominator_string = '1' if (denominator_list == []) else f"{''.join(denominator_list)}"

    if numerator_string == '1' and denominator_string == '1':
        return ''
    else:
        return f"\\frac{{{numerator_string}}}{{{denominator_string}}}"


def _prepare_third_z_latex(term_list, split_width=7, remove_f_terms=False, print_prefactors=True):
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
        if print_prefactors: # pragma: no cover
            term_string += _build_hz_latex_prefactor(h, None, z_right)

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

    split_equation_list = []  # pragma: no cover
    for i in range(0, len(return_list) // split_width):  # pragma: no cover
        split_equation_list.append(' + '.join(return_list[i*split_width:(i+1)*split_width]))  # pragma: no cover

    # make sure we pickup the last few terms
    last_few_terms = (len(return_list) % split_width)-split_width+1  # pragma: no cover
    split_equation_list.append(' + '.join(return_list[last_few_terms:]))  # pragma: no cover

    # join the lists with the equation splitting string
    splitting_string = r'\\  &+  % split long equation'  # pragma: no cover
    final_string = f"\n{tab}{splitting_string}\n".join(split_equation_list)  # pragma: no cover

    # and we're done!
    return f"(\n{final_string}\n"  # pragma: no cover


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
        if print_prefactors:  # pragma: no cover
            raise NotImplementedError("prefactor code for z stuff is not done")
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
    """ LHS *  H
    The first ZHZ permutation has no Z terms and is therefore quite simple
    """
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
    """ LHS * Z * H
    This one basically needs to be like the t term stuff EXCEPT:
        - there is a single z term
        - it is always on the left side
        - always bond to projection operator in same dimension (^i ^i)
    """
    valid_term_list = []

    # generate all valid combinations
    _filter_out_valid_z_terms(LHS, H, Z, None, valid_term_list)

    if valid_term_list == []:  # pragma: no cover
        return ""

    return _prepare_second_z_latex(valid_term_list, remove_f_terms=remove_f_terms)


def _build_third_z_term(LHS, H, Z, remove_f_terms=False):
    """ LHS * H * Z
    This one basically needs to be like the t term stuff EXCEPT:
        - there is a `single z term
        - it is always on the right side
        - always bond to projection operator in opposite dimension (^i _i)
    """
    valid_term_list = []

    # generate all valid combinations
    _filter_out_valid_z_terms(LHS, H, None, Z, valid_term_list)

    if valid_term_list == []:  # pragma: no cover
        return ""

    return _prepare_third_z_latex(valid_term_list, remove_f_terms=remove_f_terms)


def _build_fourth_z_term(LHS, H, Z, remove_f_terms=False):
    """ LHS * Z * H * Z
    This one basically needs to be like the t term stuff EXCEPT:
        - there is always two z terms
        - one on each side
        - all combinations are considered here
    """
    valid_term_list = []

    # generate all valid combinations
    _filter_out_valid_z_terms(LHS, H, Z, Z, valid_term_list)

    if valid_term_list == []:  # pragma: no cover
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

    not_implemented_yet_message = (
        "The logic for the supporting functions (such as `_filter_out_valid_z_terms` and others)\n"
        "Has only been verified to work for the LHS * H * Z (`third_z`) case.\n"
        "The code may produce some output without halting, but the output is meaningless from a theory standpoint.\n"
        "Do not remove this Exception without consulting with someone else and implementing the requisite functions."
    )

    # the second (subtraction) term
    if not only_ground_state:  # If we are acting on the vaccum state then these terms don't exist pragma: no cover
        raise NotImplementedError(not_implemented_yet_message)
        return_string += r'\\&-\Big(' + _build_second_z_term(LHS, H, Z, remove_f_terms) + r'\Big)'

    # the third (addition) term
    return_string += r'\\&+\sum\Big(' + _build_third_z_term(LHS, H, Z, remove_f_terms) + r'\Big)(1-\delta_{cb})'

    # the fourth (subtraction) term
    if not only_ground_state:  # If we are acting on the vaccum state then these terms don't exist pragma: no cover
        raise NotImplementedError(not_implemented_yet_message)
        return_string += r'\\&-\sum\Big(' + _build_fourth_z_term(LHS, H, Z, remove_f_terms) + r'\Big)(1-\delta_{db})'

    if only_ground_state:  # If we are acting on the vacuum state then we add these extra terms
        temporary_string = r"\text{all permutations of }\dv{\hat{t}_{\gamma}}{\tau}\hat{z}"
        return_string += r'\\&-i\sum\Big(' + _build_fifth_z_term(LHS, Z) + r'\Big)'

    # remove all empty ^{}/_{} terms that are no longer needed
    return return_string.replace("^{}", "").replace("_{}", "")
    # return r'%'


def generate_z_t_symmetric_latex(truncations, **kwargs):
    """Generates and saves to a file the latex equations for full CC expansion."""

    # unpack kwargs
    only_ground_state = kwargs['only_ground_state']
    remove_f_terms = kwargs['remove_f_terms']
    path = kwargs['path']

    # unpack truncations
    maximum_h_rank = truncations[tkeys.H]
    maximum_cc_rank = truncations[tkeys.CC]
    omega_max_order = truncations[tkeys.P]

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

        # apparently marcel wants the annihilation operator projects added back in?

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
                if char == "d":  # pragma: no cover
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
    else:  # pragma: no cover
        # use the predefined header in `reference_latex_headers.py`
        header = headers.full_z_t_symmetric_latex_header

    header += '\\textbf{Note that all terms with a $f$ prefactor have been removed}\n' if remove_f_terms else ''

    # write the new header with latex code attached
    with open(path, 'w') as fp:
        fp.write(header + latex_code + r'\end{document}')

    return
