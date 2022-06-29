# system imports
import itertools as it
import functools
import inspect
import math
from fractions import Fraction
import pdb

# third party imports

# local imports
import helper_funcs
from helper_funcs import unique_permutations, named_line
from namedtuple_defines import general_operator_namedtuple, omega_namedtuple
from common_imports import tab, tab_length, summation_indices, unlinked_indices, old_print_wrapper
from latex_eT_zhz import (
    generate_eT_taylor_expansion,
    generate_pruned_H_operator,
    generate_z_operator,
    _filter_out_valid_eTz_terms,
)
from latex_full_cc import generate_omega_operator
from namedtuple_defines import disconnected_namedtuple
from code_w_equations import taylor_series_order_tag, hamiltonian_order_tag
import code_import_statements_module
from truncations import _verify_eT_z_t_truncations
from truncation_keys import TruncationsKeys as tkeys

# temp logging fix
import log_conf

log = log_conf.get_filebased_logger('output.txt', submodule_name=__name__)
header_log = log_conf.HeaderAdapter(log, {})
subheader_log = log_conf.SubHeaderAdapter(log, {})

##########################################################################################
# Defines for labels and spacing

s1, s2 = 75, 28
l1, l2 = 109, 45

spaced_named_line = functools.partial(helper_funcs.spaced_named_line, spacing_line=f"# {'-'*s1} #\n")
long_spaced_named_line = functools.partial(helper_funcs.long_spaced_named_line, large_spacing_line=f"# {'-'*l1} #\n")
##########################################################################################

# ----------------------------------------------------------------------------------------------- #
# ----------------------  GENERATING FULL e^T *  H * Z PYTHON EQUATIONS  ------------------------ #
# ----------------------------------------------------------------------------------------------- #


def _eT_zhz_einsum_electronic_components(t_list, z_right, b_loop_flag=False):
    """ Return a list of strings to be used in a numpy.einsum() call.

    For now this function assumes that `b_loop_flag` is always true.
    It is unclear what the function should produce when it is False.

    If `b_loop_flag` is True then we instead treat each of the t terms as NOT
    having an electronic label, the Z term as only having one electronic d.o.f
    and the H term as having the same two electronic d.o.f Such as
        `ac, c -> a`

    """
    assert b_loop_flag is True, 'Unclear how to implement function for vectorized mode'

    electronic_components = []

    # special b loop case
    if b_loop_flag:

        # treat the t terms as having no electronic labels
        electronic_components += ['', ] * len(t_list)

        # the H term always has 2
        electronic_components.append('ac')

        # we assume Z always contributes
        electronic_components.append('c')

        return electronic_components

    # otherwise
    else:  # pragma: vectorize_b_loop

        # for each t term add 1 electronic label
        electronic_components += ['a', ] * len(t_list)

        # the H term always has 2
        electronic_components.append('ac')

        # we assume Z always contributes
        electronic_components.append('c')

        return electronic_components


def _eT_zhz_einsum_electronic_components_lhs(t_list, dT, z_right, b_loop_flag=False):
    """ Return a list of strings to be used in a numpy.einsum() call.

    For now this function assumes that `b_loop_flag` is always true.
    It is unclear what the function should produce when it is False.

    If `b_loop_flag` is True then we instead treat each of the t terms as NOT
    having an electronic label, the Z term as only having one electronic d.o.f
    and the dT term as having the same two electronic d.o.f Such as
        `ac, c -> a`

    """
    assert b_loop_flag is True, 'Unclear how to implement function for vectorized mode'

    electronic_components = []

    # special b loop case
    if b_loop_flag:

        # treat the t terms as having no electronic labels
        electronic_components += ['', ] * (len(t_list))

        # the dT term has no electronic components
        electronic_components.append('')

        # Z always contributes
        electronic_components.append('c')

        return electronic_components

    # otherwise
    else:  # pragma: vectorize_b_loop

        # for each t term add 1 electronic label
        electronic_components += ['a', ] * len(t_list)

        # the dT term has 1 electronic components
        electronic_components.append('a')

        # we assume Z always contributes
        electronic_components.append('c')

        return electronic_components
# ----------------------------------------------------------------------------------------------- #
# handle all the vibrational einsum string component


def _build_z_term_python_labels(z_right, offset_dict):
    """ x """

    sum_label, unlinked_label = "", ""

    # first we add the predetermined labels from any t operators
    sum_label += offset_dict['upper_z_indices'] + offset_dict['lower_z_indices']

    # simple zero case
    if z_right.rank == 0:
        assert sum_label == '', 'just in case we got some logic wrong'
        return (sum_label, unlinked_label), offset_dict

    # superscript indices
    if (z_right.m > 0):
        b = offset_dict['unlinked_count']

        # record the characters we will place on the h term
        unlinked_label += unlinked_indices[b:b + z_right.m_lhs]

        # record the change in the offset
        offset_dict['unlinked_count'] += z_right.m_lhs

    # subscript indices
    if (z_right.n > 0):   # pragma: hot_bands_or_thermal
        b = offset_dict['unlinked_count']

        # record the characters we will place on the h term
        unlinked_label += unlinked_indices[b:b + z_right.n_lhs]

        # record the change in the offset
        offset_dict['unlinked_count'] += z_right.n_lhs

    return (sum_label, unlinked_label), offset_dict


def _build_h_term_python_labels(h, offset_dict):
    """ x """

    sum_label, unlinked_label = '', ''

    # first we add the predetermined labels from any t operators
    sum_label += offset_dict['upper_h_indices'] + offset_dict['lower_h_indices']

    # simple zero case
    if h.rank == 0:
        assert sum_label == '', 'just in case we got some logic wrong'
        return (sum_label, unlinked_label), offset_dict

    # superscript indices
    if (h.m > 0):
        a, b = offset_dict['summation_count'], offset_dict['unlinked_count']

        # determine the summation indices for contractions with Z_right
        z_str = summation_indices[a:a + h.m_r]

        # record the characters we will need to place on z
        offset_dict['lower_z_indices'] += z_str

        # record the characters we will place on the h term
        sum_label += z_str
        unlinked_label += unlinked_indices[b:b + h.m_lhs]

        # record the change in the offset
        offset_dict['summation_count'] += h.m_r
        offset_dict['unlinked_count'] += h.m_lhs

    # subscript indices
    if (h.n > 0):
        a, b = offset_dict['summation_count'], offset_dict['unlinked_count']

        # determine the summation indices for contractions with Z_right
        z_str = summation_indices[a:a + h.n_r]

        # record the characters we will need to place on z
        offset_dict['upper_z_indices'] += z_str

        # record the characters we will place on the h term
        sum_label += z_str
        unlinked_label += unlinked_indices[b:b + h.n_lhs]

        # record the change in the offset
        offset_dict['summation_count'] += h.n_r
        offset_dict['unlinked_count'] += h.n_lhs

    return (sum_label, unlinked_label), offset_dict


def _build_t_term_python_labels(term, offset_dict):
    """ x """

    sum_label, unlinked_label = "", ""

    # subscript indices
    if (term.n > 0):
        a, b = offset_dict['summation_count'], offset_dict['unlinked_count']

        # determine the summation indices
        h_stop = a + term.n_h
        h_str = summation_indices[a:h_stop]
        z_str = summation_indices[h_stop:h_stop + term.n_r]

        # record the characters we will need to place on h and z
        offset_dict['upper_h_indices'] += h_str
        offset_dict['upper_z_indices'] += z_str

        # record the characters we will place on this specific t term
        sum_label += h_str + z_str
        unlinked_label += unlinked_indices[b:b + term.n_lhs]

        # record the change in the offset
        offset_dict['summation_count'] += term.n_h + term.n_r
        offset_dict['unlinked_count'] += term.n_lhs

    # superscript indices
    if (term.m > 0):  # pragma: hot_bands_or_thermal
        a, b = offset_dict['summation_count'], offset_dict['unlinked_count']

        # determine the summation indices
        h_stop = a + term.m_h
        h_str = summation_indices[a:h_stop]
        z_str = summation_indices[h_stop:h_stop + term.m_r]

        # record the characters we will need to place on h and z
        offset_dict['lower_h_indices'] += h_str
        offset_dict['lower_z_indices'] += z_str

        # record the characters we will place on this specific t term
        sum_label += h_str + z_str
        unlinked_label += unlinked_indices[b:b + term.m_lhs]

        # record the change in the offset
        offset_dict['summation_count'] += term.m_h + term.m_r
        offset_dict['unlinked_count'] += term.m_lhs

    return sum_label, unlinked_label


def _build_t_term_python_group(t_list, h, z_right):
    """ x """

    sum_list, unlinked_list = [], []

    offset_dict = {
        'summation_count': 0, 'unlinked_count': 0,
        'upper_h_indices': '', 'lower_h_indices': '',
        'upper_z_indices': '', 'lower_z_indices': '',
    }

    for t in t_list:
        sum_label, unlinked_label = _build_t_term_python_labels(t, offset_dict)
        log.debug(t, sum_label, unlinked_label)
        sum_list.append(sum_label)
        unlinked_list.append(unlinked_label)

    # log.info(offset_dict)
    # old_print_wrapper('\n\n\n\n\n')
    # old_print_wrapper(f"{offset_dict = }")
    # import pdb; pdb.set_trace()

    return sum_list, unlinked_list, offset_dict


def _eT_zhz_einsum_vibrational_components(t_list, h, z_right, b_loop_flag=False):
    """ Return two lists of strings to be used in a numpy.einsum() call.

    The first list is all the vibrational components of each term (t, h, z),
    to be traced over; which will be appended to the electronic components.
    The second is the leftover indices, which indicated external labels; these
    will be appended to the output string.

    Each string is paired with a t, h, or z term.
    The default is to assume all terms have two electronic degrees of freedom.
    We also assume that we want the final shape to have electronic dimensions `ab`.
    Therefore we start with `ac` and simply iterate over `cdefgh` like so:
        `ac, cd, de, ef, fg, gh, hb -> ab`
    or
        `ac, cd, db -> ab`
    and so clearly the current implementation only supports up to 7 terms.

    If `b_loop_flag` is True then we instead treat each of the t terms as NOT
    having an electronic label, the Z term as only having one electronic d.o.f
    and the H term as having the same two electronic d.o.f Such as
        `ac, c -> a`

    """

    vibrational_components = []  # store return values here

    old_print_wrapper(t_list, h, z_right)

    # add t term vibrational components
    alist, blist, offset_dict = _build_t_term_python_group(t_list, h, z_right)
    for i in range(len(alist)):
        vibrational_components.append(alist[i] + blist[i])

    # add h term vibrational components
    h_labels, offset_dict = _build_h_term_python_labels(h, offset_dict)
    vibrational_components.append(h_labels[0] + h_labels[1])

    # add z term vibrational components
    z_labels, offset_dict = _build_z_term_python_labels(z_right, offset_dict)
    vibrational_components.append(z_labels[0] + z_labels[1])

    # remaining term lists
    remaining_list = [r for r in blist] + [h_labels[1], z_labels[1], ]

    return vibrational_components, ''.join(remaining_list)


def _eT_zhz_einsum_vibrational_components_lhs(t_list, h, z_right, b_loop_flag=False):
    """ Return two lists of strings to be used in a numpy.einsum() call.

    The first list is all the vibrational components of each term (t_conj, dT, z or dz),
    (dz if no dT) to be traced over; which will be appended to the electronic components.
    The second is the leftover indices, which indicated external labels; these
    will be appended to the output string.

    Each string is paired with a t_conj, dT, or z / dz term.
    The default is to assume all terms have two electronic degrees of freedom.
    We also assume that we want the final shape to have electronic dimensions `ab`.
    Therefore we start with `ac` and simply iterate over `cdefgh` like so:              ?still correct?
        `ac, cd, de, ef, fg, gh, hb -> ab`
    or
        `ac, cd, db -> ab`
    and so clearly the current implementation only supports up to 7 terms.

    """

    vibrational_components = []  # store return values here

    old_print_wrapper(t_list, h, z_right)

    # add t_conj term vibrational components
    alist, blist, offset_dict = _build_t_term_python_group(t_list, h, z_right)
    for i in range(len(alist)):
        vibrational_components.append(alist[i] + blist[i])

    # add dT term vibrational components
    h_labels, offset_dict = _build_h_term_python_labels(h, offset_dict)
    vibrational_components.append(h_labels[0] + h_labels[1])

    # add z / dz term vibrational components
    z_labels, offset_dict = _build_z_term_python_labels(z_right, offset_dict)
    vibrational_components.append(z_labels[0] + z_labels[1])

    # remaining term lists
    remaining_list = [r for r in blist] + [h_labels[1], z_labels[1], ]
    print('remaining_list= ', remaining_list)
    return vibrational_components, ''.join(remaining_list)
# ----------------------------------------------------------------------------------------------- #
# old and unused


def _eT_zhz_einsum_subscript_generator(h, t_list):  # pragma: no cover
    """ x """

    # assert lhs_flag in ['lhs', 'rhs']
    return_string = ""

    # if Proj:
    #     electronic_components = _lhs_einsum_electronic_components(t_list)
    # else:
    electronic_components = _eT_zhz_einsum_electronic_components(t_list)

    vibrational_components, remaining_indices = _eT_zhz_einsum_vibrational_components(h, t_list)

    summation_subscripts = ", ".join([
        f"{electronic_components[i]}{vibrational_components[i]}" for i in range(len(electronic_components))
    ])

    return_string = f"{summation_subscripts} -> ab{remaining_indices}"

    return return_string


def _eT_zhz_einsum_prefactor(term):  # pragma: no cover
    """ x """
    string = "0.69"

    return string
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# helper functions for _write_third_eTz_einsum_python


# ----------------------------------------------------------------------------------------------- #
# all four are dealing with prefactor determination


def _simplify_eT_zhz_python_prefactor(numerator_list, denominator_list):
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

    # count number of appearances
    numerator_dict = dict([(key, 0) for key in numerator_set])
    for string in numerator_list:
        numerator_dict[string] += 1

    log.debug('nnnn', numerator_dict)

    denominator_dict = dict([(key, 0) for key in denominator_set])
    for string in denominator_list:
        denominator_dict[string] += 1

    log.debug('dddd', denominator_dict)
    # loop over each unique prefactor
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
    new_numerator_list, new_denominator_list = [], []

    for k, v in numerator_dict.items():
        new_numerator_list.extend([k, ]*v)
    for k, v in denominator_dict.items():
        new_denominator_list.extend([k, ]*v)

    new_numerator_list.sort()
    new_denominator_list.sort()

    if len(numerator_list) > 2 or len(denominator_list) > 2:
        log.debug('xxxx', new_numerator_list)
        log.debug('yyyy', new_denominator_list)

    return new_numerator_list, new_denominator_list


def _build_eT_zhz_python_prefactor(t_list, h, z_right, simplify_flag=True):
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

    # temporary, we need to redo this whole function anyways
    if t_list == []:  # pragma: no cover, never happens?
        return ''

    # special case, single h
    if len(t_list) == 1 and t_list[0] == disconnected_namedtuple(0, 0, 0, 0):  # pragma: no cover, never happens?
        return ''

    # initialize
    numerator_value, denominator_value = 1, 1
    numerator_list, denominator_list = [], []

    # account for prefactor from Hamiltonian term `h`
    # connections = np.count_nonzero([h.m_t[i]+h.n_t[i] for i in range(len(h.m_t))])
    # connected_ts = [t for t in t_list if t.m_h > 0 or t.n_h > 0]

    extra_flag = True

    if h.m > 1:
        denominator_value *= math.factorial(h.m)
        denominator_list.append(f'factorial({h.m})')

        if extra_flag:
            # to account for the permutations of eT-H internal labels around the external labels
            external_perms = math.comb(h.m, h.m_lhs)
            if external_perms > 1:
                numerator_value *= external_perms
                numerator_list.append(f'({external_perms})')

            # like drawing cards from a deck we remove the permutations of the Proj
            new_max = h.m - h.m_lhs

            # to account for the permutations of H-Z internal labels around the external labels
            internal_perms = math.comb(new_max, h.m_r)
            if internal_perms > 1:  # pragma: no cover, probably bc only ground state
                numerator_value *= internal_perms
                numerator_list.append(f'({internal_perms})')

            # to account for the permutations of eT-Z internal labels with themselves
            # as we should have accounted for all permutations by
            # moving H-Z and external labels prior
            # count_t = sum(h.m_t)
            # if count_t > 1:
            #     # account for permutations among the internal labels
            #     numerator_value *= math.factorial(count_t)
            #     numerator_list.append(f'{count_t}!')

    if h.n > 1:
        denominator_value *= math.factorial(h.n)
        denominator_list.append(f'factorial({h.n})')

        if extra_flag:
            # to account for the permutations of eT-H internal labels around
            external_perms = math.comb(h.n, h.n_lhs)
            if external_perms > 1:  # pragma: hot_bands_or_thermal
                numerator_value *= external_perms
                numerator_list.append(f'({external_perms})')

            # like drawing cards from a deck we remove the permutations of the Proj
            new_max = h.n - h.n_lhs

            # to account for the permutations of H-Z internal labels around the external labels
            internal_perms = math.comb(new_max, h.n_r)
            if internal_perms > 1:  # pragma: hot_bands_or_thermal
                numerator_value *= internal_perms
                numerator_list.append(f'({internal_perms})')

            # to account for the permutations of eT-Z internal labels with themselves
            # as we should have accounted for all permutations by
            # moving H-Z and external labels prior
            # count_t = sum(h.n_t)
            # if count_t > 1:
            #     # account for permutations among the internal labels
            #     numerator_value *= math.factorial(count_t)
            #     numerator_list.append(f'{count_t}!')

    if z_right.m > 1:
        denominator_value *= math.factorial(z_right.m)
        denominator_list.append(f'factorial({z_right.m})')

        if extra_flag:

            # to account for the permutations of external labels with other labels on z
            # (we don't account for permuting with themselves as we will symmetrize them later)
            external_perms = math.comb(z_right.m, z_right.m_lhs)
            if external_perms > 1:
                numerator_value *= external_perms
                numerator_list.append(f'({external_perms})')

            # like drawing cards from a deck we remove the permutations of the Proj
            new_max = z_right.m - z_right.m_lhs

            # to account for the permutations of H-Z internal labels with other labels on z
            internal_perms = math.comb(new_max, z_right.m_h)
            if internal_perms > 1:
                numerator_value *= internal_perms
                numerator_list.append(f'({internal_perms})')

            # like drawing cards from a deck we remove the permutations of the H
            new_max -= z_right.m_h

            # to account for the permutations of H-Z internal labels with themselves
            if z_right.m_h > 1:
                numerator_value *= math.factorial(z_right.m_h)
                numerator_list.append(f'{z_right.m_h}!')

            # to account for the permutations of eT-Z internal labels with themselves
            # as we should have accounted for all permutations by
            # moving H-Z and external labels prior
            count_t = sum(z_right.m_t)
            # if count_t > 1:
            #     # account for permutations among the internal labels
            #     numerator_value *= math.factorial(count_t)
            #     numerator_list.append(f'{count_t}!')

            # account for permutations with respect to other labels
            # (this is just to prove that we already accounted for these permutations)
            number = math.comb(new_max, count_t)
            assert number == 1, 'you broke something!!!'
            if number > 1:  # pragma: no cover
                numerator_value *= number
                numerator_list.append(f'{number}')

    if z_right.n > 1:  # pragma: hot_bands_or_thermal
        denominator_value *= math.factorial(z_right.n)
        denominator_list.append(f'factorial({z_right.n})')

        if extra_flag:
            # to account for the permutations of external labels
            number = math.comb(z_right.n, z_right.n_lhs)
            if number > 1:
                numerator_value *= number
                numerator_list.append(f'{number}')

            # to account for the permutations of internal labels (with h)
            if z_right.n_h > 1:
                numerator_value *= math.factorial(z_right.n_h)
                numerator_list.append(f'{z_right.n_h}!')

            # to account for the permutations of internal labels (with t terms)
            count_t = sum(z_right.n_t)
            # if count_t > 1:
            #     # account for permutations among the internal labels
            #     numerator_value *= math.factorial(count_t)
            #     numerator_list.append(f'{count_t}!')

            #     # account for permutations with respect to other labels
            #     number = math.comb(z_right.n, count_t)
            #     numerator_value *= number
            #     numerator_list.append(f'{number}')

    # # account for the number of permutations of all t-amplitudes
    # if len(t_list) > 1:
    #     denominator_value *= math.factorial(len(t_list))
    #     denominator_list.append(f'factorial({len(t_list)})')

    for t in t_list:
        if t.m > 1:  # pragma: t1_truncation
            # by definition
            denominator_value *= math.factorial(t.m)
            denominator_list.append(f'{t.m}!')

        if t.n > 1:  # pragma: t1_truncation
            # by definition
            denominator_value *= math.factorial(t.n)
            denominator_list.append(f'{t.n}!')

    # simplify
    if simplify_flag:
        numerator_list, denominator_list = _simplify_eT_zhz_python_prefactor(numerator_list, denominator_list)

    # glue the numerator and denominator together
    numerator_string = '1' if (numerator_list == []) else f"({' * '.join(numerator_list)})"
    denominator_string = '1' if (denominator_list == []) else f"({' * '.join(denominator_list)})"

    prefactor_string = f"{numerator_string}/{denominator_string} * ".replace('factorial(2)', '2')

    if numerator_string == '1' and denominator_string == '1':
        prefactor_string = ''

    return prefactor_string, numerator_value, denominator_value


def _multiple_perms_logic(term, print_indist_perms: bool = False):
    """ probably need to re-check the multiple t term permutations?

    The flag `print_indist_perms` is meant to indicate if we want to print out all possible permutations;
    even permutations of indistinguishable terms.
    Normally we don't want to do this for computational efficiency reasons.
    """
    omega, t_list, h, z_pair, not_sure_what_this_one_is_for = term

    # first create a dictionary counting the number of distinguishable t terms
    # keys are the terms and the value is the count
    unique_dict = {}
    for t in t_list:
        unique_dict[t] = 1 + unique_dict.get(t, 0)

    # 1 unique t term found
    if len(unique_dict.keys()) == 1:

        # how many copies of that t term are present
        count = list(unique_dict.values())[0]

        # this produces a single permutation
        if not print_indist_perms:
            # if there is a single t term (`count` = 1) then the permutation is a tuple: (0, )
            permutations = [range(count), ]

        # this can produce 1 or more permutations
        else:
            # here permutations is simply all re-orderings of identical t terms
            permutations = unique_permutations(range(count))

        return permutations, unique_dict

    # if multiple distinguishable t terms are present
    if len(unique_dict.keys()) > 1:

        # old_print_wrapper(f"{unique_dict = }")
        # import pdb; pdb.set_trace()

        # how many copies of that t term are present
        count = len(t_list)

        # this produces a single permutation
        if not print_indist_perms:
            # if there is a single t term (`count` = 1) then the permutation is a tuple: (0, )
            permutations = [range(count), ]

        # this can produce 1 or more permutations
        else:
            # here permutations is simply all re-orderings of identical t terms
            permutations = unique_permutations(range(count))

        return permutations, unique_dict

    #     # lst = []
    #     # for v in unique_dict.values():
    #     #     lst.append(*unique_permutations(range(v)))
    #     old_print_wrapper([unique_permutations(range(v)) for k, v in unique_dict.items()])
    #     return dict([(k, list(*unique_permutations(range(v)))) for k, v in unique_dict.items()]), unique_dict

    raise Exception("Should not get here")  # pragma: no cover


def compute_prefactor_adjustment(h, z_right):  # pragma: deprecated
    """ x """

    adjustment = 1
    bottom = 1

    if h.m > 1:
        bottom *= math.factorial(h.m)

        # to account for the permutations of eT-H internal labels around
        external_perms = math.comb(h.m, h.m_lhs)
        if external_perms > 1:
            adjustment *= external_perms

        # like drawing cards from a deck we remove the permutations of the Proj
        new_max = h.m - h.m_lhs

        # to account for the permutations of H-Z internal labels around the external labels
        internal_perms = math.comb(new_max, h.m_r)
        if internal_perms > 1:
            adjustment *= internal_perms

        # to account for the permutations of eT-Z internal labels with themselves
        # as we should have accounted for all permutations by
        # moving H-Z and external labels prior
        count_t = sum(h.m_t)
        if count_t > 1:
            # account for permutations among the internal labels
            adjustment *= math.factorial(count_t)

    if h.n > 1:
        bottom *= math.factorial(h.n)

        # to account for the permutations of eT-H internal labels around
        external_perms = math.comb(h.n, h.n_lhs)
        if external_perms > 1:
            adjustment *= external_perms

        # like drawing cards from a deck we remove the permutations of the Proj
        new_max = h.n - h.n_lhs

        # to account for the permutations of H-Z internal labels around the external labels
        internal_perms = math.comb(new_max, h.n_r)
        if internal_perms > 1:
            adjustment *= internal_perms

        # to account for the permutations of eT-Z internal labels with themselves
        # as we should have accounted for all permutations by
        # moving H-Z and external labels prior
        count_t = sum(h.n_t)
        if count_t > 1:
            # account for permutations among the internal labels
            adjustment *= math.factorial(count_t)

    # ---------------------------------------------------------------------------------------------------------

    if z_right.m > 1:
        bottom *= math.factorial(z_right.m)

        # to account for the permutations of external labels with other labels on z
        # (we don't account for permuting with themselves as we will symmetrize them later)
        external_perms = math.comb(z_right.m, z_right.m_lhs)
        if external_perms > 1:
            adjustment *= external_perms

        # like drawing cards from a deck we remove the permutations of the Proj
        new_max = z_right.m - z_right.m_lhs

        # to account for the permutations of H-Z internal labels with other labels on z
        internal_perms = math.comb(new_max, z_right.m_h)
        if internal_perms > 1:
            adjustment *= internal_perms

        # like drawing cards from a deck we remove the permutations of the H
        new_max -= z_right.m_h

        # to account for the permutations of H-Z internal labels with themselves
        if z_right.m_h > 1:
            adjustment *= math.factorial(z_right.m_h)

        # to account for the permutations of eT-Z internal labels with themselves
        # as we should have accounted for all permutations by
        # moving H-Z and external labels prior
        count_t = sum(z_right.m_t)
        if count_t > 1:
            # account for permutations among the internal labels
            adjustment *= math.factorial(count_t)

        # account for permutations with respect to other labels
        # (this is just to prove that we already accounted for these permutations)
        number = math.comb(new_max, count_t)
        assert number == 1, 'you broke something!!!'
        if number > 1:  # pragma: no cover
            adjustment *= number

    if z_right.n > 1:
        bottom *= math.factorial(z_right.n)

        # to account for the permutations of external labels
        number = math.comb(z_right.n, z_right.n_lhs)
        if number > 1:
            adjustment *= number

        # to account for the permutations of internal labels (with h)
        if z_right.n_h > 1:
            adjustment *= math.factorial(z_right.n_h)

        # to account for the permutations of internal labels (with t terms)
        count_t = sum(z_right.n_t)
        if count_t > 1:
            # account for permutations among the internal labels
            adjustment *= math.factorial(count_t)

            # account for permutations with respect to other labels
            number = math.comb(z_right.n, count_t)
            adjustment *= number

    print(f"{adjustment = }")
    print(f"{bottom = }")
    print(f"{ adjustment / bottom = }")

    return adjustment / bottom


# ----------------------------------------------------------------------------------------------- #
# these handle collating multiple-lines

def _handle_multiline_same_prefactor(
    output_list, prefactor, string_list, lhs_rhs,
    nof_tabs=0
):
    """ Specific formatting of multiple terms that share the same prefactor

    When we have multiple einsum terms which share the same prefactor
    we collate them in a specific  manner like so:
    R += prefactor * (
        <einsum equation> +
        <einsum equation> +
        ...
        <einsum equation> +
        <einsum equation>
    )

    This function simply glues them together in that fashion.
    """

    tabber = tab*nof_tabs

    if len(string_list) > 1:

        # header
        if lhs_rhs == 'RHS':
            output_list.append(f"{tabber}R += {prefactor}(")
        elif lhs_rhs == 'LHS':
            output_list.append(f"{tabber}Z += {prefactor}(")

        # add each line
        for string in string_list:
            output_list.append(f"{tabber}{tab}{string} +")

        # remove the last plus symbol
        output_list[-1] = output_list[-1][:-2]

        # footer
        output_list.append(f"{tabber})")

    else:
        # print(f"c {string_list = }")
        if lhs_rhs == 'RHS':
            output_list.append(f"{tabber}R += {prefactor}{string_list[0]}")
        elif lhs_rhs == 'LHS':
            output_list.append(f"{tabber}Z += {prefactor}{string_list[0]}")

    return output_list


def _collect_z_contributions(
    h_dict, return_array, lhs_rhs, trunc_obj_name,
    suppress_empty_if_checks=True, nof_tabs=0,
):
    """ This is a gatekeeper for all einsum equations that are printed
    everything gets parsed through this function
    every single line that gets printed with `einsum` in it comes from the `return_array`
    which this function appends to
    """

    # h^0_0 with zero order Taylor series contributions
    for z_order, z_dict in h_dict.items():

        # exception case for z_order == 0 which has no if statement
        tab_adjust = nof_tabs if z_order == 0 else nof_tabs + 1

        temp_z_list = []  # if not empty

        for prefactor, string_list in z_dict.items():
            if string_list == []:  # skip if empty
                continue

            _handle_multiline_same_prefactor(
                temp_z_list, prefactor, string_list, lhs_rhs,
                nof_tabs=tab_adjust
            )

        # processing
        if suppress_empty_if_checks and (temp_z_list == []):
            continue

        if z_order == 0:
            return_array.extend(temp_z_list)

        else:
            tabstr = tab * nof_tabs
            z_header_if_string = (
                f"{tabstr}if {trunc_obj_name}.z_at_least_{hamiltonian_order_tag[z_order]}:"
            )

            if temp_z_list == []:  # pragma: no cover
                return_array.append(z_header_if_string)
                return_array.append(f"{tabstr}{tab}pass")
            else:
                return_array.append(z_header_if_string)
                return_array.extend(temp_z_list)


# ----------------------------------------------------------------------------------------------- #
# big boy function that does most of the work


def _write_third_eTz_einsum_python(rank, operators, t_term_list, lhs_rhs, trunc_obj_name='truncation', b_loop_flag=False, suppress_empty_if_checks=True):
    """ Still being written

    the flag `suppress_empty_if_checks` is to toggle the "suppression" of code such as
    ```
        if truncation.singles:
            pass
    ```
    which doesn't actually do anything.
    """

    master_omega, H, Z, eT_taylor_expansion = operators
    log.info("Starting this function")

    if t_term_list == []:
        return ["pass  # no valid terms here", ]

    return_list = []

    # hamiltonian_rank_list is as:
    # list
    #   of dictionaries
    #       of dictionaries
    #           of lists
    hamiltonian_rank_list = []
    for i in range(H.maximum_rank+1):
        hamiltonian_rank_list.append(dict([(i, {}) for i in range(master_omega.maximum_rank+1)]))

    # -------------------------------------------------------------- #
    # the big loop!
    # -------------------------------------------------------------- #
    for term in t_term_list:

        omega, t_list, h, z_pair, not_sure_what_this_one_is_for = term

        # define the indexing of the `h_args` dictionary
        if lhs_rhs == 'RHS':
            h_operand = f"h_args[({h.m}, {h.n})]"

        elif lhs_rhs == 'LHS':
            if h.m == 0 and h.n == 0:
                h_operand = None
                dT = False
            else:
                h_operand = f"dT[({h.m}, {h.n})]"
                dT = True

        # define the indexing of the `z_args` dictionary
        if lhs_rhs == 'RHS':
            z_left, z_right = z_pair
            z_operand = f"z_args[({z_right.m}, {z_right.n})]"

        elif lhs_rhs == 'LHS':
            # TODO
            # dz_args is a list of dz_# where if Z is max order 3 then the length is 3 - the value of the current function call.
            # So compute_z_0_residual(Z, t_conj, dT, dz_args) that dz_args would be length 3 (3-0) and include dz_3, dz_2, dz_1

            # z_operand = f"dz_args[({dz3}, {dz2}, {dz1})]"
            if dT is False:
                z_left, z_right = z_pair
                z_operand = f"dz_args[({z_right.m}, {z_right.n})]"
            else:
                z_left, z_right = z_pair
                z_operand = f"z_args[({z_right.m}, {z_right.n})]"

        # the t counts as identity
        if t_list == []:  # pragma: deprecated
            permutations = None
            max_t_rank = 0
            prefactor = ''

        # if we have one or more t terms we need to figure out the permutations
        # this part is quite tricky
        else:

            # note that to correctly do the full prefactors for all terms you need both
            # `print_indist_perms = True` AND you need to have `simplify_flag = True`

            print_indist_perms = False

            # the easy case where we just print EVERYTHING
            if print_indist_perms is True:  # pragma: no cover 'we want this functionality for debugging and investigation'

                # logic about multiple permutations, generate lists of unique t terms
                permutations, unique_dict = _multiple_perms_logic(term, print_indist_perms)

                prefactor, *_ = _build_eT_zhz_python_prefactor(t_list, h, z_right, simplify_flag=False)

                max_t_rank = max(t.rank for t in t_list)

                old_print_wrapper(omega, h, t_list, permutations, sep='\n')

            # the hard case where we have to fiddle with prefactors and permutations etc
            elif print_indist_perms is False:

                # first we need to calculate what the prefactors would be IF we counted all permutations
                total_perms, total_unique_dict = _multiple_perms_logic(term, print_indist_perms=True)

                # then we need to calculate the prefactors NOT counting indistinguishable permutations
                permutations, unique_dict = _multiple_perms_logic(term, print_indist_perms=False)

                # adjust the prefactor by the number of t permutations we didn't print because they were indistinguishable
                prefactor_adjustment = len(total_perms) // len(permutations)

                # we also do a prefactor adjustment based on the number of terms h and z share because we only want to permute them once
                # even though `_build_eT_zhz_python_prefactor` counts them 'twice' in a sense
                assert (h.m_r == z_right.n_h) and (h.n_r == z_right.m_h), f'f{h.m_r = }    {h.n_r}\n{z_right.n_h = }    {z_right.m_h}\n'
                # prefactor_adjustment2 = compute_prefactor_adjustment(h, z_right)
                # prefactor_adjustment2 = max(h.m_r + h.n_r, 1)
                prefactor_adjustment = 1
                prefactor_adjustment2 = 1

                prefactor, n, d = _build_eT_zhz_python_prefactor(t_list, h, z_right, simplify_flag=True)

                adjusted_prefactor = Fraction(n*prefactor_adjustment*prefactor_adjustment2, d)
                n, d = adjusted_prefactor.as_integer_ratio()

                old_print_wrapper(f"\n{len(total_perms) = } \n{len(permutations) = } \n{prefactor_adjustment = } \n{prefactor_adjustment2 = } \n{adjusted_prefactor = } \n{prefactor = }\n{n = }\n{d = }")

                # adjust the prefactor if necessary
                if n != d:
                    prefactor = f'({n} / {d}) * '

                # force the prefactor to be 1 (which we represent with an empty string)
                # this is to take care of the case where `n == d` because of adjustment but
                # the current value of the string `prefactor` is NOT an empty string
                else:
                    prefactor = ''

                # import pdb; pdb.set_trace()

                max_t_rank = max(t.rank for t in t_list)

                old_print_wrapper(omega, h, t_list, permutations, sep='\n')

        # we still need to account for output/omega permutations

        # -----------------------------------------------------------------------------------------
        # build with permutations
        hamiltonian_rank_list[max(h.m, h.n)].setdefault(
            max_t_rank, {}
        ).setdefault(
            z_right.rank, {}
        ).setdefault(
            prefactor, []
        )

        # old_print_wrapper(f"{t_list = }")
        if lhs_rhs == 'RHS':
            e_a = _eT_zhz_einsum_electronic_components(t_list, z_right, b_loop_flag)
            v_a, remaining_indices = _eT_zhz_einsum_vibrational_components(t_list, h, z_right, b_loop_flag)

        elif lhs_rhs == 'LHS':
            e_a = _eT_zhz_einsum_electronic_components_lhs(t_list, h, z_right, b_loop_flag)
            v_a, remaining_indices = _eT_zhz_einsum_vibrational_components_lhs(t_list, h, z_right, b_loop_flag)

        # the output einsum argument is different for RHS versus LHS
        if lhs_rhs == 'RHS':
            e_char = 'a'
        elif lhs_rhs == 'LHS':
            e_char = 'c'

        print(len(e_a), len(v_a))
        # if there is only a single distinguishable t term
        # eg: t1 * t1 * t1 ---> 3 indistinguishable t terms
        # as opposed to t1 * t2 being two distinguishable t terms
        if len(unique_dict.keys()) == 1:

            # extract the single key
            disconnected_t_term = next(iter(unique_dict.keys()))

            # if this is an instance of t^0_0
            if disconnected_t_term.rank == 0:

                # no remaining (external) indices/labels means we simply proceed as normal and glue everything together
                if remaining_indices == '':

                    # create the initial indices
                    combined_electronic_vibrational = [f"{e_a[i]}{v_a[i]}" for i in range(len(e_a))]

                    # prune all empty contributions
                    combined_electronic_vibrational = [s for s in combined_electronic_vibrational if s != '']

                    # glue them together
                    string = ", ".join(combined_electronic_vibrational)

                    # stick the indices into the full einsum function call
                    if h_operand is None:  # pragma: hot_bands_or_thermal
                        string = f"np.einsum('{string} -> {e_char}{remaining_indices}', {z_operand})"
                    else:
                        string = f"np.einsum('{string} -> {e_char}{remaining_indices}', {h_operand}, {z_operand})"

                    # append that string to the current list
                    hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][z_right.rank][prefactor].append(string)

                # this means we need to permute over the remaining (external) indices/labels
                elif len(remaining_indices) >= 1:

                    for perm in unique_permutations(remaining_indices):

                        # create the initial indices
                        combined_electronic_vibrational = [f"{e_a[i]}{v_a[i]}" for i in range(len(e_a))]

                        # prune all empty contributions
                        combined_electronic_vibrational = [s for s in combined_electronic_vibrational if s != '']

                        # glue them together
                        string = ", ".join(combined_electronic_vibrational)

                        # stick the indices into the full einsum function call
                        if h_operand is None:  # pragma: hot_bands_or_thermal
                            string = f"np.einsum('{string} -> {e_char}{remaining_indices}', {z_operand})"
                        else:
                            string = f"np.einsum('{string} -> {e_char}{remaining_indices}', {h_operand}, {z_operand})"

                        # append that string to the current list
                        hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][z_right.rank][prefactor].append(string)

                    # wait why are we passing instead of exiting?
                    if len(remaining_indices) >= 2:
                        # sys.exit(0)
                        pass

            # if this is a t term with at least 1 creation or annihilation operator
            elif disconnected_t_term.rank > 0:

                # create the string of (t terms/t_operands) that we are tracing over
                # since they are all identical we don't care about ordering
                if lhs_rhs == 'RHS':
                    arg_string_name = 't_args'
                elif lhs_rhs == 'LHS':
                    arg_string_name = 't_conj'
                t_operands = ', '.join([
                    f"{arg_string_name}[({t.m_lhs + t.m_h + t.m_r}, {t.n_lhs + t.n_h + t.n_r})]"
                    for t in t_list
                ])
                for perm in permutations:
                    old_print_wrapper(f"{permutations = } {perm = }")
                    old_print_wrapper(t_list, unique_dict)

                    # create the t term indices
                    combined_electronic_vibrational = [f"{e_a[i]}{v_a[p]}" for i, p in enumerate(perm)]

                    # add the h term indices
                    combined_electronic_vibrational.append(f"{e_a[-2]}{v_a[-2]}")

                    # add the Z term indices
                    combined_electronic_vibrational.append(f"{e_a[-1]}{v_a[-1]}")

                    # prune all empty contributions
                    combined_electronic_vibrational = [s for s in combined_electronic_vibrational if s != '']

                    # glue them together
                    string = ", ".join(combined_electronic_vibrational)

                    # stick the indices into the full einsum function call
                    if h_operand is None:
                        string = f"np.einsum('{string} -> {e_char}{remaining_indices}', {t_operands}, {z_operand})"
                    else:
                        string = f"np.einsum('{string} -> {e_char}{remaining_indices}', {t_operands}, {h_operand}, {z_operand})"

                    # append that string to the current list
                    hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][z_right.rank][prefactor].append(string)

        # if there is multiple distinguishable t terms
        # eg: t1 * t2 * t1 ---> 2 distinguishable t terms (t1, t2)
        elif len(unique_dict.keys()) > 1:

            old_print_wrapper(f"{unique_dict = }")
            old_print_wrapper(f"{len(unique_dict) = } is > 1 ?{len(unique_dict) > 1}")

            # import pdb; pdb.set_trace()

            for perm in permutations:

                # create the string of (t terms/t_operands) that we are tracing over
                # since they are NOT identical, we do have to pay attention to the ordering

                # for simplicity we create a new re-ordered list
                re_ordered_t_list = [t_list[i] for i in perm]

                # now we can just easily iterate over the re-ordered list
                if lhs_rhs == 'RHS':
                    arg_string_name = 't_args'
                elif lhs_rhs == 'LHS':
                    arg_string_name = 't_conj'
                t_operands = ', '.join([
                    f"{arg_string_name}[({t.m_lhs + t.m_h + t.m_r}, {t.n_lhs + t.n_h + t.n_r})]"
                    for t in re_ordered_t_list
                ])

                # create the t term indices
                combined_electronic_vibrational = [f"{e_a[i]}{v_a[p]}" for i, p in enumerate(perm)]

                # add the h term indices
                combined_electronic_vibrational.append(f"{e_a[-2]}{v_a[-2]}")

                # add the Z term indices
                combined_electronic_vibrational.append(f"{e_a[-1]}{v_a[-1]}")

                # glue them together
                string = ", ".join(combined_electronic_vibrational)

                # stick the indices into the full einsum function call
                if h_operand is None:  # pragma: hot_bands_or_thermal
                    string = f"np.einsum('{string} -> {e_char}{remaining_indices}', {t_operands}, {z_operand})"
                else:
                    string = f"np.einsum('{string} -> {e_char}{remaining_indices}', {t_operands}, {h_operand}, {z_operand})"

                # append that string to the current list
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][z_right.rank][prefactor].append(string)

            old_print_wrapper(f"{string = }")

            # import pdb; pdb.set_trace()

        else:  # pragma: no cover
            raise Exception("Should not get here")

    """ lazy hack because we I don't trust myself to properly modify `latex_eT_zhz.py` at the moment

    the point is that we expect to have duplicate strings in our hamiltonian_rank_list and we want to remove them
    so we just loop over all elements/sub elements of the list and apply list(set(x)) on that respective element
    """
    investigate_flag = False
    for i, e in enumerate(hamiltonian_rank_list):
        old_print_wrapper(f"{type(e) = }")
        for key_t, t_dict in hamiltonian_rank_list[i].items():
            old_print_wrapper(f"{key_t = } {t_dict = }")
            for key_z, z_dict in t_dict.items():
                for pref, p_list in z_dict.items():

                    unique_list = sorted(list(set(p_list)))

                    # if there are no duplicates then simply move on
                    if len(p_list) == len(unique_list):
                        pass

                    else:
                        # print("Found duplicates")
                        # import pdb; pdb.set_trace()
                        hamiltonian_rank_list[i][key_t][key_z][pref] = unique_list

                        # temp_dict = {}
                        # for s in e_list:
                        #     temp_dict[s] = 1 + temp_dict.get(s, 0)
                        # no_dupe_list = [s for s, count in temp_dict.items() if count == 1]
                        # dupe_dict = dict([(s, count) for s, count in temp_dict.items() if s not in no_dupe_list])
                        # old_print_wrapper('')
                        # old_prefactor = pref
                        # for s, count in dupe_dict.items():
                        #     old_print_wrapper(f"{old_prefactor = } {count = } \n{s = }\n")

                        # # import pdb; pdb.set_trace()

                        # hamiltonian_rank_list[i][key][pref] = no_dupe_list

                    # # otherwise we remove the duplicates and re-assign them to different prefactors
                    # else:
                    #     old_print_wrapper(f"\n{pref = }       {len(hamiltonian_rank_list[i][key][pref])}")
                    #     old_print_wrapper(f"{e_list = } \n{hamiltonian_rank_list[i][key][pref] = }")

                    #     # we first collect all the items into a dictionary
                    #     temp_dict = {}
                    #     for s in e_list:
                    #         temp_dict[s] = 1 + temp_dict.get(s, 0)

                    #     # create a list of the non duplicated summations
                    #     no_dupe_list = [s for s, count in temp_dict.items() if count == 1]

                    #     # put the non dupes back with their appropriate prefactor
                    #     hamiltonian_rank_list[i][key][pref] = no_dupe_list.copy()

                    #     # remove the non duplicate items
                    #     dupe_dict = dict([(s, count) for s, count in temp_dict.items() if s not in no_dupe_list])
                    #     for s in no_dupe_list:
                    #         del temp_dict[s]

                    #     # now on a case-by-case basis determine what the new prefactors should be
                    #     for s, count in dupe_dict.items():
                    #         # this is how much we need to change the prefactor by
                    #         adjustment_factor = count

                    #         n, rest = pref[1:].split('/')
                    #         d = rest.split(')')[0]
                    #         old_print_wrapper(f"{n = } {d = }")

                    #         old_print_wrapper(f"{adjustment_factor = } {pref = } {s = }")
                    #         import pdb; pdb.set_trace()

                    #     hamiltonian_rank_list[i][key][pref] = list(set(e_list))

                    #     old_print_wrapper(f"\n{pref = }       {len(hamiltonian_rank_list[i][key][pref])}")
                    #     old_print_wrapper(f"{e_list = } \n{hamiltonian_rank_list[i][key][pref] = }\n")

                # print(f"before removal of empty z terms\n{hamiltonian_rank_list[i][key_t][pref]}")
                # remove all empty prefactors
                hamiltonian_rank_list[i][key_t][key_z] = dict([
                    (pref, p_list)
                    for pref, p_list in sorted(hamiltonian_rank_list[i][key_t][key_z].items())
                    if p_list != []
                ])
                # print(f"after removal of empty z terms\n{hamiltonian_rank_list[i][key_t][pref]}")
                if investigate_flag: import pdb; pdb.set_trace();

            # print(f"before removal of empty prefactors\n{hamiltonian_rank_list[i][key_t]}")
            # remove all empty prefactors
            hamiltonian_rank_list[i][key_t] = dict([
                (key_z, z_dict)
                for key_z, z_dict in sorted(hamiltonian_rank_list[i][key_t].items())
                if z_dict != {}
            ])
            # print(f"after removal of empty prefactors\n{hamiltonian_rank_list[i][key_t]}")
            if investigate_flag: import pdb; pdb.set_trace();

    def compact_display_of_hamiltonian_rank_list(hamiltonian_rank_list):
        """ for debugging purposes """

        # clear the console
        old_print_wrapper('\n'*15)
        old_print_wrapper("Compact display of `hamiltonian_rank_list`:")

        for i, h_rank_dict in enumerate(hamiltonian_rank_list):
            old_print_wrapper('')

            if h_rank_dict == {}:
                old_print_wrapper(f"{i = :<3d} {h_rank_dict = }")
            else:
                old_print_wrapper(f"{i = :<3d} {len(h_rank_dict) = }")

            for key, t_rank_dict in h_rank_dict.items():

                if t_rank_dict == {}:
                    old_print_wrapper(f"{tab}{key = :<3} {t_rank_dict = }")
                else:
                    old_print_wrapper(f"{tab}{key = :<3} {len(t_rank_dict) = }")

                for key2, z_rank_dict in t_rank_dict.items():

                    if z_rank_dict == {}:
                        old_print_wrapper(f"{tab}{key2 = :<3} {z_rank_dict = }")
                    else:
                        old_print_wrapper(f"{tab}{key2 = :<3} {len(z_rank_dict) = }")

                    for key3, prefactor_list in z_rank_dict.items():

                        if prefactor_list == []:
                            old_print_wrapper(f"{tab}{tab}{key3 = :<20s} {tab}{prefactor_list = }")
                        else:
                            old_print_wrapper(f"{tab}{tab}{key3 = :<20s} {tab}{len(prefactor_list) = }")

    compact_display_of_hamiltonian_rank_list(hamiltonian_rank_list)

    """
    somewhere between here and the end of the function we are 'losing' the terms that we want to print???
    """

    # import pdb; pdb.set_trace()

    # -----------------------------------------------------------------------------------------
    # remove any duplicates
    # for h_rank_list in hamiltonian_rank_list:
    #     for t_rank_list in h_rank_list:
    #         for prefactor_list in t_rank_list:

    # -----------------------------------------------------------------------------------------

    h_contribution_list = []

    _collect_z_contributions(
        hamiltonian_rank_list[0][0], h_contribution_list,
        lhs_rhs, trunc_obj_name,
        suppress_empty_if_checks=suppress_empty_if_checks,
        nof_tabs=0,
    )

    # loop over first order (and higher) Taylor series contributions
    for j in range(1, master_omega.maximum_rank+1):
        if hamiltonian_rank_list[0][j] is {}: continue;  # skip if empty

        temp_list = []  # if not empty

        # h^0_0 with first order (and higher) Taylor series contributions
        _collect_z_contributions(
            hamiltonian_rank_list[0][j], temp_list,
            lhs_rhs, trunc_obj_name,
            suppress_empty_if_checks=suppress_empty_if_checks,
            nof_tabs=1,
        )

        # processing
        if temp_list == [] and suppress_empty_if_checks:
            continue
        t_header_if_string = f"if {trunc_obj_name}.t_{taylor_series_order_tag[j]}:"
        if temp_list == []:
            h_contribution_list.append(t_header_if_string)
            h_contribution_list.append(f"{tab}pass")
        else:
            h_contribution_list.append(t_header_if_string)
            h_contribution_list.extend(temp_list)

    # loop over first order (and higher) h^i_j terms
    for i in range(1, H.maximum_rank+1):
        if hamiltonian_rank_list[i] is {}: continue;  # skip if empty

        temp_list = []  # if not empty

        # h^i_j with zero order Taylor series contributions
        _collect_z_contributions(
            hamiltonian_rank_list[i][0], temp_list,
            lhs_rhs, trunc_obj_name,
            suppress_empty_if_checks=suppress_empty_if_checks,
            nof_tabs=1,
        )

        # for prefactor in hamiltonian_rank_list[i][0].keys():
        #     if hamiltonian_rank_list[i][0][prefactor] is {}: continue;  # skip if empty

        #     for z_order, string_list in hamiltonian_rank_list[i][0][prefactor].items():
        #         _handle_multiline_same_prefactor(temp_list, prefactor, string_list, nof_tabs=1)

        # processing
        h_header_if_string = f"if {trunc_obj_name}.h_at_least_{hamiltonian_order_tag[i]}:"
        h_contribution_list.append("")  # spacer
        h_contribution_list.append(h_header_if_string)

        if temp_list != []:
            h_contribution_list.extend(temp_list)

        # loop over first order (and higher) Taylor series contributions
        for j in range(1, master_omega.maximum_rank+1):
            if hamiltonian_rank_list[i][j] is {}: continue;  # skip if empty

            temp_list = []  # if not empty

            # h^i_j with first order (and higher) Taylor series contributions
            _collect_z_contributions(
                hamiltonian_rank_list[i][j], temp_list,
                lhs_rhs, trunc_obj_name,
                suppress_empty_if_checks=suppress_empty_if_checks,
                nof_tabs=2,
            )

            # processing
            if temp_list == [] and suppress_empty_if_checks:
                continue

            # else
            t_header_if_string = f"{tab}if {trunc_obj_name}.t_{taylor_series_order_tag[j]}:"

            if temp_list == []:
                h_contribution_list.append(t_header_if_string)
                h_contribution_list.append(f"{tab}{tab}pass")
            else:
                h_contribution_list.append(t_header_if_string)
                h_contribution_list.extend(temp_list)

            if not suppress_empty_if_checks:
                h_contribution_list.append("")

        # j loop
        if not suppress_empty_if_checks:
            h_contribution_list.append("")

        # if the header didn't actually have any contributions underneath it then simply remove it
        if h_contribution_list[-1] == h_header_if_string:
            del h_contribution_list[-1]

    # add empty string -> creates empty line for better formatting
    # i loop

    # old_print_wrapper(f"\n{h_contribution_list = }")
    # compact_display_of_hamiltonian_rank_list(hamiltonian_rank_list)
    # import pdb; pdb.set_trace()

    if h_contribution_list == []:
        return_list.append("pass")
    else:
        return_list.extend(h_contribution_list)

    """ in case we have any double-empty-lines remove them
    note: should find a better way to solve this in the future
    this removes any empty strings '' when they follow each other
        ['a', '', ''] -> ['a', '']
    or
        ['a', '', 'b', '', '', 'c'] -> ['a', '', 'b', '', 'c']
    """
    no_duplicate_list = []
    for i in range(len(return_list)-1):
        # print(i)
        # print(return_list[i])
        if (return_list[i+1] == '') and (return_list[i] == ''):
            continue
        else:
            no_duplicate_list.append(return_list[i])

    # append the last element from the list
    no_duplicate_list.append(return_list[-1])

    # the list should never begin with a newline
    if no_duplicate_list[0] == '':
        no_duplicate_list.pop(0)

    # print(f"{no_duplicate_list = }")
    # import pdb; pdb.set_trace()

    return no_duplicate_list
# ----------------------------------------------------------------------------------------------- #


def remove_all_excited_state_t_terms(eT_taylor_expansion):
    """ this is a really sloppy method so it should be improved in the future """
    i_list, j_list = [], []

    # loop over the outer list
    for i, e1 in enumerate(eT_taylor_expansion):

        # if this element is not a list
        if not isinstance(e1, list):
            # and the element has annihilation operators
            if e1.m > 0:  # pragma: hot_bands_or_thermal
                # record this index as one that we have to remove
                i_list.append(i)

            continue

        # loop over the inner list
        for j, e2 in enumerate(e1):

            # if this element is not a list
            if not isinstance(e2, list):
                # and the element has annihilation operators
                if e2.m > 0:  # pragma: hot_bands_or_thermal
                    # record this index as one that we have to remove
                    j_list.append(1)
                continue

            # simply counting how many elements in the inner-most list
            # have annihilation operators
            # (note that we have AT MOST lists nested twice, therefore)
            # (all elements in `e2` are guaranteed to be namedtuples and NOT lists)
            k_list = [1 for k, e3 in enumerate(e2) if e3.m > 0]

    # here is the "buggy/bad" part
    # what i want to do is delete specific elements
    # but if I delete 1-by-1 then other elements index changes
    # so we cheat for now and just try and delete a # of elements and
    # hope they're all at the front of the list
    # obviously this needs to be changed

            # delete # elements
            del e2[:sum(k_list)]

        # delete # elements
        del e1[:sum(j_list)]

    # delete # elements
    del eT_taylor_expansion[:sum(i_list)]

    # filter out all the empty lists
    filtered_eT_taylor_expansion = [x for x in eT_taylor_expansion if x != []]

    # return
    return filtered_eT_taylor_expansion
# ----------------------------------------------------------------------------------------------- #
# this function definitely needs a rework its not sure what its trying to do exactly


def _generate_eT_zhz_einsums(Proj, operators, lhs_rhs, only_ground_state=False, remove_f_terms=False, opt_einsum=False):
    """Return a string containing python code to be placed into a .py file.
    This does all the work of generating the einsums.

    Proj * (t*t*t) * H * Z

    This one basically needs to be like the t term stuff EXCEPT:
        - there is a single z term
        - it is always on the right side
        - always bond to projection operator in opposite dimension (^i _i)
    """
    master_omega, H, Z, eT_taylor_expansion = operators

    valid_zero_list = []   # store all valid Omega * (1)        * h * Z  terms here
    valid_term_list = []   # store all valid Omega * (t*t*...t) * h * Z  terms here

    """ First we want to generate a list of valid terms.
    We start with the list of lists `eT_taylor_expansion` which is processed by `_filter_out_valid_eT_terms`.
    This function identifies valid pairings AND places those pairings in the `valid_term_list`.
    Specifically we replace the `general_operator_namedtuple`s with `connected_namedtuple`s and/or
    `disconnected_namedtuple`s.
    """

    # remove excited state contributions
    if only_ground_state:

        # remove all excited state Z's
        Z.operator_list[:] = [z for z in Z.operator_list if z.n == 0]

        eT_taylor_expansion = remove_all_excited_state_t_terms(eT_taylor_expansion)

        if lhs_rhs == 'LHS':
            # in this case H is actually masquerading as dT/dtau which needs to be treated the same as Z
            H.operator_list[:] = [h for h in H.operator_list if h.n == 0]

        #

    # do the terms without T contributions first
    zero_eT_term = eT_taylor_expansion[0]
    log.debug(zero_eT_term, "-"*100, "\n\n")
    _filter_out_valid_eTz_terms(Proj, zero_eT_term, H, None, Z, valid_zero_list, lhs_rhs)

    """ remove the actual dz/dtau term that we are trying to calculate as in d(z^m_n)/dtau = RHS `- LHS`
    where this whole module is trying to write code to compute the `LHS` part and then we subtract it later

    We need to make sure we DONT subtract the specific 1 * 1 * d(z^m_n)/dtau term

    At this point in the code we will try to remove that term from `valid_zero_list` so that it is NOT printed as a contribution
    also assume that we do not support thermal equations (LHS.m IS ALWAYS ZERO)
    """
    for term in valid_zero_list:

        # if the Hamiltonian / (dt/tau) term is rank 0
        middle_term_rank_zero = bool(term[2].rank == 0)
        z_derivative_is_correct_order = bool(term[3][1].m == term[0].n)

        if middle_term_rank_zero and z_derivative_is_correct_order:
            valid_zero_list.remove(term)
            break  # we should only ever remove a single term

    # import pdb; pdb.set_trace()

    # cheat and remove all t terms
    # for i, _ in enumerate(valid_zero_list):
    #     valid_zero_list[i][1] = []

    # then do the rest
    for eT_series_term in eT_taylor_expansion[1:]:
        log.debug(eT_series_term, "-"*100, "\n\n")

        # generate all valid combinations
        _filter_out_valid_eTz_terms(Proj, eT_series_term, H, None, Z, valid_term_list, 'RHS')

    if False:
        old_print_wrapper('\n\n\n')

        # save in a human readable format to a file
        file_str = '\n]\n'.join([f'\n{tab}'.join(['['] + [str(y) for y in x]) for x in valid_term_list])
        with open('temp.txt', 'w') as fp:
            fp.write(file_str + '\n]')

        # for i, a in enumerate(valid_term_list):
        #     old_print_wrapper(f"{i+1:>4d}", a)

        pdb.set_trace() if inspect.stack()[-1].filename == 'driver.py' else None

    if valid_zero_list == [] and valid_term_list == []:
        return [
            ["pass  # all terms invalid", ],
            ["pass  # all terms invalid", ],
        ]

    for i, _ in enumerate(valid_term_list):
        # old_print_wrapper(i, valid_term_list[i])
        old_print_wrapper(i, valid_term_list[i][1])

    # import pdb; pdb.set_trace()

    # return _prepare_third_eTz_latex(valid_term_list, remove_f_terms=remove_f_terms)

    # alst = [
    #     _write_third_eTz_einsum_python(Proj.rank, operators, valid_zero_list, b_loop_flag=True),
    # ]

    # blst = [
    #     _write_third_eTz_einsum_python(Proj.rank, operators, valid_term_list, b_loop_flag=True),
    # ]

    # old_print_wrapper('\n\na', alst)
    # import pdb; pdb.set_trace()
    # old_print_wrapper('\n\na', blst)
    # import pdb; pdb.set_trace()

    return_list = [
        # all summation terms where e^T = 1
        _write_third_eTz_einsum_python(Proj.rank, operators, valid_zero_list, lhs_rhs, b_loop_flag=True),
        # the rest of the summation terms with e^T != 1
        _write_third_eTz_einsum_python(Proj.rank, operators, valid_term_list, lhs_rhs, b_loop_flag=True),
    ]

    return return_list
# ----------------------------------------------------------------------------------------------- #
# everything below here is pretty standard wrapping and string processing stuff (nothing too fancy)


def _construct_eT_zhz_compute_function(Proj, operators, lhs_rhs, only_ground_state=False, opt_einsum=False):
    """ x """

    return_string = ""  # concatenate all results to this

    # pre-defines
    specifier_string = f"m{Proj.m}_n{Proj.n}"
    four_tabbed_newline = "\n" + tab*4
    five_tabbed_newline = "\n" + tab*5

    # generate ground state einsums
    ground_state_only_einsums = _generate_eT_zhz_einsums(Proj, operators, lhs_rhs, only_ground_state=True, opt_einsum=opt_einsum)

    # generate ground + excited state einsums
    if not only_ground_state:  # pragma: hot_bands_or_thermal
        ground_and_excited_state_einsums = _generate_eT_zhz_einsums(Proj, operators, lhs_rhs, only_ground_state=False,  opt_einsum=opt_einsum)
    else:
        ground_and_excited_state_einsums = [("raise Exception('Hot Band amplitudes not implemented!')", ), ]*2

    # the ordering of the functions is linked to the output ordering from `_generate_eT_zhz_einsums`
    # they must be in the same order
    # list = ['HZ', ]
    debugging_list = ['HZ', 'eT_HZ']
    # for i, term_type in enumerate(['HZ', ]):  # ['h', 'zh', 'hz', 'zhz']
    for i, term_type in enumerate(debugging_list):  # ['h', 'zh', 'hz', 'zhz']

        # the name of the function
        if lhs_rhs == 'RHS':
            if not opt_einsum:
                func_name = f"add_{specifier_string}_{term_type}_terms"
            else:
                func_name = f"add_{specifier_string}_{term_type}_terms_optimized"
        elif lhs_rhs == 'LHS':
            if not opt_einsum:
                func_name = f"compute_{specifier_string}_{term_type}_LHS"
            else:
                func_name = f"compute_{specifier_string}_{term_type}_LHS_optimized"

        # the positional arguments it takes (no keyword arguments are used currently)
        if lhs_rhs == 'RHS':
            if not opt_einsum:
                positional_arguments = "R, ansatz, truncation, t_args, h_args, z_args"
            else:
                positional_arguments = "R, ansatz, truncation, t_args, h_args, z_args, opt_einsum"
        elif lhs_rhs == 'LHS':
            if not opt_einsum:
                positional_arguments = "Z, ansatz, truncation, t_conj, dT, z_args, dz_args"
            else:
                positional_arguments = "Z, ansatz, truncation, t_conj, dT, z_args, dz_args, opt_einsum"

        # specify if the function is only calculating (H) * (Z) terms
        if term_type == 'HZ':
            vib_docstring = (
                f"These terms have no vibrational contribution from the e^T operator.{four_tabbed_newline}"
                f"This reduces the number of possible non-zero permutations of creation/annihilation operators."
            )
        # or (e^T) * (H) * (Z) terms
        elif term_type == 'eT_HZ':
            vib_docstring = (
                f"These terms include the vibrational contributions from the e^T operator.{four_tabbed_newline}"
                "This increases the number of possible non-zero permutations of creation/annihilation operators."
            )

        # specify the docstring based on whether or not this is an optimized function
        if not opt_einsum:
            opt_docstring = f"Calculate the {Proj} {term_type} terms.{four_tabbed_newline}"
        else:
            opt_docstring = f"Optimized calculation of the {Proj} {term_type} terms.{four_tabbed_newline}"

        # glue all these strings together in a specific manner to form the function definition
        function_string = f'''
            def {func_name}({positional_arguments}):
                """ {opt_docstring}{vib_docstring}
                """

                if ansatz.ground_state:
                    {five_tabbed_newline.join(ground_state_only_einsums[i])}
                else:
                    {five_tabbed_newline.join(ground_and_excited_state_einsums[i])}

                return
        '''

        """
            remove 3 consecutive tabs from the multi-line string `function_string`
            this is because we use triple single quotes over multiple lines
            therefore introducing 3 extra tabs of indentation that we DO NOT want
            to be present when we write the string to a file
        """
        function_string = "\n".join([line[tab_length*3:].rstrip() for line in function_string.splitlines()])

        # add an additional line between each function
        return_string += function_string + '\n'

    return return_string


def _wrap_eT_zhz_generation(master_omega, operators, lhs_rhs, only_ground_state=False, opt_einsum=False):
    """ x """
    return_string = ""

    for i, Proj in enumerate(master_omega.operator_list):

        # only print the header when we change rank (from linear to quadratic for example)
        if Proj.rank > master_omega.operator_list[i-1].rank:
            return_string += spaced_named_line(f"RANK {Proj.rank:2d} FUNCTIONS", s2) + '\n'

        # header
        return_string += '\n' + named_line(f"{Proj} TERMS", s2//2)

        # functions
        return_string += _construct_eT_zhz_compute_function(Proj, operators, lhs_rhs, only_ground_state, opt_einsum)

    return return_string


def _write_master_eT_zhz_compute_function(Proj, lhs_rhs, opt_einsum=False):
    """ Write the wrapper function which `vibronic_hamiltonian.py` calls. """

    specifier_string = f"m{Proj.m}_n{Proj.n}"

    tab = ' '*4
    four_tabs = tab*4

    # shared by all functions
    truncation_checks = 'truncation.confirm_at_least_singles()'
    truncation_checks += f'\n{four_tabs}truncation.confirm_at_least_doubles()' if ((Proj.m >= 2) or (Proj.n >= 2)) else ''
    truncation_checks += f'\n{four_tabs}truncation.confirm_at_least_triples()' if ((Proj.m >= 3) or (Proj.n >= 3)) else ''
    truncation_checks += f'\n{four_tabs}truncation.confirm_at_least_quadruples()' if ((Proj.m >= 4) or (Proj.n >= 4)) else ''
    truncation_checks += f'\n{four_tabs}truncation.confirm_at_least_quintuples()' if ((Proj.m >= 5) or (Proj.n >= 5)) else ''
    truncation_checks += f'\n{four_tabs}truncation.confirm_at_least_sextuples()' if ((Proj.m >= 6) or (Proj.n >= 6)) else ''

    # shared by all functions
    if lhs_rhs == 'RHS':
        common_positional_arguments = 'ansatz, truncation, t_args, h_args, z_args'
        tensor = 'R'
        step_name = 'add'
        t_type = 'terms'
    elif lhs_rhs == 'LHS':
        common_positional_arguments = 'ansatz, truncation, t_conj, dT, z_args, dz_args'
        tensor = 'dz'
        step_name = 'compute'
        t_type = 'LHS'

    if not opt_einsum:
        func_string = f'''
            def compute_{specifier_string}_amplitude(A, N, {common_positional_arguments}):
                """Compute the {Proj} amplitude."""
                {truncation_checks:s}

                # the residual tensor
                {tensor} = np.zeros(shape=({', '.join(['A',] + ['N',]*Proj.rank)}), dtype=complex)

                # {step_name} the terms
                {step_name}_{specifier_string}_HZ_{t_type}({tensor}, {common_positional_arguments})
                {step_name}_{specifier_string}_eT_HZ_{t_type}({tensor}, {common_positional_arguments})
                return {tensor}

        '''
    else:
        func_string = f'''
            def compute_{specifier_string}_amplitude_optimized(A, N, {common_positional_arguments}, opt_paths):
                """Compute the {Proj} amplitude."""
                {truncation_checks:s}

                # the residual tensor
                {tensor} = np.zeros(shape=({', '.join(['A',] + ['N',]*Proj.rank)}), dtype=complex)

                # unpack the optimized paths
                optimized_HZ_paths, optimized_eT_HZ_paths = opt_paths

                # {step_name} the terms
                {step_name}_{specifier_string}_HZ_{t_type}_optimized({tensor}, {common_positional_arguments}, optimized_HZ_paths)
                {step_name}_{specifier_string}_eT_HZ_{t_type}_optimized({tensor}, {common_positional_arguments}, optimized_eT_HZ_paths)
                return {tensor}

        '''

    # remove three indents from the multi-line string `func_string`
    lines = func_string.splitlines()
    # 3 consecutive tabs that we are ignoring/skipping
    trimmed_string = "\n".join([line[tab_length*3:] for line in lines])

    return trimmed_string


def _generate_eT_zhz_python_file_contents(truncations, **kwargs):
    """ Return a string containing the python code.
    Requires the following header: `"import numpy as np\nfrom math import factorial"`.
    """

    # unpack kwargs
    only_ground_state = kwargs['only_ground_state']
    lhs_rhs = kwargs['lhs_rhs']
    # unpack truncations
    _verify_eT_z_t_truncations(truncations)
    maximum_h_rank = truncations[tkeys.H]
    maximum_cc_rank = truncations[tkeys.CC]
    maximum_T_rank = truncations[tkeys.T]
    eT_taylor_max_order = truncations[tkeys.eT]
    omega_max_order = truncations[tkeys.P]

    # generate our quantum mechanical operators
    master_omega = generate_omega_operator(maximum_cc_rank, omega_max_order)
    H = generate_pruned_H_operator(maximum_h_rank)  # remember this is not just ordinary H
    Z = generate_z_operator(maximum_cc_rank, only_ground_state)
    eT_taylor_expansion = generate_eT_taylor_expansion(maximum_T_rank, eT_taylor_max_order)

    if True:  # until we work with thermal equations this should always be true (need to remove when working on thermal stuff)
        master_omega.operator_list[:] = [p for p in master_omega.operator_list if p.m == 0]

    # stick them in a tuple
    operators = master_omega, H, Z, eT_taylor_expansion

    # ------------------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------------------- #
    # header for default functions (as opposed to the optimized functions)
    string = long_spaced_named_line("DEFAULT FUNCTIONS", l2)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("INDIVIDUAL TERMS", l2) + '\n\n'
    # the actual code
    string += _wrap_eT_zhz_generation(master_omega, operators, lhs_rhs, only_ground_state)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("RESIDUAL FUNCTIONS", l2)
    # the wrapper functions that call the code made inside `_wrap_eT_zhz_generation`
    string += "".join([
        _write_master_eT_zhz_compute_function(Proj, lhs_rhs)
        for Proj in master_omega.operator_list
    ])

    # ------------------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------------------- #
    # header for optimized functions
    string += long_spaced_named_line("OPTIMIZED FUNCTIONS", l2-1)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("INDIVIDUAL TERMS", l2) + '\n\n'
    # the actual code
    string += _wrap_eT_zhz_generation(master_omega, operators, lhs_rhs, only_ground_state, opt_einsum=True)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("RESIDUAL FUNCTIONS", l2)
    # the wrapper functions that call the code made inside `_wrap_eT_zhz_generation`
    string += "".join([
        _write_master_eT_zhz_compute_function(Proj, lhs_rhs, opt_einsum=True)
        for Proj in master_omega.operator_list
    ])

    # ------------------------------------------------------------------------------------------- #

    # header for optimized paths function
    string += '\n' + named_line("OPTIMIZED PATHS FUNCTION", l2)

    # write the code for generating optimized paths for full CC, this is probably different than the W code?!?
    # maybe... im not sure?

    return string


def generate_eT_zhz_python(truncations, **kwargs):
    """Generates and saves to a file the code to calculate the terms for the full CC approach."""

    # start with the import statements
    file_data = code_import_statements_module.eT_zhz_import_statements

    # write the functions to calculate the W operators
    file_data += _generate_eT_zhz_python_file_contents(truncations, **kwargs)

    # save data
    with open(kwargs['path'], 'w') as fp:
        fp.write(file_data)

    return
