# system imports
import itertools as it
import functools
import inspect
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

# temp logging fix
import log_conf

log = log_conf.get_filebased_logger('output.txt')
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

    # for each t term add 1 electronic label
    electronic_components += ['a', ] * len(t_list)

    # the H term always has 2
    electronic_components.append('ac')

    # we assume Z always contributes
    electronic_components.append('c')

    return electronic_components


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
    if (z_right.n > 0):
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
    if (term.m > 0):
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
    # print('\n\n\n\n\n')
    # print(f"{offset_dict = }")
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


def _eT_zhz_einsum_subscript_generator(h, t_list):  # pragma: no cover
    """ x """

    return_string = ""

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


def _build_eT_zhz_python_prefactor(h, t_list, simplify_flag=True):
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
        and t_list[0].m_lhs == 1
        and t_list[1].m_h == 1
        and t_list[1].m_lhs == 1
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
        numerator_list, denominator_list = _simplify_eT_zhz_python_prefactor(numerator_list, denominator_list)

    # glue the numerator and denominator together
    numerator = '1' if (numerator_list == []) else f"({' * '.join(numerator_list)})"
    denominator = '1' if (denominator_list == []) else f"({' * '.join(denominator_list)})"

    if numerator == '1' and denominator == '1':
        return ''
    else:
        return f"{numerator}/{denominator} * "


def _multiple_perms_logic(term):
    """ x """
    omega, t_list, h, z_pair, not_sure_what_this_one_is_for = term

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


def _write_third_eTz_einsum_python(rank, operators, t_term_list, trunc_obj_name='truncation', b_loop_flag=True):
    """ Still being written """

    H, Z, eT_taylor_expansion = operators
    log.info("Starting this function")

    if t_term_list == []:
        return ["pass  # no valid terms here", ]

    return_list = []

    hamiltonian_rank_list = []
    for i in range(H.maximum_rank+1):
        hamiltonian_rank_list.append(dict([(i, {}) for i in range(rank+1)]))

    for term in t_term_list:

        omega, t_list, h, z_pair, not_sure_what_this_one_is_for = term

        # define the indexing of the `h_args` dictionary
        h_operand = f"h_args[({h.m}, {h.n})]"

        # define the indexing of the `z_args` dictionary
        z_left, z_right = z_pair
        z_operand = f"z_args[({z_right.m}, {z_right.n})]"

        if (len(t_list) == 1) and t_list[0] == disconnected_namedtuple(0, 0, 0, 0):
            return_list.append(f"R += {h_operand}")
            continue

        # logic about multiple permutations
        # generate lists of unique t terms
        permutations, unique_dict = _multiple_perms_logic(term)
        prefactor = _build_eT_zhz_python_prefactor(h, t_list)
        max_t_rank = max(t.rank for t in t_list)
        old_print_wrapper(omega, h, t_list, permutations, sep='\n')

        # if omega.rank == 1 and permutations != None:
        #     sys.exit(0)

        # we still need to account for output/omega permutations

        # -----------------------------------------------------------------------------------------
        # build with permutations
        hamiltonian_rank_list[max(h.m, h.n)].setdefault(max_t_rank, {}).setdefault(prefactor, [])

        e_a = _eT_zhz_einsum_electronic_components(t_list, z_right, b_loop_flag)
        v_a, remaining_indices = _eT_zhz_einsum_vibrational_components(t_list, h, z_right, b_loop_flag)

        if permutations is None:
            t_operands = ', '.join([f"t_args[({t.m_lhs + t.m_h + t.m_r}, {t.n_lhs + t.n_h + t.n_r})]" for t in t_list])

            # string = f"np.einsum('{summation_subscripts}', {h_operand}, {t_operands})"
            if remaining_indices == '':
                string = ", ".join([f"{e_a[i]}{v_a[i]}" for i in range(len(e_a))])
                string = f"np.einsum('{string} -> a{remaining_indices}', {t_operands}, {h_operand}, {z_operand})"
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

            elif len(remaining_indices) >= 1:
                for perm in unique_permutations(remaining_indices):
                    string = ", ".join([f"{e_a[i]}{v_a[i]}" for i in range(len(e_a))])
                    string = f"np.einsum('{string} -> a{remaining_indices}', {t_operands}, {h_operand}, {z_operand})"
                    old_print_wrapper(perm)
                    old_print_wrapper(remaining_indices)
                    hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

                if len(remaining_indices) >= 2:
                    # sys.exit(0)
                    pass

        elif len(unique_dict.keys()) == 1:

            t_operands = ', '.join([f"t_args[({t.m_lhs + t.m_h + t.m_r}, {t.n_lhs + t.n_h + t.n_r})]" for t in t_list])

            for perm in permutations:
                string = ", ".join([f"{e_a[i]}{v_a[p]}" for i, p in enumerate(perm)] + [f"{e_a[-2]}{v_a[-2]}"] + [f"{e_a[-1]}{v_a[-1]}"])
                string = f"np.einsum('{string} -> a{remaining_indices}', {t_operands}, {h_operand}, {z_operand})"
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

        elif len(unique_dict) > 1:

            for perm in permutations:
                t_operands = ', '.join([
                    f"t_args[({t_list[i].m_lhs + t_list[i].m_h + t_list[i].m_r}, {t_list[i].n_lhs + t_list[i].n_h + t_list[i].n_r})]"
                    for i in perm
                ])
                string = ", ".join([f"{e_a[i]}{v_a[p]}" for i, p in enumerate(perm)] + [f"{e_a[-2]}{v_a[-2]}"] + [f"{e_a[-1]}{v_a[-1]}"])
                string = f"np.einsum('{string} -> a{remaining_indices}', {t_operands}, {h_operand}, {z_operand})"
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

        else:
            raise Exception('')

    # -----------------------------------------------------------------------------------------
    # remove any duplicates
    # for h_rank_list in hamiltonian_rank_list:
    #     for t_rank_list in h_rank_list:
    #         for prefactor_list in t_rank_list:

    # -----------------------------------------------------------------------------------------

    def _handle_multiline_same_prefactor(output_list, prefactor, string_list, nof_tabs=0):
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
            output_list.append(f"{tabber}R += {prefactor}(")

            # add each line
            for string in string_list:
                output_list.append(f"{tabber}{tab}{string} +")

            # remove the last plus symbol
            output_list[-1] = output_list[-1][:-2]

            # footer
            output_list.append(f"{tabber})")

        else:
            output_list.append(f"{tabber}R += {prefactor}{string_list[0]}")

        return

    h_contribution_list = []

    # h^0_0 with zero order Taylor series contributions
    for prefactor, string_list in hamiltonian_rank_list[0][0].items():
        _handle_multiline_same_prefactor(h_contribution_list, prefactor, string_list, nof_tabs=0)

    # loop over first order (and higher) Taylor series contributions
    for j in range(1, rank+1):

        # skip if empty
        if hamiltonian_rank_list[0][j] == {}:
            continue

        # if not empty
        h_contribution_list.append(f"if {trunc_obj_name}.{taylor_series_order_tag[j]}:")

        # h^0_0 with first order (and higher) Taylor series contributions
        for prefactor, string_list in hamiltonian_rank_list[0][j].items():
            _handle_multiline_same_prefactor(h_contribution_list, prefactor, string_list, nof_tabs=1)

    # loop over first order (and higher) h^i_j terms
    for i in range(1, H.maximum_rank+1):

        # skip if empty
        if hamiltonian_rank_list[i][0] == {}:
            continue

        # if not empty
        h_contribution_list.append(f"if {trunc_obj_name}.at_least_{hamiltonian_order_tag[i]}:")

        # h^i_j with zero order Taylor series contributions
        for prefactor, string_list in hamiltonian_rank_list[i][0].items():
            _handle_multiline_same_prefactor(h_contribution_list, prefactor, string_list, nof_tabs=1)

        # loop over first order (and higher) Taylor series contributions
        for j in range(1, rank+1):

            # skip if empty
            if hamiltonian_rank_list[i][j] == {}:
                continue

            # if not empty
            h_contribution_list.append(f"{tab}if {trunc_obj_name}.{taylor_series_order_tag[j]}:")

            # h^i_j with first order (and higher) Taylor series contributions
            for prefactor, string_list in hamiltonian_rank_list[i][j].items():
                _handle_multiline_same_prefactor(h_contribution_list, prefactor, string_list, nof_tabs=2)

    if h_contribution_list == []:
        return_list.append('pass')
    else:
        return_list.extend(h_contribution_list)

    return return_list


def _generate_eT_zhz_einsums(LHS, operators, only_ground_state=False, remove_f_terms=False, opt_einsum=False):
    """Return a string containing python code to be placed into a .py file.
    This does all the work of generating the einsums.

    LHS * (t*t*t) * H * Z

    This one basically needs to be like the t term stuff EXCEPT:
        - there is a single z term
        - it is always on the right side
        - always bond to projection operator in opposite dimension (^i _i)
    """
    H, Z, eT_taylor_expansion = operators

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
        Z.operator_list[:] = it.takewhile(lambda z: z.n == 0,  Z.operator_list)

        # remove all excited state t's
        # this is a really sloppy method so it should be improved in the future
        i_list, j_list= [], []

        for i, e1 in enumerate(eT_taylor_expansion):
            if not isinstance(e1, list):
                if e1.m > 0:
                    i_list.append(i)
                continue

            for j, e2 in enumerate(e1):
                if not isinstance(e2, list):
                    if e2.m > 0:
                        j_list.append(1)
                    continue

                k_list = [1 for k, e3 in enumerate(e2) if e3.m > 0]

                del e2[:sum(k_list)]
            del e1[:sum(j_list)]
        del eT_taylor_expansion[:sum(i_list)]

        eT_taylor_expansion = [x for x in eT_taylor_expansion if x != []]

    # do the terms without T contributions first
    zero_eT_term = eT_taylor_expansion[0]
    log.debug(zero_eT_term, "-"*100, "\n\n")
    _filter_out_valid_eTz_terms(LHS, zero_eT_term, H, None, Z, valid_zero_list)

    # then do the rest
    for eT_series_term in eT_taylor_expansion[1:]:
        log.debug(eT_series_term, "-"*100, "\n\n")

        # generate all valid combinations
        _filter_out_valid_eTz_terms(LHS, eT_series_term, H, None, Z, valid_term_list)

    if False:  # debug
        print('\n\n\n')

        # save in a human readable format to a file
        file_str = '\n]\n'.join([f'\n{tab}'.join(['['] + [str(y) for y in x]) for x in valid_term_list])
        with open('temp.txt', 'w') as fp:
            fp.write(file_str + '\n]')

        # for i, a in enumerate(valid_term_list):
        #     print(f"{i+1:>4d}", a)

        pdb.set_trace() if inspect.stack()[-1].filename == 'driver.py' else None

    if valid_zero_list == [] and valid_term_list == []:
        return ""

    # return _prepare_third_eTz_latex(valid_term_list, remove_f_terms=remove_f_terms)

    return_list = [
        _write_third_eTz_einsum_python(LHS.rank, operators, valid_zero_list),
        _write_third_eTz_einsum_python(LHS.rank, operators, valid_term_list),
    ]

    return return_list


def _construct_eT_zhz_compute_function(LHS, operators, only_ground_state=False, opt_einsum=False):
    """ x """

    return_string = ""
    specifier_string = f"m{LHS.m}_n{LHS.n}"
    four_tab = "\n" + tab*4
    five_tab = "\n" + tab*5

    # generate ground state einsums
    ground_state_only_einsums = _generate_eT_zhz_einsums(LHS, operators, only_ground_state=True, opt_einsum=opt_einsum)

    # generate ground + excited state einsums
    if not only_ground_state:
        einsums = _generate_eT_zhz_einsums(LHS, operators, only_ground_state=False,  opt_einsum=opt_einsum)
    else:
        einsums = [("raise Exception('Hot Band amplitudes not implemented!')", ), ]*2

    # the ordering of the functions is linked to the output ordering from `_generate_eT_zhz_einsums`
    # they must be in the same order
    for i, term_type in enumerate(['HZ', 'eT_HZ']):  # ['h', 'zh', 'hz', 'zhz']

        # the name of the function
        if not opt_einsum:
            func_name = f"add_{specifier_string}_{term_type}_terms"
        else:
            func_name = f"add_{specifier_string}_{term_type}_terms_optimized"

        # the positional arguments it takes (no keyword arguments are used currently)
        if not opt_einsum:
            positional_arguments = "R, ansatz, truncation, t_args, h_args, z_args"
        else:
            positional_arguments = "R, ansatz, truncation, t_args, h_args, z_args, opt_einsum"

        # the docstring of the function
        if not opt_einsum:
            docstring = (
                f"Calculate the {LHS} {term_type} terms.\n{four_tab}"
                f"These terms have no vibrational contribution from the e^T operator.\n{four_tab}"
                "This reduces the number of possible non-zero permutations of creation/annihilation operators."
            )
        else:
            docstring = (
                f"Optimized calculation of the {LHS} {term_type} terms.\n{four_tab}"
                f"These terms include the vibrational contributions from the e^T operator.\n{four_tab}"
                "This increases the number of possible non-zero permutations of creation/annihilation operators."
            )

        # glue all these strings together in a specific manner to form the function definition
        function_string = f'''
            def {func_name}({positional_arguments}):
                """{docstring}
                """

                if ansatz.ground_state:
                    {five_tab.join(ground_state_only_einsums[i])}
                else:
                    {five_tab.join(einsums[i])}

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


def _wrap_eT_zhz_generation(master_omega, operators, only_ground_state=False, opt_einsum=False):
    """ x """
    return_string = ""

    for i, LHS in enumerate(master_omega.operator_list):

        # only print the header when we change rank (from linear to quadratic for example)
        if LHS.rank > master_omega.operator_list[i-1].rank:
            return_string += spaced_named_line(f"RANK {LHS.rank:2d} FUNCTIONS", s2) + '\n'

        # header
        return_string += '\n' + named_line(f"{LHS} TERMS", s2//2)

        # functions
        return_string += _construct_eT_zhz_compute_function(LHS, operators, only_ground_state, opt_einsum)

    return return_string


def _write_master_eT_zhz_compute_function(LHS, opt_einsum=False):
    """ Write the wrapper function which `vibronic_hamiltonian.py` calls. """

    specifier_string = f"m{LHS.m}_n{LHS.n}"

    tab = ' '*4
    four_tabs = tab*4

    # shared by all functions
    truncation_checks = 'truncation.confirm_at_least_singles()'
    truncation_checks += f'\n{four_tabs}truncation.confirm_at_least_doubles()' if ((LHS.m >= 2) or (LHS.n >= 2)) else ''
    truncation_checks += f'\n{four_tabs}truncation.confirm_at_least_triples()' if ((LHS.m >= 3) or (LHS.n >= 3)) else ''
    truncation_checks += f'\n{four_tabs}truncation.confirm_at_least_quadruples()' if ((LHS.m >= 4) or (LHS.n >= 4)) else ''
    truncation_checks += f'\n{four_tabs}truncation.confirm_at_least_quintuples()' if ((LHS.m >= 5) or (LHS.n >= 5)) else ''
    truncation_checks += f'\n{four_tabs}truncation.confirm_at_least_sextuples()' if ((LHS.m >= 6) or (LHS.n >= 6)) else ''

    # shared by all functions
    common_positional_arguments = 'ansatz, truncation, t_args, h_args, z_args'

    if not opt_einsum:
        func_string = f'''
            def compute_{specifier_string}_amplitude(A, N, {common_positional_arguments}):
                """Compute the {LHS} amplitude."""
                {truncation_checks:s}

                # the residual tensor
                R = np.zeros(shape=({', '.join(['A','A',] + ['N',]*LHS.rank)}), dtype=complex)

                # add the terms
                add_{specifier_string}_HZ_terms(R, {common_positional_arguments})
                add_{specifier_string}_eT_HZ_terms(R, {common_positional_arguments})
                return R

        '''
    else:
        func_string = f'''
            def compute_{specifier_string}_amplitude_optimized(A, N, {common_positional_arguments}, opt_paths):
                """Compute the {LHS} amplitude."""
                {truncation_checks:s}

                # the residual tensor
                R = np.zeros(shape=({', '.join(['A','A',] + ['N',]*LHS.rank)}), dtype=complex)

                # unpack the optimized paths
                optimized_HZ_paths, optimized_eT_HZ_paths = opt_paths

                # add the terms
                add_{specifier_string}_HZ_terms_optimized(R, {common_positional_arguments}, optimized_HZ_paths)
                add_{specifier_string}_eT_HZ_terms_optimized(R, {common_positional_arguments}, optimized_eT_HZ_paths)
                return R

        '''

    # remove three indents from the multi-line string `func_string`
    lines = func_string.splitlines()
    # 3 consecutive tabs that we are ignoring/skipping
    trimmed_string = "\n".join([line[tab_length*3:] for line in lines])

    return trimmed_string


def _generate_eT_zhz_python_file_contents(truncations, only_ground_state=False):
    """ Return a string containing the python code.
    Requires the following header: `"import numpy as np\nfrom math import factorial"`.
    """

    # unpack truncations
    assert len(truncations) == 5, "truncations argument needs to be tuple of five integers!!"
    maximum_h_rank, maximum_cc_rank, maximum_T_rank, eT_taylor_max_order, omega_max_order = truncations

    # generate our quantum mechanical operators
    master_omega = generate_omega_operator(maximum_cc_rank, omega_max_order)
    H = generate_pruned_H_operator(maximum_h_rank)  # remember this is not just ordinary H
    Z = generate_z_operator(maximum_cc_rank, only_ground_state)
    eT_taylor_expansion = generate_eT_taylor_expansion(maximum_T_rank, eT_taylor_max_order)

    # stick them in a tuple
    operators = H, Z, eT_taylor_expansion

    # ------------------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------------------- #
    # header for default functions (as opposed to the optimized functions)
    string = long_spaced_named_line("DEFAULT FUNCTIONS", l2)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("INDIVIDUAL TERMS", l2) + '\n\n'
    # the actual code
    string += _wrap_eT_zhz_generation(master_omega, operators, only_ground_state)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("RESIDUAL FUNCTIONS", l2)
    # the wrapper functions that call the code made inside `_wrap_eT_zhz_generation`
    string += "".join([
        _write_master_eT_zhz_compute_function(LHS)
        for LHS in master_omega.operator_list
    ])

    # ------------------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------------------- #
    # header for optimized functions
    string += long_spaced_named_line("OPTIMIZED FUNCTIONS", l2-1)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("INDIVIDUAL TERMS", l2) + '\n\n'
    # the actual code
    string += _wrap_eT_zhz_generation(master_omega, operators, only_ground_state, opt_einsum=True)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("RESIDUAL FUNCTIONS", l2)
    # the wrapper functions that call the code made inside `_wrap_eT_zhz_generation`
    string += "".join([
        _write_master_eT_zhz_compute_function(LHS, opt_einsum=True)
        for LHS in master_omega.operator_list
    ])

    # ------------------------------------------------------------------------------------------- #

    # header for optimized paths function
    string += '\n' + named_line("OPTIMIZED PATHS FUNCTION", l2)

    # write the code for generating optimized paths for full CC, this is probably different than the W code?!?
    # maybe... im not sure?

    return string


def generate_eT_zhz_python(truncations, only_ground_state=False, path="./eT_zhz_equations.py"):
    """Generates and saves to a file the code to calculate the terms for the full CC approach."""

    # start with the import statements
    file_data = code_import_statements_module.eT_zhz_import_statements

    # write the functions to calculate the W operators
    file_data += _generate_eT_zhz_python_file_contents(truncations, only_ground_state)

    # save data
    with open(path, 'w') as fp:
        fp.write(file_data)

    return
