# system imports
import functools

# third party imports

# local imports
import helper_funcs
from helper_funcs import unique_permutations, named_line
from namedtuple_defines import general_operator_namedtuple, omega_namedtuple
from common_imports import tab, tab_length, summation_indices, unlinked_indices, old_print_wrapper
from latex_full_cc import (
    generate_full_cc_hamiltonian_operator,
    generate_s_taylor_expansion,
    generate_omega_operator,
    _filter_out_valid_s_terms,
    _seperate_s_terms_by_connection,
    _debug_print_valid_term_list,
    _debug_print_different_types_of_terms
)
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


def _full_cc_einsum_subscript_generator(h, t_list):  # pragma: no cover
    """ x """
    return_string = ""

    electronic_components = _full_cc_einsum_electronic_components(t_list)

    vibrational_components, remaining_indices = _full_cc_einsum_vibrational_components(h, t_list)

    summation_subscripts = ", ".join([
        f"{electronic_components[i]}{vibrational_components[i]}" for i in range(len(electronic_components))
    ])

    return_string = f"{summation_subscripts} -> ab{remaining_indices}"

    return return_string


def _full_cc_einsum_prefactor(term):  # pragma: no cover
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


def _write_cc_einsum_python_from_list(truncations, t_term_list, trunc_obj_name='truncation'):
    """ Do all the work here.
    Context: for a given omega(m,n) we generate (based on on `trunc_obj_name`'s value):
     - fully
     - linked-disconnected
     - disconnected
     terms.

    We do this by first building up all einsums in a big list of dicts of dicts of lists
    The `hamiltonian_rank_list` stores all the einsums, and they are glued together after
    the loop over `t_term_list` is done.
    1 - The outer list is indexed by the maximum rank of h, effectively checking the rank of h.
    2 - For a given dict in the outer list it is indexed by the maximum t-rank.
    3 - The value associated with a given t-rank is a dictionary
    4 - an individual key,value pair is a string representation of a prefactor
        in a minimal fractional form; if the prefactor is 0.25 then the string representation
        is `1/4`

    An example:
        `outer_list = [e1, e2, e3, ....]`
        `e1 = {'0': d1, '1': d2, 2: d3, ....}`
        ``
    """

    maximum_h_rank, maximum_cc_rank, _, _ = truncations

    if t_term_list == []:
        return ["pass  # no valid terms here", ]

    return_list = []

    hamiltonian_rank_list = []
    for i in range(maximum_h_rank+1):
        hamiltonian_rank_list.append(dict([(i, {}) for i in range(maximum_cc_rank+1)]))
    print('\n', hamiltonian_rank_list, '\n')
    import pdb; pdb.set_trace()

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
        print(hamiltonian_rank_list)

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
        print(hamiltonian_rank_list)

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
        _write_cc_einsum_python_from_list(truncations, fully),
        _write_cc_einsum_python_from_list(truncations, linked),
        _write_cc_einsum_python_from_list(truncations, unlinked),
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


def _wrap_full_cc_generation(truncations, master_omega, s2, named_line, spaced_named_line, only_ground_state=False, opt_einsum=False):
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
    # header for default functions (as opposed to the optimized functions)
    string = long_spaced_named_line("DEFAULT FUNCTIONS", l2)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("INDIVIDUAL TERMS", l2) + '\n\n'
    # generate
    string += _wrap_full_cc_generation(truncations, master_omega, s2, named_line, spaced_named_line, only_ground_state)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("RESIDUAL FUNCTIONS", l2)
    # generate
    string += "".join([
        _write_master_full_cc_compute_function(omega_term)
        for omega_term in master_omega.operator_list
    ])

    # ------------------------------------------------------------------------------------------- #
    # header for optimized functions
    string += long_spaced_named_line("OPTIMIZED FUNCTIONS", l2-1)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("INDIVIDUAL TERMS", l2) + '\n\n'
    # generate
    string += _wrap_full_cc_generation(truncations, master_omega, s2, named_line, spaced_named_line, only_ground_state, opt_einsum=True)
    # ----------------------------------------------------------------------- #
    # generate
    string += '\n' + named_line("RESIDUAL FUNCTIONS", l2)
    string += "".join([
        _write_master_full_cc_compute_function(omega_term, opt_einsum=True)
        for omega_term in master_omega.operator_list
    ])

    # ------------------------------------------------------------------------------------------- #
    # header for optimized paths function
    string += '\n' + named_line("OPTIMIZED PATHS FUNCTION", l2)
    # write the code for generating optimized paths for full CC, this is probably different than the W code?!?
    # maybe... im not sure?
    # both VEMX and VECC
    # ------------------------------------------------------------------------------------------- #
    return string


def generate_full_cc_python(truncations, only_ground_state=False, path="./full_cc_equations.py"):
    """Generates and saves to a file the code to calculate the terms for the full CC approach."""

    # start with the import statements
    file_data = code_import_statements_module.full_cc_import_statements

    # write the functions to calculate the W operators
    file_data += _generate_full_cc_python_file_contents(truncations, only_ground_state)

    # save data
    with open(path, 'w') as fp:
        fp.write(file_data)

    return
