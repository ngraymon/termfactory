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
from truncations import _verify_fcc_truncations
from truncation_keys import TruncationsKeys as tkeys


# temp logging fix
import log_conf

log = log_conf.get_filebased_logger(f'{__name__}.txt', submodule_name=__name__)
header_log = log_conf.HeaderAdapter(log, {})
subheader_log = log_conf.SubHeaderAdapter(log, {})

##########################################################################################
# Defines for labels and spacing

s1, s2, s3 = 75, 28, 25
l1, l2, l3 = 109, 45, 41

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


# ----------------------------------------------------------------------------------------------- #
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


# ----------------------------------------------------------------------------------------------- #
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

    raise Exception("Shouldn't get here")  # pragma: no cover


def _write_cc_einsum_python_from_list(truncations, t_term_list, opt_einsum=False, trunc_obj_name='truncation'):
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

    maximum_h_rank = truncations[tkeys.H]
    maximum_cc_rank = truncations[tkeys.CC]

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
        log.debug(
            f"omega =        {omega.__str__()}"
            f"h =            {h.__str__()}"
            f"t_list =       {t_list.__str__()}"
            f"permutations = {permutations.__str__()}"
        )
        # if omega.rank == 1 and permutations != None:
        #     sys.exit(0)

        # we still need to account for output/omega permutations

        # -----------------------------------------------------------------------------------------
        # build with permutations
        hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor] = []
        log.debug(f"{hamiltonian_rank_list = }")

        if permutations is None:
            t_operands = ', '.join([f"t_args[({t.m_h + t.m_o}, {t.n_h + t.n_o})]" for t in t_list])

            e_a = _full_cc_einsum_electronic_components(t_list)
            v_a, remaining_indices = _full_cc_einsum_vibrational_components(h, t_list)

            # string = f"np.einsum('{summation_subscripts}', {h_operand}, {t_operands})"
            if remaining_indices == '':

                # prepare einsum string
                if not opt_einsum:
                    string = ", ".join([f"{e_a[i]}{v_a[i]}" for i in range(len(e_a))])
                    string = f"np.einsum('{string} -> ab{remaining_indices}', {h_operand}, {t_operands})"
                else:
                    string = f"next(optimized_einsum)({h_operand}, {t_operands})"

                # save it
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

            elif len(remaining_indices) >= 1:
                for perm in unique_permutations(remaining_indices):

                    # prepare einsum string
                    if not opt_einsum:
                        string = ", ".join([f"{e_a[i]}{v_a[i]}" for i in range(len(e_a))])
                        string = f"np.einsum('{string} -> ab{remaining_indices}', {h_operand}, {t_operands})"
                    else:
                        string = f"next(optimized_einsum)({h_operand}, {t_operands})"

                    # save it
                    hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

                if len(remaining_indices) >= 2:
                    # sys.exit(0)
                    pass

        elif len(unique_dict.keys()) == 1:

            t_operands = ', '.join([f"t_args[({t.m_h + t.m_o}, {t.n_h + t.n_o})]" for t in t_list])

            e_a = _full_cc_einsum_electronic_components(t_list)
            v_a, remaining_indices = _full_cc_einsum_vibrational_components(h, t_list)

            for perm in permutations:

                # prepare einsum string
                if not opt_einsum:
                    string = ", ".join([f"{e_a[0]}{v_a[0]}"] + [f"{e_a[i+1]}{v_a[p+1]}" for i, p in enumerate(perm)])
                    string = f"np.einsum('{string} -> ab{remaining_indices}', {h_operand}, {t_operands})"
                else:  # pragma: no cover
                    string = f"next(optimized_einsum)({h_operand}, {t_operands})"

                # save it
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

        elif len(unique_dict) > 1:

            e_a = _full_cc_einsum_electronic_components(t_list)
            v_a, remaining_indices = _full_cc_einsum_vibrational_components(h, t_list)

            for perm in permutations:

                # these can be quite long; so we prepare them here
                t_operands = ', '.join([
                    f"t_args[({t_list[i].m_h + t_list[i].m_o}, {t_list[i].n_h + t_list[i].n_o})]"
                    for i in perm
                ])

                # prepare einsum string
                if not opt_einsum:
                    string = ", ".join([f"{e_a[0]}{v_a[0]}"] + [f"{e_a[i+1]}{v_a[p+1]}" for i, p in enumerate(perm)])
                    string = f"np.einsum('{string} -> ab{remaining_indices}', {h_operand}, {t_operands})"
                else:  # pragma: no cover
                    string = f"next(optimized_einsum)({h_operand}, {t_operands})"

                # save it
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

        else:
            raise Exception('')  # pragma: no cover

    # -----------------------------------------------------------------------------------------
    # remove any duplicates
    # for h_rank_list in hamiltonian_rank_list:
    #     for t_rank_list in h_rank_list:
    #         for prefactor_list in t_rank_list:

    # -----------------------------------------------------------------------------------------

    def _handle_multiline_same_prefactor(return_list, prefactor, string_list, nof_tabs=0):
        """ Attempts to format multi-line summations in a visually appealing manner.

        If `string_list` has 2 or more elements then attempts to produce:
            R += <prefactor>(
                np.einsum(<contents>) +
                np.einsum(<contents>) +
                np.einsum(<contents>) +
                np.einsum(<contents>)
            )

        otherwise just returns a string like:
            R += <prefactor>np.einsum(<contents>)
        """

        tabber = tab*nof_tabs

        # multi-line case
        if len(string_list) > 1:
            # open the summation scope
            return_list.append(f"{tabber}R += {prefactor}(")

            # add multi lines of `<contents> +`
            for string in string_list:
                return_list.append(f"{tabber}{tab}{string} +")

            # remove the last plus symbol
            return_list[-1] = return_list[-1][:-2]

            # close the `+=` scope
            return_list.append(f"{tabber})")

        # singular line; just return simple `R += <contents>`
        else:
            return_list.append(f"{tabber}R += {prefactor}{string_list[0]}")

        return

    # do the fixed H with rank 0 (and no CC component)
    for prefactor, string_list in hamiltonian_rank_list[0][0].items():  # pragma: no cover
        _handle_multiline_same_prefactor(return_list, prefactor, string_list, nof_tabs=0)

    # do all the possible CC truncations with rank 1+ (for a fixed H with rank 0)
    for j in range(1, maximum_cc_rank+1):
        if hamiltonian_rank_list[0][j] != {}:

            # add the `h * (t*t* ... )` header string
            cc_if_statement_string = f"if {trunc_obj_name}.{taylor_series_order_tag[j]}:"
            return_list.append(cc_if_statement_string)

            for prefactor, string_list in hamiltonian_rank_list[0][j].items():
                _handle_multiline_same_prefactor(return_list, prefactor, string_list, nof_tabs=1)

            # prevent hanging if-statements by removing them if they have no code in their scope
            if return_list[-1] == cc_if_statement_string:  # pragma: no cover
                del return_list[-1]

    # do all the possible H's with rank 1+
    for i in range(1, maximum_h_rank+1):

        temp_list = []

        # add spacing
        temp_list.append('')

        # add the `h` header string
        h_if_statement_string = f"if {trunc_obj_name}.at_least_{hamiltonian_order_tag[i]}:"
        temp_list.append(h_if_statement_string)

        for prefactor, string_list in hamiltonian_rank_list[i][0].items():  # pragma: no cover
            _handle_multiline_same_prefactor(temp_list, prefactor, string_list, nof_tabs=1)

        # do all the possible CC truncations with rank 1+ (for all H's with rank 1+)
        for j in range(1, maximum_cc_rank+1):
            if hamiltonian_rank_list[i][j] != {}:

                # add the `h * (t*t* ... )` header string
                cc_if_statement_string = f"{tab}if {trunc_obj_name}.{taylor_series_order_tag[j]}:"
                temp_list.append(cc_if_statement_string)

                for prefactor, string_list in hamiltonian_rank_list[i][j].items():
                    _handle_multiline_same_prefactor(temp_list, prefactor, string_list, nof_tabs=2)

                # prevent hanging if-statements by removing them if they have no code in their scope
                if temp_list[-1] == cc_if_statement_string:  # pragma: no cover
                    del temp_list[-1]

        # prevent hanging if-statements by not including them if they have no code in their scope
        if temp_list[-1] == h_if_statement_string:
            continue

        # otherwise we can include this list since it actually contains einsum equations
        else:
            return_list.extend(temp_list)

    return return_list


def _generate_full_cc_einsums(omega_term, truncations, only_ground_state=False, opt_einsum=False):
    """Return a string containing python code to be placed into a .py file.
    This does all the work of generating the einsums.
    """

    # unpack truncations
    maximum_h_rank = truncations[tkeys.H]
    maximum_cc_rank = truncations[tkeys.CC]
    s_taylor_max_order = truncations[tkeys.S]

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
        _write_cc_einsum_python_from_list(truncations, fully, opt_einsum=opt_einsum),
        _write_cc_einsum_python_from_list(truncations, linked, opt_einsum=opt_einsum),
        _write_cc_einsum_python_from_list(truncations, unlinked, opt_einsum=opt_einsum),
    ]

    return return_list


def _generate_full_cc_compute_functions(omega_term, truncations, only_ground_state=False, opt_einsum=False):
    """ This builds the strings representing the `add_m0_n0_fully_connected_terms`
    functions
    """
    return_string = ""
    specifier_string = f"m{omega_term.m}_n{omega_term.n}"
    five_tab = "\n" + tab*5

    # ----------------------------------------------------------------------------------------------- #
    """ Preforms the bulk of the work!!!
    this part generates the ground state einsums
    this is the majority of the code that will be generated
    (most everything else is just glue + window dressing)
    """
    ground_state_only_einsums = _generate_full_cc_einsums(omega_term, truncations, only_ground_state=True, opt_einsum=opt_einsum)

    """
    the current code `_generate_full_cc_einsums` DOES produce "something" when asked to try and produce hot band residual equations
    but I do not trust the output as I haven't verified ANY of it, and I don't even think the code logic is correct
    additionally the theory here is still in development and may not be pushed forward, it makes no sense to try and write this
    code if there isn't any theory to inform the rules defining the equations
    """
    full_einsums = [("raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')", ), ]*3
    # ----------------------------------------------------------------------------------------------- #

    # for distinguishing the different types of lists of optimized einsum paths
    optnames = ['connected', 'linked', 'unlinked']

    # now we glue everything together
    for i, term_type in enumerate(['fully_connected', 'linked_disconnected', 'unlinked_disconnected']):

        # the name of the function
        if not opt_einsum:
            func_name = f"add_{specifier_string}_{term_type}_terms"
        else:
            func_name = f"add_{specifier_string}_{term_type}_terms_optimized"

        # the positional arguments it takes (no keyword arguments are used currently)
        if not opt_einsum:
            positional_arguments = "R, ansatz, truncation, h_args, t_args"
        else:
            positional_arguments = "R, ansatz, truncation, h_args, t_args" + f", opt_{optnames[i]}_path_list"

        # the docstring of the function
        if not opt_einsum:
            docstring = f"Calculate the {omega_term} {term_type} terms."
        else:
            docstring = f"Optimized calculation of the {omega_term} {term_type} terms."

        # if we need to unpack the optimize einsum paths
        if not opt_einsum:
            unpack_optimized_einsum = ''
        else:
            unpack_optimized_einsum = (
                f"\n{tab*4}# make an iterable out of the `opt_{optnames[i]}_path_list`"
                f"\n{tab*4}optimized_einsum = iter(opt_{optnames[i]}_path_list)"
                "\n"
            )

        # glue all these strings together in a specific manner to form the function definition
        function_string = f'''
            def {func_name}({positional_arguments}):
                """{docstring}"""
                {unpack_optimized_einsum}
                if ansatz.ground_state:
                    {five_tab.join(ground_state_only_einsums[i])}
                else:
                    {five_tab.join(full_einsums[i])}
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


# ----------------------------------------------------------------------------------------------- #
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
        return_string += _generate_full_cc_compute_functions(omega_term, truncations, only_ground_state, opt_einsum=opt_einsum)

    return return_string


def _write_master_full_cc_compute_function(omega_term, opt_einsum=False):
    """Write the wrapper function which `vibronic_hamiltonian.py` calls."""

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
            def compute_{specifier_string}_amplitude_optimized(A, N, ansatz, truncation, h_args, t_args, opt_path_lists):
                """Compute the {omega_term} amplitude."""
                truncation.confirm_at_least_singles()

                # the residual tensor
                R = np.zeros(shape=({', '.join(['A','A',] + ['N',]*omega_term.rank)}), dtype=complex)

                # unpack the optimized paths
                opt_connected_path_list, opt_linked_path_list, opt_unlinked_path_list = opt_path_lists[({omega_term.m}, {omega_term.n})]

                # add each of the terms
                add_{specifier_string}_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_connected_path_list)
                add_{specifier_string}_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_linked_path_list)
                add_{specifier_string}_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_unlinked_path_list)
                return R

        '''

    # remove three indents from the multi-line string `func_string`
    lines = func_string.splitlines()
    # 3 consecutive tabs that we are ignoring/skipping
    trimmed_string = "\n".join([line[tab_length*3:] for line in lines])

    return trimmed_string


# ----------------------------------------------------------------------------------------------- #
term_type_name_list = ['fully_connected', 'linked_disconnected', 'unlinked_disconnected']


def _A_A_term_shape_string(order):
    """Return the string `(A, A, N, ...)` with `order` number of `N`'s."""
    return f"({', '.join(['A','A',] + ['N',]*order)})"


def _write_cc_optimized_paths_from_list(truncations, t_term_list, local_list_name, trunc_obj_name='truncation'):
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

    maximum_h_rank = truncations[tkeys.H]
    maximum_cc_rank = truncations[tkeys.CC]

    if t_term_list == []:
        return ["pass  # no valid terms here", ]

    return_list = []

    hamiltonian_rank_list = []
    for i in range(maximum_h_rank+1):
        hamiltonian_rank_list.append(dict([(i, {}) for i in range(maximum_cc_rank+1)]))

    for term in t_term_list:

        omega, h, t_list = term

        # we only care about the size of the tensor
        h_operand = _A_A_term_shape_string(h.m + h.n)
        assert h.rank == h.m + h.n

        # these terms are simply added to the residual and don't use einsum
        # so we don't need to compute an optimal path
        if (len(t_list) == 1) and t_list[0] == disconnected_namedtuple(0, 0, 0, 0):
            # return_list.append(f"R += {h_operand}")
            continue

        # logic about multiple permutations
        # generate lists of unique t terms
        permutations, unique_dict = _multiple_perms_logic(term)
        prefactor = _build_full_cc_python_prefactor(h, t_list)
        max_t_rank = max(_rank_of_t_term_namedtuple(t) for t in t_list)
        log.debug(
            f"omega =        {omega.__str__()}"
            f"h =            {h.__str__()}"
            f"t_list =       {t_list.__str__()}"
            f"permutations = {permutations.__str__()}"
        )

        # we still need to account for output/omega permutations
        # -----------------------------------------------------------------------------------------
        # build with permutations
        hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor] = []
        log.debug(f"{hamiltonian_rank_list = }")

        for t in t_list:
            assert sum(t) == (t.m_h + t.n_h + t.m_o + t.n_o)

        # no permutations
        if permutations is None:
            t_operands = ', '.join([_A_A_term_shape_string(sum(t)) for t in t_list])

            # we still need to call this function to get the `remaining_indices
            _, remaining_indices = _full_cc_einsum_vibrational_components(h, t_list)

            # if we trace over everything
            if remaining_indices == '':

                # compute contraction
                string = f"oe.contract_expression({h_operand}, {t_operands})"

                # save it
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

            # if there are some external labels
            elif len(remaining_indices) >= 1:
                for perm in unique_permutations(remaining_indices):

                    # compute contraction
                    string = f"oe.contract_expression({h_operand}, {t_operands})"

                    # save it
                    hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

        # only 1 permutation
        elif len(unique_dict.keys()) == 1:
            t_operands = ', '.join([_A_A_term_shape_string(sum(t)) for t in t_list])

            # there are some external labels
            for perm in permutations:

                # compute contraction
                string = f"oe.contract_expression({h_operand}, {t_operands})"

                # save it
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

        # multiple permutations
        elif len(unique_dict) > 1:

            for perm in permutations:

                t_operands = ', '.join([
                    _A_A_term_shape_string(sum(t_list[i]))
                    for i in perm
                ])

                # compute contraction
                string = f"oe.contract_expression({h_operand}, {t_operands})"

                # save it
                hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][prefactor].append(string)

        else:
            raise Exception('')  # pragma: no cover

    # -----------------------------------------------------------------------------------------
    # remove any duplicates
    # for h_rank_list in hamiltonian_rank_list:
    #     for t_rank_list in h_rank_list:
    #         for prefactor_list in t_rank_list:
    # -----------------------------------------------------------------------------------------

    def _handle_multiline_same_prefactor(return_list, string_list, nof_tabs=0):
        """ Attempts to format multi-line list extensions in a visually appealing manner.

        If `string_list` has 2 or more elements then attempts to produce:
            local_list_name.extend([
                oe.contract_expression(<contents>),
                oe.contract_expression(<contents>),
                oe.contract_expression(<contents>),
                oe.contract_expression(<contents>)
            ])

        otherwise just returns a string like:
            local_list_name.append(oe.contract_expression(<contents>))
        """

        tabber = tab*nof_tabs

        # multi-line case
        if len(string_list) > 1:
            # open the summation scope
            return_list.append(f"{tabber}{local_list_name}.extend([")

            # add multi lines of `oe.contract_expression(<contents>),`
            for string in string_list:
                return_list.append(f"{tabber}{tab}{string},")

            # remove the last plus comma
            return_list[-1] = return_list[-1][:-1]

            # close the `extend([` scope
            return_list.append(f"{tabber}])")

        # singular line; just return simple `local_list_name.append(oe.contract_expression(<contents>))`
        else:
            return_list.append(f"{tabber}{local_list_name}.append({string_list[0]})")

        return

    # do the fixed H with rank 0 (and no CC component)
    for prefactor, string_list in hamiltonian_rank_list[0][0].items():  # pragma: no cover
        _handle_multiline_same_prefactor(return_list, string_list, nof_tabs=0)

    # do all the possible CC truncations with rank 1+ (for a fixed H with rank 0)
    for j in range(1, maximum_cc_rank+1):
        if hamiltonian_rank_list[0][j] != {}:

            # add the `h * (t*t* ... )` header string
            cc_if_statement_string = f"if {trunc_obj_name}.{taylor_series_order_tag[j]}:"
            return_list.append(cc_if_statement_string)

            for prefactor, string_list in hamiltonian_rank_list[0][j].items():
                _handle_multiline_same_prefactor(return_list, string_list, nof_tabs=1)

            # prevent hanging if-statements by removing them if they have no code in their scope
            if return_list[-1] == cc_if_statement_string:  # pragma: no cover
                del return_list[-1]

    # do all the possible H's with rank 1+
    for i in range(1, maximum_h_rank+1):

        temp_list = []

        # add spacing
        temp_list.append('')

        # add the `h` header string
        h_if_statement_string = f"if {trunc_obj_name}.at_least_{hamiltonian_order_tag[i]}:"
        temp_list.append(h_if_statement_string)

        for prefactor, string_list in hamiltonian_rank_list[i][0].items():  # pragma: no cover
            _handle_multiline_same_prefactor(temp_list, string_list, nof_tabs=1)

        # do all the possible CC truncations with rank 1+ (for all H's with rank 1+)
        for j in range(1, maximum_cc_rank+1):
            if hamiltonian_rank_list[i][j] != {}:

                # add the `h * (t*t* ... )` header string
                cc_if_statement_string = f"{tab}if {trunc_obj_name}.{taylor_series_order_tag[j]}:"
                temp_list.append(cc_if_statement_string)

                for prefactor, string_list in hamiltonian_rank_list[i][j].items():
                    _handle_multiline_same_prefactor(temp_list, string_list, nof_tabs=2)

                # prevent hanging if-statements by removing them if they have no code in their scope
                if temp_list[-1] == cc_if_statement_string:  # pragma: no cover
                    del temp_list[-1]

        # prevent hanging if-statements by not including them if they have no code in their scope
        if temp_list[-1] == h_if_statement_string:  # pragma: no cover
            continue

        # otherwise we can include this list since it actually contains einsum equations
        else:
            return_list.extend(temp_list)

    if return_list == []:
        return_list.append("pass  # no valid terms here")

    return return_list


def _generate_full_cc_optimized_paths(omega_term, truncations, only_ground_state=False):
    """Return a string containing python code to be placed into a .py file.
    This does all the work of generating the optimized einsum paths.
    """

    # unpack truncations
    maximum_h_rank = truncations[tkeys.H]
    maximum_cc_rank = truncations[tkeys.CC]
    s_taylor_max_order = truncations[tkeys.S]

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

    return_list = [
        _write_cc_optimized_paths_from_list(truncations, fully, local_list_name=f"{term_type_name_list[0]}_opt_path_list"),
        _write_cc_optimized_paths_from_list(truncations, linked, local_list_name=f"{term_type_name_list[1]}_opt_path_list"),
        _write_cc_optimized_paths_from_list(truncations, unlinked, local_list_name=f"{term_type_name_list[2]}_opt_path_list"),
    ]

    return return_list


def _generate_optimized_paths_functions(omega_term, truncations, only_ground_state):
    """Return strings to write all the constant `oe.contract_expression` calls."""
    return_string = ""
    specifier_string = f"m{omega_term.m}_n{omega_term.n}"
    five_tab = "\n" + tab*5

    # ----------------------------------------------------------------------------------------------- #
    """ Preforms the bulk of the work!!!
    this part generates the optimized paths for the ground state einsums
    this is the majority of the code that will be generated
    (most everything else is just glue + window dressing)
    """
    ground_state_only_paths = _generate_full_cc_optimized_paths(omega_term, truncations, only_ground_state=True)

    """
    the current code `_generate_full_cc_optimized_paths` DOES produce "something" when asked to try and produce hot band residual equations
    but I do not trust the output as I haven't verified ANY of it, and I don't even think the code logic is correct
    additionally the theory here is still in development and may not be pushed forward, it makes no sense to try and write this
    code if there isn't any theory to inform the rules defining the equations
    """
    full_paths = [("raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')", ), ]*3
    # ----------------------------------------------------------------------------------------------- #

    for i, term_type in enumerate(term_type_name_list):

        # glue all these strings together in a specific manner to form the function definition
        function_string = f'''
            def compute_{specifier_string}_{term_type}_optimized_paths(A, N, ansatz, truncation):
                """Calculate optimized einsum paths for the {term_type} terms."""

                {term_type}_opt_path_list = []

                if ansatz.ground_state:
                    {five_tab.join(ground_state_only_paths[i])}
                else:
                    {five_tab.join(full_paths[i])}

                return {term_type}_opt_path_list
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


def _wrap_optimized_paths_generation(truncations, master_omega, s2, named_line, spaced_named_line, only_ground_state=False):
    """ x """
    return_string = ""

    for i, omega_term in enumerate(master_omega.operator_list):

        # only print the header when we change rank (from linear to quadratic for example)
        if omega_term.rank > master_omega.operator_list[i-1].rank:
            return_string += spaced_named_line(f"RANK {omega_term.rank:2d} OPTIMIZED PATHS", s2) + '\n'

        # header
        return_string += '\n' + named_line(f"{omega_term} OPTIMIZED PATHS", s2//2)
        # functions
        return_string += _generate_optimized_paths_functions(omega_term, truncations, only_ground_state)

    return return_string


def _write_grouped_optimized_paths_function(omega_term):
    """Write the projection-specific wrapper function for the optimized paths.
    This provides the optimized paths for a specific (m, n) projection
    """
    M, N = omega_term.m, omega_term.n

    specifier_string = f"m{M}_n{N}"

    func_string = f'''
        def compute_{specifier_string}_optimized_paths(A, N, ansatz, truncation):
            """Compute the optimized paths for this {omega_term}."""
            truncation.confirm_at_least_singles()

            connected_opt_path_list = compute_{specifier_string}_fully_connected_optimized_paths(A, N, ansatz, truncation)
            linked_opt_path_list = compute_{specifier_string}_linked_disconnected_optimized_paths(A, N, ansatz, truncation)
            unlinked_opt_path_list = compute_{specifier_string}_unlinked_disconnected_optimized_paths(A, N, ansatz, truncation)

            return_dict = {{
                ({M}, {N}): [connected_opt_path_list, linked_opt_path_list, unlinked_opt_path_list]
            }}

            return return_dict

    '''

    # remove three indents from the multi-line string `func_string`
    lines = func_string.splitlines()
    # 2 consecutive tabs that we are ignoring/skipping
    trimmed_string = "\n".join([line[tab_length*2:] for line in lines])

    return trimmed_string


def _write_optimized_master_paths_function(master_omega):
    """Return wrapper function for creating ALL optimized einsum paths"""

    main_strings = []

    for i, omega_term in enumerate(master_omega.operator_list):
        M, N = omega_term.m, omega_term.n
        main_strings.append(f"all_opt_path_lists[({M}, {N})] = compute_m{M}_n{N}_optimized_paths(A, N, ansatz, truncation)[({M}, {N})]\n")

    main_strings = f"{tab*3}".join(main_strings)

    func_string = f'''
        def compute_all_optimized_paths(A, N, ansatz, truncation):
            """Return dictionary containing optimized contraction paths.
            Calculates all optimized paths for the `opt_einsum` calls up to
                a maximum order of m+n={master_omega.maximum_rank} for a projection operator P^m_n
            """
            all_opt_path_lists = []

            {main_strings}
            return all_opt_path_lists

    '''

    # remove three indents from the multi-line string `func_string`
    lines = func_string.splitlines()
    # 2 consecutive tabs that we are ignoring/skipping
    trimmed_string = "\n".join([line[tab_length*2:] for line in lines])

    return trimmed_string


# ----------------------------------------------------------------------------------------------- #
def _generate_full_cc_python_file_contents(truncations, only_ground_state=False):
    """Return a string containing the python code to generate w operators up to (and including) `max_order`.
    Requires the following header: `"import numpy as np\nfrom math import factorial"`.
    """

    # unpack truncations
    _verify_fcc_truncations(truncations)
    maximum_cc_rank = truncations[tkeys.CC]
    omega_max_order = truncations[tkeys.P]

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
    # header
    string += '\n' + named_line("RESIDUAL FUNCTIONS", l2)
    # generate
    string += "".join([
        _write_master_full_cc_compute_function(omega_term, opt_einsum=True)
        for omega_term in master_omega.operator_list
    ])
    # # ------------------------------------------------------------------------------------------- #
    # header for optimized paths function
    string += long_spaced_named_line("OPTIMIZED PATHS FUNCTIONS", l3)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("INDIVIDUAL OPTIMIZED PATHS", l3) + '\n\n'
    # generate
    string += _wrap_optimized_paths_generation(truncations, master_omega, s3, named_line, spaced_named_line, only_ground_state)
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("GROUPED BY PROJECTION OPERATOR", l3)
    # generate
    string += "".join([
        _write_grouped_optimized_paths_function(omega_term)
        for omega_term in master_omega.operator_list
    ])
    # ----------------------------------------------------------------------- #
    # header
    string += '\n' + named_line("MASTER OPTIMIZED PATH FUNCTION", l3)
    # generate
    string += _write_optimized_master_paths_function(master_omega) + '\n'
    # ------------------------------------------------------------------------------------------- #
    return string


def generate_full_cc_python(truncations, **kwargs):
    """Generates and saves to a file the code to calculate the terms for the full CC approach."""

    # unpack kwargs
    only_ground_state = kwargs['only_ground_state']
    path = kwargs['path']

    # start with the import statements
    file_data = code_import_statements_module.full_cc_import_statements

    # write the functions to calculate the W operators
    file_data += _generate_full_cc_python_file_contents(truncations, only_ground_state)

    # save data
    with open(path, 'w') as fp:
        fp.write(file_data)

    return
