# system imports

# third party imports

# local imports
from namedtuple_defines import t_term_namedtuple
from helper_funcs import (
    generate_un_linked_disconnected_partitions_of_n,
    generate_linked_disconnected_partitions_of_n,
    unique_permutations,
)
from common_imports import tab, tab_length, old_print_wrapper

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
        if len(partition) == 1:  # no permutation is needed for this term # pragma: no cover
            # code never gets here I think? bc if order < 2 has a return above
            old_print_wrapper(partition, partition[0])
            # we have to space the line correct (how many tabs)
            if max(partition) >= 2:  
                print('reached')
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
        if len(partition) == 1:  # no permutation is needed for this term # pragma: no cover
            # order 1 can't get here?
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

        if len(partition) == order:  # pragma: no cover, code never gets here
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
