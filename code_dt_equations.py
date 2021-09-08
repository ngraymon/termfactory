# system imports

# third party imports

# local imports
from namedtuple_defines import t_term_namedtuple


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

