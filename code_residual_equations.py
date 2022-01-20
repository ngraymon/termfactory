# system imports
from collections import namedtuple
import math

# third party imports
import numpy as np

# local imports
from namedtuple_defines import general_operator_namedtuple, hamiltonian_namedtuple, h_namedtuple, w_namedtuple
from helper_funcs import print_residual_data
from code_w_equations import taylor_series_order_tag
from common_imports import tab, old_print_wrapper

# rterm_namedtuple = namedtuple('rterm_namedtuple', ['prefactor', 'h', 'w'])
# we originally defined a class so that we can overload the `__eq__` operator
# because we needed to compare the rterm tuples, however I think I might have removed that code
# so Shanmei or I should check if we even need the class anymore


class residual_term(namedtuple('residual_term', ['prefactor', 'h', 'w'])):

    __slots__ = ()

    def __eq__(self, other_term):
        return bool(
            self.prefactor == other_term.prefactor and
            np.array_equal(self.h, other_term.h) and
            np.array_equal(self.w, other_term.w)
        )


# our Hamiltonian is
# H = (h_0 + omega + h^1 + h_1) + h^1_1 + h^2 + h_2
# but we can ignore the omega when calculating residuals as we add it back in at a later point
# so we use this H = h_0 + h^1 + h_1 + h^1_1 + h^2 + h_2


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

    if pre == "*(1/2)":    # case when 1/2 is the only prefactor, delete * sign  # pragma: no cover, looks like this can't happen
        pre = "1/2"
    elif denominator == numerator:
        # case when (1/1!) or (2/2!) is present, which will both be recognized as 1
        if "*(1/2)" in pre:
            if denominator == 1 or denominator == 2:
                pre = "(1/2)"     # 1*(1/2) = (1/2)
            else:  # pragma: no cover, looks like this can't happen
                pre = f"({numerator}/(2*{denominator}))"
        else:
            if denominator == 1 or denominator == 2:
                pre = ""   # use empty string to represent 1

    # case when 1/2 is multiplied to the prefactor
    elif "*(1/2)" in pre:
        if numerator % 2 == 0:  # pragma: no cover, looks like this can't happen
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

    if str_fac == "(1/2) * ":  # pragma: no cover
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
        prefactor = construct_prefactor(h_operator, order, True)

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
    if suppress_1_prefactor and (term.prefactor == 1.0):  # if we don't want to print prefactors that are 1  #pragma: no cover
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

        else:  # pragma: no cover
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
