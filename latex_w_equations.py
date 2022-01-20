# system imports
import os
import itertools as it
from collections import namedtuple

# third party imports

# local imports
from latex_defines import *
from common_imports import tab
import reference_latex_headers as headers
from namedtuple_defines import t_namedtuple_latex, w_namedtuple_latex
from helper_funcs import generate_partitions_of_n


# ----------------------------------------------------------------------------------------------- #
# -------------------------------- Latex of W operators ----------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

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
    if os.path.isfile(path): # pragma: no cover

        # read the entire file contents
        with open(path, 'r') as fp:
            file_contents = fp.readlines()

        # keep only the header
        header = ''.join(file_contents[0:29])

        # write the new header with latex code attached
        with open(path, 'w') as fp:
            fp.write(header + latex_code + r'\end{document}')

    # otherwise write a new file
    else: # pragma: no cover
        with open(path, 'w') as fp:
            fp.write(latex_code)

    return