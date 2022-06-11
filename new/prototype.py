"""
Program to print permutations of P^a_bH^c_dZ^e_f that satisfy the following constraints

a + c + e = b + d +f
a <= d + f
b <= c + e
c <= b + f
d <= a + e
e <= b + d
f <= a + c

"""

# system imports
import itertools as it
from collections import Counter

# third party imports
import numpy as np
from scipy import optimize

# local imports

letter_array = ['a', 'b', 'c', 'd', 'e', 'f']


def _older_fancy_string(constraint):
    """ x """
    c_string = " + ".join([
        f"{n: d}{letter_array[i]}"
        for i, n in enumerate(constraint)
    ])
    c_string = c_string.replace('+ -', '- ')
    c_string = c_string.replace('+  ', '+ ')
    return c_string


def fancy_constraint_string(constraint):
    """ Uses the `letter_array` and some string processing to
    try its best to make human readable summary of the given
    equality constraint.

    For a constraint of
        [1, -1, 1, -1, 1, -1]

    produces two strings
        lhs = 'a + c + e'
        rhs = 'b + d + f'

    which can be joined to produce
        'a + c + e == b + d + f'


    For a constraint of
        [1, 0, 0, -1, 0, -1]

    produces two strings
        lhs = 'a'
        rhs = 'b + f'

    which can be joined to produce
        'a <= b + f'
    """

    lhs, rhs = '', ''

    lhs = " + ".join([
        f"{n:d}{letter_array[i]}"
        for i, n in enumerate(constraint)
        if n > 0
    ])

    rhs = " + ".join([
        f"{np.abs(n):d}{letter_array[i]}"
        for i, n in enumerate(constraint)
        if n < 0
    ])

    # lazy replacement of 1 as long as all numbers are single digit
    if np.all(np.abs(constraint) < 10):
        lhs = lhs.replace('1', '')
        rhs = rhs.replace('1', '')

    return lhs, rhs


def fancy_print_equalities(equality_matrix):
    """ wrapper function """

    nof_rows = equality_matrix.shape[0]

    print("Equality constraints:")
    for i in range(nof_rows):
        constraint = equality_matrix[i]
        lhs, rhs = fancy_constraint_string(constraint)
        print(f"{lhs:>15} == {rhs:<15}")

    print("")
    return


def fancy_print_inequalities(upper_bound_matrix):
    """ wrapper function """

    nof_rows = upper_bound_matrix.shape[0]

    print("Inequality constraints:")
    for i in range(nof_rows):
        constraint = upper_bound_matrix[i]
        lhs, rhs = fancy_constraint_string(constraint)
        print(f"{lhs:>15s} <= {rhs:<15s}")

    print("")
    return


def first_try(nof_variables):
    """ Exploring the use of scipy.optimize.linprog """

    # We have no linear objective function to optimize
    coefficients = np.zeros(nof_variables)

    equality_matrix = np.array([
        [1, -1, 1, -1, 1, -1],
    ])

    equality_vector = np.array([0, ])

    fancy_print_equalities(equality_matrix)

    # the inequality constraint matrix
    # It has one row per constraint, and one column per variable { a, b, c, d, e, f }
    upper_bound_matrix = np.array([
        [1, 0, 0, -1, 0, -1],
        [0, 1, -1, 0, -1, 0],
        [0, -1, 1, 0, 0, -1],
        [-1, 0, 0, 1, -1, 0],
        [0, -1, 0, -1, 1, 0],
        [-1, 0, -1, 0, 0, 1]
    ])
    nof_equations = len(upper_bound_matrix)

    # the inequality constraint vector
    upper_bound_vector = np.array([0, ] * nof_equations)

    fancy_print_inequalities(upper_bound_matrix)

    # bounds defines the minimum and maximum value for each variable;
    # we will redefine it in each iteration of the loop
    bounds = [(0, 5), ] * nof_variables

    # store solutions to optimization problems in here
    solution_values, solution_objects = [], []

    # define min max values specific to each operator
    p_max, h_max, z_max = 3, 2, 4
    min_max_dict = {'p': (0, p_max), 'h': (0, h_max), 'z': (0, z_max)}

    # possible methods
    methods = ["highs-ds", "highs-ipm", "highs", "interior-point", "revised simplex", "simplex"]
    method = methods[4]

    # produce list of permutations
    p_perms = [[p_m, p_n] for p_m, p_n in it.product(range(*min_max_dict['p']), repeat=2)]
    h_perms = [[h_m, h_n] for h_m, h_n in it.product(range(*min_max_dict['h']), repeat=2)]
    z_perms = [[z_m, z_n] for z_m, z_n in it.product(range(*min_max_dict['z']), repeat=2)]

    permutation_list = [list(it.chain(*a)) for a in it.product(p_perms, h_perms, z_perms)]

    # We run a nested loop over the values min to max for each variable, making sure we test all values of each variable
    print(f"Beginning big loop (Using {method = })")
    for perm in permutation_list:

        a, b, c, d, e, f = perm
        # redefine the bounds array
        bounds = [
            # projection bounds
            (a, p_max), (b, p_max),
            # Hamiltonian bounds
            (c, h_max), (d, h_max),
            # z bounds
            (e, z_max), (f, z_max),
        ]

        # try to optimize
        solution = optimize.linprog(
            coefficients,
            A_ub=upper_bound_matrix, b_ub=upper_bound_vector,
            A_eq=equality_matrix, b_eq=equality_vector,
            bounds=bounds,
            method=method,
        )
        solution_objects.append(solution)
        solution_values.append(tuple(solution.x))
    print("Finished Big Loop!\n")

    # remove duplicates, although not guaranteed to be sorted the same
    unique_solutions = list(set(solution_values))
    # sort them
    unique_solutions.sort()

    print(f"Found   {len(solution_values)} solutions")
    print(f"Only    {len(unique_solutions)} are unique")
    print(f"Removed {len(solution_values) - len(unique_solutions)} duplicates\n")

    # print a few solutions just to see?
    print(f"First three solutions: {unique_solutions[0:3]}")
    print(f"Last three solutions: {unique_solutions[-3:]}\n")

    # check how many different status we obtained (in case some small issues)
    status_list = [obj.status for obj in solution_objects]
    count_dict = Counter(status_list)

    print(
        "Analysis of linprog:\n"
        f"{count_dict[0]:6d} occurrences of 'Optimization terminated successfully.'\n"
        f"{count_dict[1]:6d} occurrences of 'Iteration limit reached.'\n"
        f"{count_dict[2]:6d} occurrences of 'Problem appears to be infeasible.'\n"
        f"{count_dict[3]:6d} occurrences of 'Problem appears to be unbounded.'\n"
        f"{count_dict[4]:6d} occurrences of 'Numerical difficulties encountered.'\n"
    )

    non_integer_solutions = [
        res
        for res in unique_solutions
        if not np.all([n.is_integer() for n in res])
    ]

    # if there are any non integer solutions print them
    if non_integer_solutions != []:
        print(non_integer_solutions)
        print(f"There are {len(non_integer_solutions)} non integer solutions\n")

    # try to do a simple writing of terms
    file_contents = ''

    # write all the latex equations
    for solution in unique_solutions:
        a, b, c, d, e, f = map(int, solution)

        file_contents += r'\[ '
        file_contents += f"P^{{{a:d}}}_{{{b:d}}}  "
        file_contents += f"H^{{{c:d}}}_{{{d:d}}}  "
        file_contents += f"Z^{{{e:d}}}_{{{f:d}}}"
        file_contents += r'\]'
        file_contents += '\n'

    # header and footer for latex file
    header = (
        r'\documentclass{article}'
        '\n'
        r'\usepackage{multicol}'
        '\n'
        r'\begin{document}'
        '\n'
        r'\begin{multicols}{6}'
        '\n'
    )
    footer = (
        r'\end{multicols}'
        '\n'
        r'\end{document}'
        '\n'
    )

    # write to file
    with open('first_try.tex', 'w') as fp:
        fp.write(header + file_contents + footer)


if (__name__ == '__main__'):
    """ x """

    number_of_variables = 6
    first_try(number_of_variables)
