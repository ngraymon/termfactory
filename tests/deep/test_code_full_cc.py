# system imports
# # import pytest
from os.path import abspath, dirname, join
deep_dir = dirname(abspath(__file__))
root_dir = join(deep_dir, 'files')
classtest = 'test_code_full_cc'

# local imports
from . import context
import code_full_cc as cfcc
import latex_full_cc as fcc
from truncation_keys import TruncationsKeys as tkeys
from . import large_test_data

# global vars
zero_h_op_nt = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0])


class Test_generate_omega_operator:

    def test_basic(self):
        """x"""

        # input data
        function_output = cfcc.generate_omega_operator(maximum_cc_rank=2, omega_max_order=3)

        # run function
        expected_result = cfcc.omega_namedtuple(
            maximum_rank=2,
            operator_list=[
                cfcc.general_operator_namedtuple(name='', rank=0, m=0, n=0),
                cfcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1),
                cfcc.general_operator_namedtuple(name='d', rank=1, m=1, n=0),
                cfcc.general_operator_namedtuple(name='bb', rank=2, m=0, n=2),
                cfcc.general_operator_namedtuple(name='db', rank=2, m=1, n=1),
                cfcc.general_operator_namedtuple(name='dd', rank=2, m=2, n=0)
            ]
        )

        assert function_output == expected_result


class Test_rank_of_t_term_namedtuple:

    def test_basic(self):
        """x"""

        # input data
        t = fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)

        # run function
        function_output = cfcc._rank_of_t_term_namedtuple(t)
        expected_result = 1

        assert function_output == expected_result


class Test_full_cc_einsum_electronic_components:

    def test_basic(self):
        """x"""

        # input data
        t_list = (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),)

        # run function
        function_output = cfcc._full_cc_einsum_electronic_components(t_list)
        expected_result = ['ac', 'cb']

        assert function_output == expected_result


class Test_build_h_term_python_labels:

    def test_basic(self):
        """x"""

        # input data
        h = fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])

        # run function
        function_output = cfcc._build_h_term_python_labels(h, condense_offset=0)
        expected_result = ('i', '')

        assert function_output == expected_result


class Test_build_h_term_python_labels_zero_rank:

    def test_basic(self):
        """x"""

        # input data
        h = zero_h_op_nt

        # run function
        function_output = cfcc._build_h_term_python_labels(h, condense_offset=0)
        expected_result = ('', '')

        assert function_output == expected_result


class Test_build_t_term_python_labels:

    def test_basic(self):
        """x"""

        # input data
        term = fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
        offset_dict = {'summation_upper': 0, 'summation_lower': 0, 'unlinked': 0}

        # run function
        function_output = cfcc._build_t_term_python_labels(term, offset_dict)
        expected_result = ('i', '')

        assert function_output == expected_result

    def test_if_cond(self):
        """trigger (term.n_h > 0) or (term.n_o > 0)"""

        # input data
        term = fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0)
        offset_dict = {'summation_upper': 1, 'summation_lower': 0, 'unlinked': 0}

        # run function
        function_output = cfcc._build_t_term_python_labels(term, offset_dict)
        expected_result = ('i', '')

        assert function_output == expected_result


class Test_build_t_term_python_group:

    def test_basic(self):
        """x"""

        # input data
        t_list = (
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
        )
        h = fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])

        # run function
        function_output = cfcc._build_t_term_python_group(t_list, h)
        expected_result = (['i'], [''])

        assert function_output == expected_result


class Test_full_cc_einsum_vibrational_components:

    def test_basic(self):
        """x"""

        # input data
        t_list = (
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0),
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)
        )
        h = fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1])

        # run function
        function_output = cfcc._full_cc_einsum_vibrational_components(h, t_list)
        expected_result = (['ij', 'iz', 'jy'], 'zy')

        assert function_output == expected_result


class Test_simplify_full_cc_python_prefactor:

    def test_basic(self):
        """x"""

        # input data
        numerator_list = []
        denominator_list = ['factorial(2)', 'factorial(2)']

        # run function
        function_output = cfcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = ([], ['factorial(2)', 'factorial(2)'])

        assert function_output == expected_result

    def test_full(self):
        """x"""

        # input data
        numerator_list = ['factorial(2)']
        denominator_list = ['factorial(2)', 'factorial(3)']

        # run function
        function_output = cfcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = ([], ['factorial(3)'])

        assert function_output == expected_result

    def test_no_factors(self):
        """x"""

        # input data
        numerator_list = ['factorial(3)']
        denominator_list = ['factorial(4)', 'factorial(4)']

        # run function
        function_output = cfcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = (['factorial(3)'], ['factorial(4)', 'factorial(4)'])

        assert function_output == expected_result

    def test_if_case(self):
        """elif a < b"""

        # input data
        numerator_list = ['factorial(2)']
        denominator_list = ['factorial(2)', 'factorial(2)', 'factorial(3)']

        # run function
        function_output = cfcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result1 = ([], ['factorial(3)', 'factorial(2)'])
        expected_result2 = ([], ['factorial(2)', 'factorial(3)'])

        assert function_output == expected_result1 or function_output == expected_result2

    def test_if_case_2(self):
        """elif b > a"""

        # input data
        numerator_list = ['factorial(2)', 'factorial(2)', 'factorial(2)', 'factorial(2)', 'factorial(2)', 'factorial(2)']
        denominator_list = ['factorial(2)', 'factorial(2)']

        # run function
        function_output = cfcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = (['factorial(2)', 'factorial(2)', 'factorial(2)', 'factorial(2)'], [])

        assert function_output == expected_result


class Test_build_full_cc_python_prefactor:

    def test_basic(self):
        """x"""

        # input data
        h = fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        t_list = (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),)

        # run function
        function_output = cfcc._build_full_cc_python_prefactor(h, t_list, simplify_flag=True)
        expected_result = ''

        assert function_output == expected_result

    def test_single_h(self):
        """x"""

        # input data
        h = fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        t_list = (fcc.connected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)

        # run function
        function_output = cfcc._build_full_cc_python_prefactor(h, t_list, simplify_flag=True)
        expected_result = ''

        assert function_output == expected_result

    def test_if_case_1(self):
        """x > 1"""

        # input data
        h = fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1])
        t_list = (
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)
        )

        # run function
        function_output = cfcc._build_full_cc_python_prefactor(h, t_list, simplify_flag=True)
        expected_result = '1/(factorial(2)) * '

        assert function_output == expected_result

    def test_if_case_2(self):
        """h.m > 1"""

        # input data
        h = fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=1, n_o=0, m_t=[1], n_t=[0])
        t_list = (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1),)

        # run function
        function_output = cfcc._build_full_cc_python_prefactor(h, t_list, simplify_flag=True)
        expected_result = '1/(factorial(2)) * '

        assert function_output == expected_result


class Test_multiple_perms_logic:

    def test_basic(self):
        """x"""

        # input data
        term = [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1)
            )
        ]

        # run function
        function_output = cfcc._multiple_perms_logic(term)
        expected_result = (
            [
                (0, 1),
                (1, 0)
            ],
            {
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1): 1,
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1): 1
            }
        )

        assert function_output == expected_result

    def test_t_unique(self):
        """x"""

        # input data
        term = [
            fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0, 0], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]),
            (
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
            )
        ]

        # run function
        function_output = cfcc._multiple_perms_logic(term)
        expected_result = ([(0, 1), (1, 0)], {fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0): 2})

        assert function_output == expected_result

    def test_no_perms(self):
        """x"""

        # input data
        term = [
            fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
            fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]),
            (
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
            )
        ]

        # run function
        function_output = cfcc._multiple_perms_logic(term)
        expected_result = (None, {fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0): 1})

        assert function_output == expected_result


class Test_write_cc_einsum_python_from_list:

    def test_basic(self):
        """x"""

        # input data
        rank = 1
        truncations = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }
        t_term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=0, m_t=[0], n_t=[1]),
                zero_h_op_nt,
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                )
            ]
        ]

        # run function
        function_output = cfcc._write_cc_einsum_python_from_list(truncations, t_term_list, trunc_obj_name='truncation')
        expected_result = [
            'if truncation.singles:',
            "    R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])",
        ]

        assert function_output == expected_result

    def test_zero_dc_case(self):
        """Case where t_list has length 1 and is populated by a 0,0,0,0 disconnected named tuple"""

        # input data
        rank = 1
        truncations = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }
        t_term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=1, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),
                )
            ]
        ]

        # run function
        function_output = cfcc._write_cc_einsum_python_from_list(truncations, t_term_list, trunc_obj_name='truncation')
        expected_result = [
            'R += h_args[(1, 0)]',
        ]

        assert function_output == expected_result

    def test_0_indices(self):
        """x"""

        # input data
        rank = 0
        truncations = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }
        t_term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                zero_h_op_nt,
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),
                )
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]),
                (
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                )
            ]
        ]

        # run function
        function_output = cfcc._write_cc_einsum_python_from_list(truncations, t_term_list, trunc_obj_name='truncation')
        expected_result = [
            'R += h_args[(0, 0)]',
            '',
            'if truncation.at_least_linear:',
            '    if truncation.singles:',
            "        R += np.einsum('aci, cbi -> ab', h_args[(0, 1)], t_args[(1, 0)])"
        ]

        assert function_output == expected_result

    def test_single_unique_key(self):
        """x"""

        # input data
        rank = 2
        truncations = {
            tkeys.H: 2,
            tkeys.CC: 2,
            tkeys.S: 2,
            tkeys.P: 2
        }
        t_term_list = large_test_data.write_cc_einsum_python_from_list_single_unique_key.t_term_list

        # run function
        function_output = cfcc._write_cc_einsum_python_from_list(truncations, t_term_list, trunc_obj_name='truncation')

        assert function_output == large_test_data.write_cc_einsum_python_from_list_single_unique_key.output


class Test_write_cc_optimized_paths_from_list:

    def test_basic(self):
        """x"""

        # input data
        rank = 1
        truncations = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }
        t_term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=0, m_t=[0], n_t=[1]),
                zero_h_op_nt,
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                )
            ]
        ]
        local_list_name = 'fully_connected_opt_path_list'

        # run function
        function_output = cfcc._write_cc_optimized_paths_from_list(truncations, t_term_list, local_list_name, trunc_obj_name='truncation')
        expected_result = [
            'if truncation.singles:',
            "    fully_connected_opt_path_list.append(oe.contract_expression((A, A), (A, A, N)))",
        ]

        assert function_output == expected_result

    def test_zero_dc_case(self):
        """Case where t_list has length 1 and is populated by a 0,0,0,0 disconnected named tuple"""

        # input data
        rank = 1
        truncations = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }
        t_term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=1, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),
                )
            ]
        ]
        local_list_name = 'fully_connected_opt_path_list'

        # run function
        function_output = cfcc._write_cc_optimized_paths_from_list(truncations, t_term_list, local_list_name, trunc_obj_name='truncation')
        expected_result = [
            'pass  # no valid terms here',
        ]

        assert function_output == expected_result

    def test_0_indices(self):
        """x"""

        # input data
        rank = 0
        truncations = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }
        t_term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                zero_h_op_nt,
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),
                )
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]),
                (
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                )
            ]
        ]
        local_list_name = 'fully_connected_opt_path_list'

        # run function
        function_output = cfcc._write_cc_optimized_paths_from_list(truncations, t_term_list, local_list_name, trunc_obj_name='truncation')
        expected_result = [
            '',
            'if truncation.at_least_linear:',
            '    if truncation.singles:',
            "        fully_connected_opt_path_list.append(oe.contract_expression((A, A, N), (A, A, N)))"
        ]

        assert function_output == expected_result

    def test_single_unique_key(self):
        """x"""

        # input data
        rank = 2
        truncations = {
            tkeys.H: 2,
            tkeys.CC: 2,
            tkeys.S: 2,
            tkeys.P: 2
        }
        t_term_list = large_test_data.write_cc_optimized_paths_from_list_single_unique_key.t_term_list
        local_list_name = 'fully_connected_opt_path_list'

        # run function
        function_output = cfcc._write_cc_optimized_paths_from_list(truncations, t_term_list, local_list_name, trunc_obj_name='truncation')

        assert function_output == large_test_data.write_cc_optimized_paths_from_list_single_unique_key.output


class Test_generate_full_cc_einsums:

    def test_basic(self):
        """x"""

        # input data
        omega_term = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        truncations = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }

        # run function
        function_output = cfcc._generate_full_cc_einsums(
            omega_term,
            truncations,
            only_ground_state=False,
            opt_einsum=False
        )
        expected_result = [
            [
                'R += h_args[(1, 0)]',
            ],
            [
                'pass  # no valid terms here'
            ],
            [
                'if truncation.singles:',
                "    R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])",
            ]
        ]

        assert function_output == expected_result


class Test_generate_full_cc_compute_functions:

    def test_basic(self):
        """x"""

        # input data
        omega_term = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        truncations = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }

        # run function
        function_output = cfcc._generate_full_cc_compute_functions(
            omega_term,
            truncations,
            only_ground_state=False,
            opt_einsum=False
        )

        # open file
        func_name = "generate_full_cc_compute_functions_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result

    def test_opt_einsum(self):
        """x"""

        # input data
        omega_term = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        truncations = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }

        # run function
        function_output = cfcc._generate_full_cc_compute_functions(
            omega_term,
            truncations,
            only_ground_state=False,
            opt_einsum=True
        )

        # open file
        func_name = "generate_full_cc_compute_function_opt_einsum_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_generate_optimized_paths_functions:

    def test_basic(self):
        """x"""

        # input data
        omega_term = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        truncations = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }

        # run function
        function_output = cfcc._generate_optimized_paths_functions(
            omega_term,
            truncations,
            only_ground_state=False
        )

        # open file
        func_name = "generate_optimized_paths_functions_out.py"
        file_name = join(root_dir, classtest, func_name)
        # with open(file_name, 'w') as fp:
        #     fp.write(function_output)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_write_master_full_cc_compute_function:

    def test_basic(self):
        """x"""

        # input data
        omega_term = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)

        # run function
        function_output = cfcc._write_master_full_cc_compute_function(omega_term, opt_einsum=False)

        # open file
        func_name = "write_master_full_cc_compute_function_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result

    def test_opt_einsum(self):
        """x"""

        # input data
        omega_term = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)

        # run function
        function_output = cfcc._write_master_full_cc_compute_function(omega_term, opt_einsum=True)

        # open file
        func_name = "write_master_full_cc_compute_function_opt_einsum_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_wrap_full_cc_generation:

    def test_basic(self):
        """x"""

        # define test variables and functions
        s1, s2 = 75, 28
        spacing_line = "# " + "-"*s1 + " #\n"

        def named_line(name, width):
            return "# " + "-"*width + f" {name} " + "-"*width + " #"

        def spaced_named_line(name, width):
            return spacing_line + named_line(name, width) + '\n' + spacing_line

        # input data
        truncations = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }
        master_omega = fcc.omega_namedtuple(
            maximum_rank=1,
            operator_list=[
                fcc.general_operator_namedtuple(name='', rank=0, m=0, n=0),
                fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1),
                fcc.general_operator_namedtuple(name='d', rank=1, m=1, n=0)
            ]
        )

        # run function
        function_output = cfcc._wrap_full_cc_generation(
            truncations,
            master_omega,
            s2,
            named_line,
            spaced_named_line,
            only_ground_state=False,
            opt_einsum=False
        )

        # open file
        func_name = "wrap_full_cc_generation_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_generate_full_cc_python_file_contents:

    def test_basic(self):
        """run main gen"""

        # input data
        fcc_trunc = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }
        # run function
        function_output = cfcc._generate_full_cc_python_file_contents(fcc_trunc, only_ground_state=False)

        # open file
        func_name = "generate_full_cc_python_file_contents_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_generate_full_cc_python:

    def test_run_main(self, tmpdir):
        """runs main function and compares it to a reference file"""

        # run function
        output_path = join(tmpdir, "full_cc_equations.py")
        fcc_trunc = {
            tkeys.H: 1,
            tkeys.CC: 1,
            tkeys.S: 1,
            tkeys.P: 1
        }
        cfcc.generate_full_cc_python(fcc_trunc, only_ground_state=False, path=output_path)

        with open(output_path, 'r') as fp:
            file_data = fp.read()

        func_name = "code_output_full_cc_equations.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            reference_file_data = fp.read()

        assert file_data == reference_file_data, 'Fail'
