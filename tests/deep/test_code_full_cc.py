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
from . import test_vars as vars

# global vars
zero_h_op_nt = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0])


def test_generate_omega_operator():
    """x"""
    function_output = cfcc.generate_omega_operator(maximum_cc_rank=2, omega_max_order=3)
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


class Test_gen_full_cc_py_eqns:

    def test_rank_of_t_term_namedtuple(self):
        """x"""
        t = fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
        function_output = cfcc._rank_of_t_term_namedtuple(t)
        expected_result = 1
        assert function_output == expected_result

    def test_full_cc_einsum_electronic_components(self):
        """x"""
        t_list = (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),)
        function_output = cfcc._full_cc_einsum_electronic_components(t_list)
        expected_result = ['ac', 'cb']
        assert function_output == expected_result

    def test_build_h_term_python_labels(self):
        """x"""
        h = fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        function_output = cfcc._build_h_term_python_labels(h, condense_offset=0)
        expected_result = ('i', '')
        assert function_output == expected_result

    def test_build_h_term_python_labels_zero_rank(self):
        """x"""
        h = zero_h_op_nt
        function_output = cfcc._build_h_term_python_labels(h, condense_offset=0)
        expected_result = ('', '')
        assert function_output == expected_result

    def test_build_t_term_python_labels(self):
        """x"""
        term = fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
        offset_dict = {'summation_upper': 0, 'summation_lower': 0, 'unlinked': 0}
        function_output = cfcc._build_t_term_python_labels(term, offset_dict)
        expected_result = ('i', '')
        assert function_output == expected_result

    def test_build_t_term_python_labels_if_cond(self):
        """trigger (term.n_h > 0) or (term.n_o > 0)"""
        term = fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0)
        offset_dict = {'summation_upper': 1, 'summation_lower': 0, 'unlinked': 0}
        function_output = cfcc._build_t_term_python_labels(term, offset_dict)
        expected_result = ('i', '')
        assert function_output == expected_result

    def test_build_t_term_python_group(self):
        """x"""
        t_list = (
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
        )
        h = fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        function_output = cfcc._build_t_term_python_group(t_list, h)
        expected_result = (['i'], [''])
        assert function_output == expected_result

    def test_full_cc_einsum_vibrational_components(self):
        """x"""
        t_list = (
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0),
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)
        )
        h = fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1])
        function_output = cfcc._full_cc_einsum_vibrational_components(h, t_list)
        expected_result = (['ij', 'iz', 'jy'], 'zy')
        assert function_output == expected_result

    def test_simplify_full_cc_python_prefactor(self):
        """x"""
        numerator_list = []
        denominator_list = ['factorial(2)', 'factorial(2)']
        function_output = cfcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = ([], ['factorial(2)', 'factorial(2)'])
        assert function_output == expected_result

    def test_simplify_full_cc_python_prefactor_full(self):
        """x"""
        numerator_list = ['factorial(2)']
        denominator_list = ['factorial(2)', 'factorial(3)']
        function_output = cfcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = ([], ['factorial(3)'])
        assert function_output == expected_result

    def test_simplify_full_cc_python_prefactor_no_factors(self):
        """x"""
        numerator_list = ['factorial(3)']
        denominator_list = ['factorial(4)', 'factorial(4)']
        function_output = cfcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = (['factorial(3)'], ['factorial(4)', 'factorial(4)'])
        assert function_output == expected_result

    def test_simplify_full_cc_python_prefactor_if_case(self):
        """elif a < b"""
        numerator_list = ['factorial(2)']
        denominator_list = ['factorial(2)', 'factorial(2)', 'factorial(3)']
        function_output = cfcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result1 = ([], ['factorial(3)', 'factorial(2)'])
        expected_result2 = ([], ['factorial(2)', 'factorial(3)'])
        assert function_output == expected_result1 or function_output == expected_result2

    def test_simplify_full_cc_python_prefactor_if_case_2(self):
        """elif b > a"""
        numerator_list = ['factorial(2)', 'factorial(2)', 'factorial(2)', 'factorial(2)', 'factorial(2)', 'factorial(2)']
        denominator_list = ['factorial(2)', 'factorial(2)']
        function_output = cfcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = (['factorial(2)', 'factorial(2)', 'factorial(2)', 'factorial(2)'], [])
        assert function_output == expected_result

    def test_build_full_cc_python_prefactor(self):
        """x"""
        h = fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        t_list = (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),)
        function_output = cfcc._build_full_cc_python_prefactor(h, t_list, simplify_flag=True)
        expected_result = ''
        assert function_output == expected_result

    def test_build_full_cc_python_prefactor_single_h(self):
        """x"""
        h = fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        t_list = (fcc.connected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)
        function_output = cfcc._build_full_cc_python_prefactor(h, t_list, simplify_flag=True)
        expected_result = ''
        assert function_output == expected_result

    def test_build_full_cc_python_prefactor_if_case_1(self):
        """x > 1"""
        h = fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1])
        t_list = (
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)
        )
        function_output = cfcc._build_full_cc_python_prefactor(h, t_list, simplify_flag=True)
        expected_result = '1/(factorial(2)) * '
        assert function_output == expected_result

    def test_build_full_cc_python_prefactor_if_case_2(self):
        """h.m > 1"""
        h = fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=1, n_o=0, m_t=[1], n_t=[0])
        t_list = (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1),)
        function_output = cfcc._build_full_cc_python_prefactor(h, t_list, simplify_flag=True)
        expected_result = '1/(factorial(2)) * '
        assert function_output == expected_result

    def test_multiple_perms_logic(self):
        """x"""
        term = [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1)
            )
        ]
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

    def test_multiple_perms_logic_t_unique(self):
        """x"""
        term = [
            fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0, 0], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]),
            (
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
            )
        ]
        function_output = cfcc._multiple_perms_logic(term)
        expected_result = ([(0, 1), (1, 0)], {fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0): 2})
        assert function_output == expected_result

    def test_multiple_perms_logic_no_perms(self):
        """x"""
        term = [
            fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
            fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]),
            (
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
            )
        ]
        function_output = cfcc._multiple_perms_logic(term)
        expected_result = (None, {fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0): 1})
        assert function_output == expected_result

    def test_write_cc_einsum_python_from_list(self):
        """x"""
        rank = 1
        truncations = [1, 1, 1, 1]
        t_term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=0, m_t=[0], n_t=[1]),
                zero_h_op_nt,
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                )
            ]
        ]
        function_output = cfcc._write_cc_einsum_python_from_list(truncations, t_term_list, trunc_obj_name='truncation')
        expected_result = [
            'if truncation.singles:',
            "    R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])",
            '',
            'if truncation.at_least_linear:'
        ]
        assert function_output == expected_result

    def test_write_cc_einsum_python_from_list_zero_dc_case(self):
        """Case where t_list has length 1 and is populated by a 0,0,0,0 disconnected named tuple"""
        rank = 1
        truncations = [1, 1, 1, 1]
        t_term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=1, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),
                )
            ]
        ]
        function_output = cfcc._write_cc_einsum_python_from_list(truncations, t_term_list, trunc_obj_name='truncation')
        expected_result = [
            'R += h_args[(1, 0)]',
            '',
            'if truncation.at_least_linear:'
        ]
        assert function_output == expected_result

    def test_write_cc_einsum_python_from_list_0_indices(self):
        """x"""
        rank = 0
        truncations = [1, 1, 1, 1]
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
        function_output = cfcc._write_cc_einsum_python_from_list(truncations, t_term_list, trunc_obj_name='truncation')
        expected_result = [
            'R += h_args[(0, 0)]',
            '',
            'if truncation.at_least_linear:',
            '    if truncation.singles:',
            "        R += np.einsum('aci, cbi -> ab', h_args[(0, 1)], t_args[(1, 0)])"
        ]
        assert function_output == expected_result

    def test_write_cc_einsum_python_from_list_single_unique_key(self):
        """x"""
        rank = 2
        truncations = [2, 2, 2, 2]
        t_term_list = vars.write_cc_einsum_python_from_list_single_unique_key.t_term_list
        function_output = cfcc._write_cc_einsum_python_from_list(truncations, t_term_list, trunc_obj_name='truncation')
        assert function_output == vars.write_cc_einsum_python_from_list_single_unique_key.output

    def test_generate_full_cc_einsums(self):
        omega_term = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        truncations = [1, 1, 1, 1]
        function_output = cfcc._generate_full_cc_einsums(
            omega_term,
            truncations,
            only_ground_state=False,
            opt_einsum=False
        )
        expected_result = [
            [
                'R += h_args[(1, 0)]',
                '', 'if truncation.at_least_linear:'
            ],
            [
                'pass  # no valid terms here'
            ],
            [
                'if truncation.singles:',
                "    R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])",
                '',
                'if truncation.at_least_linear:'
            ]
        ]
        assert function_output == expected_result

    def test_generate_full_cc_compute_function(self):
        omega_term = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        truncations = [1, 1, 1, 1]
        function_output = cfcc._generate_full_cc_compute_function(
            omega_term,
            truncations,
            only_ground_state=False,
            opt_einsum=False
        )
        func_name = "generate_full_cc_compute_function_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_generate_full_cc_compute_function_opt_einsum(self):
        omega_term = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        truncations = [1, 1, 1, 1]
        function_output = cfcc._generate_full_cc_compute_function(
            omega_term,
            truncations,
            only_ground_state=False,
            opt_einsum=True
        )
        func_name = "generate_full_cc_compute_function_opt_einsum_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_write_master_full_cc_compute_function(self):
        omega_term = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        function_output = cfcc._write_master_full_cc_compute_function(omega_term, opt_einsum=False)
        func_name = "write_master_full_cc_compute_function_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_write_master_full_cc_compute_function_opt_einsum(self):
        omega_term = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        function_output = cfcc._write_master_full_cc_compute_function(omega_term, opt_einsum=True)
        func_name = "write_master_full_cc_compute_function_opt_einsum_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_wrap_full_cc_generation(self):
        s1, s2 = 75, 28
        spacing_line = "# " + "-"*s1 + " #\n"

        def named_line(name, width):
            return "# " + "-"*width + f" {name} " + "-"*width + " #"

        def spaced_named_line(name, width):
            return spacing_line + named_line(name, width) + '\n' + spacing_line

        truncations = [1, 1, 1, 1]
        master_omega = fcc.omega_namedtuple(
            maximum_rank=1,
            operator_list=[
                fcc.general_operator_namedtuple(name='', rank=0, m=0, n=0),
                fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1),
                fcc.general_operator_namedtuple(name='d', rank=1, m=1, n=0)
            ]
        )
        function_output = cfcc._wrap_full_cc_generation(
            truncations,
            master_omega,
            s2,
            named_line,
            spaced_named_line,
            only_ground_state=False,
            opt_einsum=False
        )
        func_name = "wrap_full_cc_generation_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_generate_full_cc_python_file_contents(self):
        truncations = [1, 1, 1, 1]
        function_output = cfcc._generate_full_cc_python_file_contents(truncations, only_ground_state=False)
        func_name = "generate_full_cc_python_file_contents_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_generate_full_cc_python(self):
        """run main func for coverage purposes"""
        cfcc.generate_full_cc_python([1, 1, 1, 1], only_ground_state=False, path="./full_cc_equations.py")
