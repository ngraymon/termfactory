# system imports
from os.path import abspath, dirname, join
deep_dir = dirname(abspath(__file__))
root_dir = join(deep_dir, 'files')
classtest = 'test_code_residual_equations'

# local imports
from . import context
from typing import ValuesView
import code_residual_equations as cre
from . import test_vars as vars

# global vars
h_0_zeros = cre.general_operator_namedtuple(name='h_0', rank=0, m=0, n=0)


class Test_generate_hamiltonian_operator:

    def test_basic(self):
        function_output = cre.generate_hamiltonian_operator(maximum_h_rank=2)
        expected_result = cre.hamiltonian_namedtuple(
            maximum_rank=2,
            operator_list=[
                h_0_zeros,
                cre.general_operator_namedtuple(name='h_1', rank=1, m=0, n=1),
                cre.general_operator_namedtuple(name='h_2', rank=2, m=0, n=2),
                cre.general_operator_namedtuple(name='h^1', rank=1, m=1, n=0),
                cre.general_operator_namedtuple(name='h^1_1', rank=2, m=1, n=1),
                cre.general_operator_namedtuple(name='h^2', rank=2, m=2, n=0)
            ]
        )
        assert function_output == expected_result


class Test_extract_numerator_denominator_from_string:

    def test_basic(self):
        s = '(3/3!)'
        function_output = cre.extract_numerator_denominator_from_string(s)
        expected_result = [3, 3]
        assert function_output == expected_result

    def test_product(self):
        s = '(1/2!)*(1/2)'
        function_output = cre.extract_numerator_denominator_from_string(s)
        expected_result = [1, 2]
        assert function_output == expected_result


class Test_simplified_prefactor:

    def test_basic(self):
        pre = '(1/1!)*(1/2)'
        function_output = cre.simplified_prefactor(pre)
        expected_result = '(1/2)'
        assert function_output == expected_result

    def test_edge_1(self):
        pre = '(1/2!)*(1/2)'
        function_output = cre.simplified_prefactor(pre)
        expected_result = '(1/(2*2!))'
        assert function_output == expected_result

    def test_edge_2(self):
        pre = '(1/1!)'
        function_output = cre.simplified_prefactor(pre)
        expected_result = ''
        assert function_output == expected_result


class Test_construct_prefactor:

    def test_special_case_1(self):
        h = h_0_zeros
        p = 0
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = ''
        assert function_output == expected_result

    def test_special_case_2(self):
        h = cre.general_operator_namedtuple(name='h^1', rank=1, m=1, n=0)
        p = 0
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = ''
        assert function_output == expected_result

    def test_special_case_3(self):
        h = cre.general_operator_namedtuple(name='h^2', rank=2, m=2, n=0)
        p = 2
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = '(1/2)'
        assert function_output == expected_result

    def test_case_1(self):
        h = cre.general_operator_namedtuple(name='h_1', rank=1, m=0, n=1)
        p = 0
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = '(1/1!)'
        assert function_output == expected_result

    def test_case_2(self):
        h = cre.general_operator_namedtuple(name='h_2', rank=2, m=0, n=2)
        p = 2
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = '(6/4!)'
        assert function_output == expected_result

    def test_case_3(self):
        h = cre.general_operator_namedtuple(name='h_2', rank=2, m=0, n=2)
        p = 0
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = '(1/2!)'
        assert function_output == expected_result

    def test_add_term(self):
        h = cre.general_operator_namedtuple(name='h^2', rank=2, m=2, n=0)
        p = 3
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = '(1/1!)*(1/2)'
        assert function_output == expected_result

    def test_simp_flag(self):
        h = cre.general_operator_namedtuple(name='h_2', rank=2, m=0, n=2)
        p = 4
        function_output = cre.construct_prefactor(h, p, simplify_flag=True)
        expected_result = '(15/6!)'
        assert function_output == expected_result


class Test_construct_upper_w_label:

    def test_basic(self):
        h = cre.general_operator_namedtuple(name='h_1', rank=1, m=0, n=1)
        p = 3
        function_output = cre.construct_upper_w_label(h, p)
        expected_result = '^{i_1,i_2,i_3,k_1}'
        assert function_output == expected_result

    def test_edge(self):
        """if (h.m == p and h.n == 0) or h.m > p:"""
        h = h_0_zeros
        p = 0
        function_output = cre.construct_upper_w_label(h, p)
        expected_result = ''
        assert function_output == expected_result


class Test_construct_upper_h_label:

    def test_basic(self):
        h = cre.general_operator_namedtuple(name='h^1_1', rank=2, m=1, n=1)
        p = 3
        function_output = cre.construct_upper_h_label(h, p)
        expected_result = '^{i_1}'
        assert function_output == expected_result

    def test_edge(self):
        """if h.m == 0 or h.m > p:"""
        h = h_0_zeros
        p = 0
        function_output = cre.construct_upper_h_label(h, p)
        expected_result = ''
        assert function_output == expected_result


class Test_construct_lower_h_label:

    def test_basic(self):
        h = cre.general_operator_namedtuple(name='h^1_1', rank=2, m=1, n=1)
        p = 3
        function_output = cre.construct_lower_h_label(h, p)
        expected_result = '_{k_1}'
        assert function_output == expected_result

    def test_edge_case_1(self):
        """if h.m == 0 and h.n == 0:"""
        h = h_0_zeros
        p = 0
        function_output = cre.construct_lower_h_label(h, p)
        expected_result = '_0'
        assert function_output == expected_result

    def test_edge_case_2(self):
        """if h.n == 0 or h.m > p: """
        h = cre.general_operator_namedtuple(name='h^1', rank=1, m=1, n=0)
        p = 0
        function_output = cre.construct_lower_h_label(h, p)
        expected_result = ''
        assert function_output == expected_result


class Test_generate_p_term:

    def test_prefactor_is_one(self):
        str_fac = ''
        function_output = cre.generate_p_term(str_fac)
        expected_result = '1.0'
        assert function_output == expected_result

    def test_all_cases_false(self):
        str_fac = '(1/2!)'
        function_output = cre.generate_p_term(str_fac)
        expected_result = '(1/2)'
        assert function_output == expected_result

    def test_div_case(self):
        """if "/(2*" in str_fac:"""
        str_fac = '(1/(2*2!))'
        function_output = cre.generate_p_term(str_fac)
        expected_result = '(1/(2*2))'
        assert function_output == expected_result


class Test_generate_h_term:

    def test_zero_case(self):
        """if "0" in str_h:"""
        str_h = 'h_0'
        function_output = cre.generate_h_term(str_h)
        expected_result = cre.h_namedtuple(max_i=0, max_k=0)
        assert function_output == expected_result

    def test_normal(self):
        """normal operation"""
        str_h = 'h^{i_1,i_2}'
        function_output = cre.generate_h_term(str_h)
        expected_result = cre.h_namedtuple(max_i=2, max_k=0)
        assert function_output == expected_result


class Test_generate_w_term:

    def test_empty_case(self):
        """if str_w == "": """
        str_w = ""
        function_output = cre.generate_w_term(str_w)
        expected_result = cre.w_namedtuple(max_i=0, max_k=0, order=0)
        assert function_output == expected_result

    def test_normal(self):
        """normal operation"""
        str_w = "w^{i_1,i_2,i_3,k_1,k_2}"
        function_output = cre.generate_w_term(str_w)
        expected_result = cre.w_namedtuple(max_i=3, max_k=2, order=5)
        assert function_output == expected_result


class Test_generate_residual_string_list:

    def test_basic(self):
        hamiltonian = [
            h_0_zeros,
            cre.general_operator_namedtuple(name='h_1', rank=1, m=0, n=1),
            cre.general_operator_namedtuple(name='h_2', rank=2, m=0, n=2),
            cre.general_operator_namedtuple(name='h^1', rank=1, m=1, n=0),
            cre.general_operator_namedtuple(name='h^1_1', rank=2, m=1, n=1),
            cre.general_operator_namedtuple(name='h^2', rank=2, m=2, n=0)
        ]
        order = 0
        function_output = cre.generate_residual_string_list(hamiltonian, order)
        expected_result = (
            [
                'h_0',
                'h_{k_1} * w^{k_1}',
                '(1/2!) * h_{k_1,k_2} * w^{k_1,k_2}'
            ],
            [
                cre.residual_term(
                    prefactor='1.0',
                    h=cre.h_namedtuple(max_i=0, max_k=0),
                    w=cre.w_namedtuple(max_i=0, max_k=0, order=0)
                ),
                cre.residual_term(
                    prefactor='1.0',
                    h=cre.h_namedtuple(max_i=0, max_k=1),
                    w=cre.w_namedtuple(max_i=0, max_k=1, order=1)
                ),
                cre.residual_term(
                    prefactor='(1/2)',
                    h=cre.h_namedtuple(max_i=0, max_k=2),
                    w=cre.w_namedtuple(max_i=0, max_k=2, order=2)
                )
            ]
        )
        assert function_output == expected_result


class Test_generate_residual_data:

    def test_basic(self):
        H = [
            h_0_zeros,
            cre.general_operator_namedtuple(name='h_1', rank=1, m=0, n=1),
            cre.general_operator_namedtuple(name='h^1', rank=1, m=1, n=0),
        ]
        max_order = 1
        function_output = cre.generate_residual_data(H, max_order)
        assert function_output == vars.generate_residual_data.expected_result


class Test_generate_einsum_h_indices:

    def test_basic(self):
        term = cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=0, max_k=1),
            w=cre.w_namedtuple(max_i=0, max_k=1, order=1)
        )
        function_output = cre._generate_einsum_h_indices(term)
        expected_result = 'acm'
        assert function_output == expected_result


class Test_generate_einsum_w_indices:

    def test_basic(self):
        term = cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=0, max_k=1),
            w=cre.w_namedtuple(max_i=0, max_k=1, order=1)
        )
        function_output = cre._generate_einsum_w_indices(term)
        expected_result = 'cbm'
        assert function_output == expected_result


class Test_generate_einsum_ouput_indices:

    def test_basic(self):
        term = cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=0, max_k=1),
            w=cre.w_namedtuple(max_i=0, max_k=1, order=1)
        )
        function_output = cre._generate_einsum_ouput_indices(term)
        expected_result = 'ab'
        assert function_output == expected_result


class Test_residual_terms_einsum:

    def test_basic(self):
        term = cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=0, max_k=1),
            w=cre.w_namedtuple(max_i=0, max_k=1, order=1)
        )
        function_output = cre._residual_terms_einsum(term, suppress_1_prefactor=True)
        expected_result = "R += 1.0 * np.einsum('acm,cbm->ab', h_abI, w_i)\n"
        assert function_output == expected_result

    def test_w_zero(self):
        """if term.w.order == 0:"""
        term = cre.residual_term(
            prefactor='(1/2)',
            h=cre.h_namedtuple(max_i=2, max_k=0),
            w=cre.w_namedtuple(max_i=0, max_k=0, order=0)
        )
        function_output = cre._residual_terms_einsum(term, suppress_1_prefactor=True)
        expected_result = 'R += (1/2) * h_abij\n'
        assert function_output == expected_result


class Test_same_w_order_term_list:

    def test_basic(self):
        current_term = cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=0, max_k=1),
            w=cre.w_namedtuple(max_i=1, max_k=1, order=2)
        )
        term_list = [
            cre.residual_term(
                prefactor='1.0',
                h=cre.h_namedtuple(max_i=0, max_k=0),
                w=cre.w_namedtuple(max_i=1, max_k=0, order=1)
            ),
            cre.residual_term(
                prefactor='1.0',
                h=cre.h_namedtuple(max_i=0, max_k=1),
                w=cre.w_namedtuple(max_i=1, max_k=1, order=2)
            ),
            cre.residual_term(
                prefactor='1.0',
                h=cre.h_namedtuple(max_i=1, max_k=0),
                w=cre.w_namedtuple(max_i=0, max_k=0, order=0)
            )
        ]
        function_output = cre._same_w_order_term_list(current_term, term_list)
        expected_result = [
            cre.residual_term(
                prefactor='1.0',
                h=cre.h_namedtuple(max_i=0, max_k=1),
                w=cre.w_namedtuple(max_i=1, max_k=1, order=2)
            )
        ]
        assert function_output == expected_result


class Test_write_residual_function_string:

    def test_basic(self):
        residual_terms_list = [
            cre.residual_term(
                prefactor='1.0',
                h=cre.h_namedtuple(max_i=0, max_k=0),
                w=cre.w_namedtuple(max_i=0, max_k=0, order=0)
            ),
            cre.residual_term(
                prefactor='1.0',
                h=cre.h_namedtuple(max_i=0, max_k=1),
                w=cre.w_namedtuple(max_i=0, max_k=1, order=1)
            )
        ]
        order = 0
        function_output = cre.write_residual_function_string(residual_terms_list, order)
        func_name = "write_residual_function_string_basic_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_high_order_h_and_w(self):
        residual_terms_list = vars.write_residual_function_string_high_order_h_and_w.res_list
        order = 2
        function_output = cre.write_residual_function_string(residual_terms_list, order)
        func_name = "write_residual_function_string_high_order_h_and_w_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result


class Test_generate_python_code_for_residual_functions:

    def test_basic(self):
        term_lists = vars.generate_python_code_for_residual_functions.t_list
        max_order = 1
        function_output = cre.generate_python_code_for_residual_functions(term_lists, max_order)
        func_name = "generate_python_code_for_residual_functions_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result


class Test_run_main_generate_files_eqs:

    def test_run_main(self):
        max_residual_order = 2
        maximum_h_rank = 2
        cre.generate_residual_equations_file(max_residual_order, maximum_h_rank, path="./residual_equations.py")
