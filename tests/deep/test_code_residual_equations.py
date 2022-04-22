# system imports
from os.path import abspath, dirname, join
deep_dir = dirname(abspath(__file__))
root_dir = join(deep_dir, 'files')
classtest = 'test_code_residual_equations'

# local imports
from . import context
from typing import ValuesView
import code_residual_equations as cre
from . import large_test_data

# global vars
h_0_zeros = cre.general_operator_namedtuple(name='h_0', rank=0, m=0, n=0)


class Test_generate_hamiltonian_operator:

    def test_basic(self):
        """basic test"""

        # run function
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
        """basic test"""

        # input data
        s = '(3/3!)'

        # run function
        function_output = cre.extract_numerator_denominator_from_string(s)
        expected_result = [3, 3]

        assert function_output == expected_result

    def test_product(self):
        """test with product"""

        # input data
        s = '(1/2!)*(1/2)'

        # run function
        function_output = cre.extract_numerator_denominator_from_string(s)
        expected_result = [1, 2]

        assert function_output == expected_result


class Test_simplified_prefactor:

    def test_basic(self):
        """basic test"""

        # input data
        pre = '(1/1!)*(1/2)'

        # run function
        function_output = cre.simplified_prefactor(pre)
        expected_result = '(1/2)'

        assert function_output == expected_result

    def test_edge_1(self):
        """x"""

        # input data
        pre = '(1/2!)*(1/2)'

        # run function
        function_output = cre.simplified_prefactor(pre)
        expected_result = '(1/(2*2!))'

        assert function_output == expected_result

    def test_edge_2(self):
        """x"""

        # input data
        pre = '(1/1!)'

        # run function
        function_output = cre.simplified_prefactor(pre)
        expected_result = ''

        assert function_output == expected_result


class Test_construct_prefactor:

    def test_special_case_1(self):
        """x"""

        # input data
        h = h_0_zeros
        p = 0

        # run function
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = ''

        assert function_output == expected_result

    def test_special_case_2(self):
        """x"""

        # input data
        h = cre.general_operator_namedtuple(name='h^1', rank=1, m=1, n=0)
        p = 0

        # run function
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = ''

        assert function_output == expected_result

    def test_special_case_3(self):
        """x"""

        # input data
        h = cre.general_operator_namedtuple(name='h^2', rank=2, m=2, n=0)
        p = 2

        # run function
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = '(1/2)'

        assert function_output == expected_result

    def test_case_1(self):
        """x"""

        # input data
        h = cre.general_operator_namedtuple(name='h_1', rank=1, m=0, n=1)
        p = 0

        # run function
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = '(1/1!)'

        assert function_output == expected_result

    def test_case_2(self):
        """x"""

        # input data
        h = cre.general_operator_namedtuple(name='h_2', rank=2, m=0, n=2)
        p = 2

        # run function
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = '(6/4!)'

        assert function_output == expected_result

    def test_case_3(self):
        """x"""

        # input data
        h = cre.general_operator_namedtuple(name='h_2', rank=2, m=0, n=2)
        p = 0

        # run function
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = '(1/2!)'

        assert function_output == expected_result

    def test_add_term(self):
        """x"""

        # input data
        h = cre.general_operator_namedtuple(name='h^2', rank=2, m=2, n=0)
        p = 3

        # run function
        function_output = cre.construct_prefactor(h, p, simplify_flag=False)
        expected_result = '(1/1!)*(1/2)'

        assert function_output == expected_result

    def test_simp_flag(self):
        """x"""

        # input data
        h = cre.general_operator_namedtuple(name='h_2', rank=2, m=0, n=2)
        p = 4

        # run function
        function_output = cre.construct_prefactor(h, p, simplify_flag=True)
        expected_result = '(15/6!)'

        assert function_output == expected_result


class Test_construct_upper_w_label:

    def test_basic(self):
        """basic test"""

        # input data
        h = cre.general_operator_namedtuple(name='h_1', rank=1, m=0, n=1)
        p = 3

        # run function
        function_output = cre.construct_upper_w_label(h, p)
        expected_result = '^{i_1,i_2,i_3,k_1}'

        assert function_output == expected_result

    def test_edge(self):
        """if (h.m == p and h.n == 0) or h.m > p:"""

        # input data
        h = h_0_zeros
        p = 0

        # run function
        function_output = cre.construct_upper_w_label(h, p)
        expected_result = ''

        assert function_output == expected_result


class Test_construct_upper_h_label:

    def test_basic(self):
        """basic test"""

        # input data
        h = cre.general_operator_namedtuple(name='h^1_1', rank=2, m=1, n=1)
        p = 3

        # run function
        function_output = cre.construct_upper_h_label(h, p)
        expected_result = '^{i_1}'

        assert function_output == expected_result

    def test_edge(self):
        """if h.m == 0 or h.m > p:"""

        # input data
        h = h_0_zeros
        p = 0

        # run function
        function_output = cre.construct_upper_h_label(h, p)
        expected_result = ''

        assert function_output == expected_result


class Test_construct_lower_h_label:

    def test_basic(self):
        """basic test"""

        # input data
        h = cre.general_operator_namedtuple(name='h^1_1', rank=2, m=1, n=1)
        p = 3

        # run function
        function_output = cre.construct_lower_h_label(h, p)
        expected_result = '_{k_1}'

        assert function_output == expected_result

    def test_edge_case_1(self):
        """if h.m == 0 and h.n == 0:"""

        # input data
        h = h_0_zeros
        p = 0

        # run function
        function_output = cre.construct_lower_h_label(h, p)
        expected_result = '_0'

        assert function_output == expected_result

    def test_edge_case_2(self):
        """if h.n == 0 or h.m > p: """

        # input data
        h = cre.general_operator_namedtuple(name='h^1', rank=1, m=1, n=0)
        p = 0

        # run function
        function_output = cre.construct_lower_h_label(h, p)
        expected_result = ''

        assert function_output == expected_result


class Test_generate_p_term:

    def test_prefactor_is_one(self):
        """prefactor==1.0"""

        # input data
        str_fac = ''

        # run function
        function_output = cre.generate_p_term(str_fac)
        expected_result = '1.0'

        assert function_output == expected_result

    def test_all_cases_false(self):
        """all elif==false"""

        # input data
        str_fac = '(1/2!)'

        # run function
        function_output = cre.generate_p_term(str_fac)
        expected_result = '(1/2)'

        assert function_output == expected_result

    def test_div_case(self):
        """if "/(2*" in str_fac:"""

        # input data
        str_fac = '(1/(2*2!))'

        # run function
        function_output = cre.generate_p_term(str_fac)
        expected_result = '(1/(2*2))'

        assert function_output == expected_result


class Test_generate_h_term:

    def test_zero_case(self):
        """if "0" in str_h:"""

        # input data
        str_h = 'h_0'

        # run function
        function_output = cre.generate_h_term(str_h)
        expected_result = cre.h_namedtuple(max_i=0, max_k=0)

        assert function_output == expected_result

    def test_normal(self):
        """normal operation"""

        # input data
        str_h = 'h^{i_1,i_2}'

        # run function
        function_output = cre.generate_h_term(str_h)
        expected_result = cre.h_namedtuple(max_i=2, max_k=0)

        assert function_output == expected_result


class Test_generate_w_term:

    def test_empty_case(self):
        """if str_w == "": """

        # input data
        str_w = ""

        # run function
        function_output = cre.generate_w_term(str_w)
        expected_result = cre.w_namedtuple(max_i=0, max_k=0, order=0)

        assert function_output == expected_result

    def test_normal(self):
        """normal operation"""

        # input data
        str_w = "w^{i_1,i_2,i_3,k_1,k_2}"

        # run function
        function_output = cre.generate_w_term(str_w)
        expected_result = cre.w_namedtuple(max_i=3, max_k=2, order=5)

        assert function_output == expected_result


class Test_generate_residual_string_list:

    def test_basic(self):
        """basic test"""

        # input data
        hamiltonian = [
            h_0_zeros,
            cre.general_operator_namedtuple(name='h_1', rank=1, m=0, n=1),
            cre.general_operator_namedtuple(name='h_2', rank=2, m=0, n=2),
            cre.general_operator_namedtuple(name='h^1', rank=1, m=1, n=0),
            cre.general_operator_namedtuple(name='h^1_1', rank=2, m=1, n=1),
            cre.general_operator_namedtuple(name='h^2', rank=2, m=2, n=0)
        ]
        order = 0

        # run function
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
        """basic test"""

        # input data
        H = [
            h_0_zeros,
            cre.general_operator_namedtuple(name='h_1', rank=1, m=0, n=1),
            cre.general_operator_namedtuple(name='h^1', rank=1, m=1, n=0),
        ]
        max_order = 1

        # run function
        function_output = cre.generate_residual_data(H, max_order)

        assert function_output == large_test_data.generate_residual_data.expected_result


class Test_generate_einsum_h_indices:

    def test_basic(self):
        """basic test"""

        # input data
        term = cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=0, max_k=1),
            w=cre.w_namedtuple(max_i=0, max_k=1, order=1)
        )

        # run function
        function_output = cre._generate_einsum_h_indices(term)
        expected_result = 'acm'

        assert function_output == expected_result


class Test_generate_einsum_w_indices:

    def test_basic(self):
        """basic test"""

        # input data
        term = cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=0, max_k=1),
            w=cre.w_namedtuple(max_i=0, max_k=1, order=1)
        )

        # run function
        function_output = cre._generate_einsum_w_indices(term)
        expected_result = 'cbm'

        assert function_output == expected_result


class Test_generate_einsum_ouput_indices:

    def test_basic(self):
        """basic test"""

        # input data
        term = cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=0, max_k=1),
            w=cre.w_namedtuple(max_i=0, max_k=1, order=1)
        )

        # run function
        function_output = cre._generate_einsum_ouput_indices(term)
        expected_result = 'ab'

        assert function_output == expected_result


class Test_residual_terms_einsum:

    def test_basic(self):
        """basic test"""

        # input data
        term = cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=0, max_k=1),
            w=cre.w_namedtuple(max_i=0, max_k=1, order=1)
        )

        # run function
        function_output = cre._residual_terms_einsum(term, suppress_1_prefactor=True)
        expected_result = "R += 1.0 * np.einsum('acm,cbm->ab', h_abI, w_i)\n"

        assert function_output == expected_result

    def test_w_zero(self):
        """if term.w.order == 0:"""

        # input data
        term = cre.residual_term(
            prefactor='(1/2)',
            h=cre.h_namedtuple(max_i=2, max_k=0),
            w=cre.w_namedtuple(max_i=0, max_k=0, order=0)
        )

        # run function
        function_output = cre._residual_terms_einsum(term, suppress_1_prefactor=True)
        expected_result = 'R += (1/2) * h_abij\n'

        assert function_output == expected_result


class Test_same_w_order_term_list:

    def test_basic(self):
        """basic test"""

        # input data
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

        # run function
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
        """basic test"""

        # input data
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

        # run function
        function_output = cre.write_residual_function_string(residual_terms_list, order)

        # open file
        func_name = "write_residual_function_string_basic_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result

    def test_high_order_h_and_w(self):
        """high order h & w"""

        # input data
        residual_terms_list = large_test_data.write_residual_function_string_high_order_h_and_w.res_list
        order = 2

        # run function
        function_output = cre.write_residual_function_string(residual_terms_list, order)

        # open file
        func_name = "write_residual_function_string_high_order_h_and_w_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_generate_python_code_for_residual_functions:

    def test_basic(self):
        """basic test"""

        # input data
        term_lists = large_test_data.generate_python_code_for_residual_functions.t_list
        max_order = 1

        # run function
        function_output = cre.generate_python_code_for_residual_functions(term_lists, max_order)

        # open file
        func_name = "generate_python_code_for_residual_functions_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_run_main_generate_files_eqs:

    def test_run_main(self, tmpdir):
        """runs main function and compares it to a reference file"""

        output_path = join(tmpdir, "code_residual_equations.py")

        max_residual_order = 2
        maximum_h_rank = 2
        cre.generate_residual_equations_file(max_residual_order, maximum_h_rank, path=output_path)

        with open(output_path, 'r') as fp:
            file_data = fp.read()

        func_name = "code_output_residual_equations.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            reference_file_data = fp.read()

        assert file_data == reference_file_data, 'Fail'
