# system imports
# import re
import pytest
from os.path import abspath, dirname, join
deep_dir = dirname(abspath(__file__))
root_dir = join(deep_dir, 'files')
classtest = 'test_code_w_equations'

# local imports
from . import context
import code_w_equations as cw


class Test_gen_w_oper:

    def test_generate_w_operator_prefactor_basic(self):
        function_output = cw._generate_w_operator_prefactor((1, 1, 1))
        expected_result = '1/factorial(3)'
        assert function_output == expected_result

    def test_generate_w_operator_prefactor_max_1(self):
        function_output = cw._generate_w_operator_prefactor((1,))
        expected_result = ''
        assert function_output == expected_result

    def test_generate_w_operator_prefactor_single(self):
        """tuple of len 1 but not containing 1"""
        function_output = cw._generate_w_operator_prefactor((3,))
        expected_result = f"1/factorial({3})"
        assert function_output == expected_result

    def test_generate_w_operator_prefactor_single_else(self):
        """else case"""
        function_output = cw._generate_w_operator_prefactor((2, 1))
        expected_result = '1/(factorial(2) * factorial(2))'
        assert function_output == expected_result


class Test_begin_code_gen:

    def test_generate_surface_index(self):
        partition = (1, 1, 1)
        function_output = cw._generate_surface_index(partition)
        expected_result = ['ac', 'cd', 'db']
        assert function_output == expected_result

    def test_generate_mode_index(self):
        partition = (1, 1, 1)
        order = 3
        function_output = cw._generate_mode_index(partition, order)
        expected_result = [['i', 'j', 'k']]
        assert function_output == expected_result

    def test_w_einsum_list(self):
        partition = (1, 1, 1)
        order = 3
        function_output = cw._w_einsum_list(partition, order)
        expected_result = ["np.einsum('aci, cdj, dbk->abijk', t_i, t_i, t_i)"]
        assert function_output == expected_result

    def test_optimized_w_einsum_list(self):
        partition = (1, 1, 1)
        order = 3
        function_output = cw._optimized_w_einsum_list(partition, order, iterator_name='optimized_einsum')
        expected_result = ['next(optimized_einsum)(t_i, t_i, t_i)']
        assert function_output == expected_result


class Test_contributions:

    def test_construct_vemx_contributions_definition(self):
        """basic test, not opt_einsum"""
        return_string = ''
        order = 3
        function_output = cw._construct_vemx_contributions_definition(
            return_string,
            order,
            opt_einsum=False,
            iterator_name='optimized_einsum'
        )
        func_name = "construct_vemx_contributions_definition_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_construct_vemx_contributions_definition_einsum(self):
        """test opt_einsum"""
        return_string = ''
        order = 3
        function_output = cw._construct_vemx_contributions_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )
        func_name = "construct_vemx_contributions_definition_einsum_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_generate_vemx_contributions_basic(self):
        """basic test"""
        order = 3
        function_output = cw._generate_vemx_contributions(order, opt_einsum=False)
        func_name = "generate_vemx_contributions_basic_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_generate_vemx_contributions_small_order(self):
        """test on small order"""
        order = 1
        function_output = cw._generate_vemx_contributions(order, opt_einsum=False)
        func_name = "generate_vemx_contributions_small_order_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_construct_vecc_contributions_definition_basic(self):
        """basic test"""
        return_string = ""
        order = 2
        function_output = cw._construct_vecc_contributions_definition(
            return_string,
            order,
            opt_einsum=False,
            iterator_name='optimized_einsum'
        )
        func_name = "construct_vecc_contributions_definition_basic_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_construct_vecc_contributions_definition_einsum(self):
        """einsum test"""
        return_string = ""
        order = 2
        function_output = cw._construct_vecc_contributions_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )
        func_name = "construct_vecc_contributions_definition_einsum_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_generate_vecc_contributions_basic(self):
        """basic test"""
        order = 2
        function_output = cw._generate_vecc_contributions(order, opt_einsum=False)
        func_name = "generate_vecc_contributions_basic_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_generate_vecc_contributions_basic_high_order(self):
        """high order test"""
        order = 5
        function_output = cw._generate_vecc_contributions(order, opt_einsum=False)
        func_name = "generate_vecc_contributions_basic_high_order_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result


class Test_def_w_func:

    def test_construct_w_function_definition(self):
        """basic test"""
        return_string = ""
        order = 2
        function_output = cw._construct_w_function_definition(
            return_string,
            order,
            opt_einsum=False,
            iterator_name='optimized_einsum'
        )
        func_name = "construct_w_function_definition_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_construct_w_function_definition_einsum(self):
        """einsum test"""
        return_string = ""
        order = 2
        function_output = cw._construct_w_function_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )
        func_name = "construct_w_function_definition_einsum_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_write_w_function_strings_basic(self):
        """basic test"""
        order = 2
        function_output = cw._write_w_function_strings(order, opt_einsum=False)
        func_name = "write_w_function_strings_basic_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_write_w_function_strings_zero_case(self):
        """zero test"""
        order = 0
        with pytest.raises(Exception, match="We should not call `_write_w_function_strings` with order 0."):
            cw._write_w_function_strings(order, opt_einsum=False)

    def test_write_w_function_strings_1st_order(self):
        """1st order test"""
        order = 1
        function_output = cw._write_w_function_strings(order, opt_einsum=False)
        func_name = "write_w_function_strings_1st_order_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_write_w_function_strings_high_order(self):
        """high order test"""
        order = 5
        function_output = cw._write_w_function_strings(order, opt_einsum=False)
        func_name = "write_w_function_strings_high_order_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_write_master_w_compute_function_basic(self):
        """basic test"""
        max_order = 2
        function_output = cw._write_master_w_compute_function(max_order, opt_einsum=False)
        func_name = "write_master_w_compute_function_basic_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_write_master_w_compute_function_einsum(self):
        """einsum test"""
        max_order = 2
        function_output = cw._write_master_w_compute_function(max_order, opt_einsum=True)
        func_name = "write_master_w_compute_function_einsum_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result


class Test_optimizers:

    def test_t_term_shape_string(self):
        order = 2
        function_output = cw._t_term_shape_string(order)
        expected_result = '(A, A, N, N)'
        assert function_output == expected_result

    def test_contracted_expressions(self):
        partition_list = [(2, 1), (1, 1, 1)]
        order = 3
        function_output = cw._contracted_expressions(partition_list, order)
        expected_result = [  # file flag
            [
                2,
                "oe.contract_expression('aci, cbjk->abijk', (A, A, N), (A, A, N, N)),\n",
                "oe.contract_expression('acij, cbk->abijk', (A, A, N, N), (A, A, N)),\n"
            ],
            [
                1,
                "oe.contract_expression('aci, cdj, dbk->abijk', (A, A, N), (A, A, N), (A, A, N)),\n"
            ]
        ]
        assert function_output == expected_result

    def test_write_optimized_vemx_paths_function(self):
        max_order = 2
        function_output = cw._write_optimized_vemx_paths_function(max_order)
        func_name = "write_optimized_vemx_paths_function_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_write_optimized_vecc_paths_function(self):
        max_order = 2
        function_output = cw._write_optimized_vecc_paths_function(max_order)
        func_name = "write_optimized_vecc_paths_function_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_write_optimized_vecc_paths_function_high_order(self):
        max_order = 5
        function_output = cw._write_optimized_vecc_paths_function(max_order)
        func_name = "write_optimized_vecc_paths_function_high_order_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result


class Test_main_w_eqn_code:

    def test_generate_w_operators_string(self):
        max_order = 2
        function_output = cw.generate_w_operators_string(max_order, s1=75, s2=28)
        func_name = "generate_w_operators_string_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_run_main_w_eqn_func(self):
        # TODO file compare assert
        cw.generate_w_operator_equations_file(2, path="./w_operator_equations.py")
