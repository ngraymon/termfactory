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


class Test_generate_w_operator_prefactor:

    def test_basic(self):
        """basic test"""

        # run function
        function_output = cw._generate_w_operator_prefactor((1, 1, 1))
        expected_result = '1/factorial(3)'

        assert function_output == expected_result

    def test_max_1(self):
        """single tuple containing 1"""

        # run function
        function_output = cw._generate_w_operator_prefactor((1,))
        expected_result = ''

        assert function_output == expected_result

    def test_single(self):
        """tuple of len 1 but not containing 1"""

        # run function
        function_output = cw._generate_w_operator_prefactor((3,))
        expected_result = f"1/factorial({3})"

        assert function_output == expected_result

    def test_single_else(self):
        """else case"""

        # run function
        function_output = cw._generate_w_operator_prefactor((2, 1))
        expected_result = '1/(factorial(2) * factorial(2))'

        assert function_output == expected_result


class Test_generate_surface_index:

    def test_basic(self):
        """basic test"""

        # input data
        partition = (1, 1, 1)

        # run function
        function_output = cw._generate_surface_index(partition)
        expected_result = ['ac', 'cd', 'db']

        assert function_output == expected_result


class Test_generate_mode_index:

    def test_basic(self):
        """basic test"""

        # input data
        partition = (1, 1, 1)
        order = 3

        # run function
        function_output = cw._generate_mode_index(partition, order)
        expected_result = [['i', 'j', 'k']]

        assert function_output == expected_result


class Test_w_einsum_list:

    def test_basic(self):
        """basic test"""

        # input data
        partition = (1, 1, 1)
        order = 3

        # run function
        function_output = cw._w_einsum_list(partition, order)
        expected_result = ["np.einsum('aci, cdj, dbk->abijk', t_i, t_i, t_i)"]

        assert function_output == expected_result

    def test_opt_einsum(self):
        """opt_einsum test"""

        # input data
        partition = (1, 1, 1)
        order = 3

        # run function
        function_output = cw._optimized_w_einsum_list(partition, order, iterator_name='optimized_einsum')
        expected_result = ['next(optimized_einsum)(t_i, t_i, t_i)']

        assert function_output == expected_result


class Test_construct_vemx_contributions_definition:

    def test_basic(self):
        """basic test, not opt_einsum"""

        # input data
        return_string = ''
        order = 3

        # run function
        function_output = cw._construct_vemx_contributions_definition(
            return_string,
            order,
            opt_einsum=False,
            iterator_name='optimized_einsum'
        )

        # open file
        func_name = "construct_vemx_contributions_definition_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result

    def test_einsum_enabled(self):
        """test opt_einsum"""

        # input data
        return_string = ''
        order = 3

        # run function
        function_output = cw._construct_vemx_contributions_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )

        # open file
        func_name = "construct_vemx_contributions_definition_einsum_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_generate_vemx_contributions:

    def test_basic(self):
        """basic test"""

        # input data
        order = 3

        # run function
        function_output = cw._generate_vemx_contributions(order, opt_einsum=False)

        # open file
        func_name = "generate_vemx_contributions_basic_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result

    def test_small_order(self):
        """test on small order"""

        # input data
        order = 1

        # run function
        function_output = cw._generate_vemx_contributions(order, opt_einsum=False)

        # open file
        func_name = "generate_vemx_contributions_small_order_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_construct_vecc_contributions_definition:

    def test_basic(self):
        """basic test"""

        # input data
        return_string = ""
        order = 2

        # run function
        function_output = cw._construct_vecc_contributions_definition(
            return_string,
            order,
            opt_einsum=False,
            iterator_name='optimized_einsum'
        )

        # open file
        func_name = "construct_vecc_contributions_definition_basic_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result

    def test_einsum(self):
        """einsum test"""

        # input data
        return_string = ""
        order = 2

        # run function
        function_output = cw._construct_vecc_contributions_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )

        # open file
        func_name = "construct_vecc_contributions_definition_einsum_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_generate_vecc_contributions:

    def test_basic(self):
        """basic test"""

        # input data
        order = 2

        # run function
        function_output = cw._generate_vecc_contributions(order, opt_einsum=False)

        # open file
        func_name = "generate_vecc_contributions_basic_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result

    def test_high_order(self):
        """high order test"""

        # input data
        order = 5

        # run function
        function_output = cw._generate_vecc_contributions(order, opt_einsum=False)

        # open file
        func_name = "generate_vecc_contributions_basic_high_order_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_construct_w_function_definition:

    def test_basic(self):
        """basic test"""

        # input data
        return_string = ""
        order = 2

        # run function
        function_output = cw._construct_w_function_definition(
            return_string,
            order,
            opt_einsum=False,
            iterator_name='optimized_einsum'
        )

        # open file
        func_name = "construct_w_function_definition_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result

    def test_einsum(self):
        """einsum test"""

        # input data
        return_string = ""
        order = 2

        # run function
        function_output = cw._construct_w_function_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )

        # open file
        func_name = "construct_w_function_definition_einsum_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_write_w_function_strings:

    def test_write_w_function_strings_basic(self):
        """basic test"""

        # input data
        order = 2

        # run function
        function_output = cw._write_w_function_strings(order, opt_einsum=False)

        # open file
        func_name = "write_w_function_strings_basic_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result

    def test_write_w_function_strings_zero_case(self):
        """zero test"""

        # input data
        order = 0

        # check if exception raised
        with pytest.raises(Exception, match="We should not call `_write_w_function_strings` with order 0."):
            cw._write_w_function_strings(order, opt_einsum=False)

    def test_write_w_function_strings_1st_order(self):
        """1st order test"""

        # input data
        order = 1

        # run function
        function_output = cw._write_w_function_strings(order, opt_einsum=False)

        # open file
        func_name = "write_w_function_strings_1st_order_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result

    def test_write_w_function_strings_high_order(self):
        """high order test"""

        # input data
        order = 5

        # run function
        function_output = cw._write_w_function_strings(order, opt_einsum=False)

        # open file
        func_name = "write_w_function_strings_high_order_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_write_master_w_compute_function:

    def test_write_master_w_compute_function_basic(self):
        """basic test"""

        # input data
        max_order = 2

        # run function
        function_output = cw._write_master_w_compute_function(max_order, opt_einsum=False)

        # open file
        func_name = "write_master_w_compute_function_basic_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result

    def test_write_master_w_compute_function_einsum(self):
        """einsum test"""

        # input data
        max_order = 2

        # run function
        function_output = cw._write_master_w_compute_function(max_order, opt_einsum=True)

        # open file
        func_name = "write_master_w_compute_function_einsum_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_t_term_shape_string:

    def test_basic(self):
        """basic test"""

        # input data
        order = 2

        # run function
        function_output = cw._t_term_shape_string(order)
        expected_result = '(A, A, N, N)'

        assert function_output == expected_result


class Test_contracted_expressions:

    def test_basic(self):
        """basic test"""

        # input data
        partition_list = [(2, 1), (1, 1, 1)]
        order = 3

        # run function
        function_output = cw._contracted_expressions(partition_list, order)
        expected_result = [
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


class Test_write_optimized_vemx_paths_function:

    def test_basic(self):
        """basic test"""

        # input data
        max_order = 2

        # run function
        function_output = cw._write_optimized_vemx_paths_function(max_order)

        # open file
        func_name = "write_optimized_vemx_paths_function_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_write_optimized_vecc_paths_function:

    def test_basic(self):
        """basic test"""

        # input data
        max_order = 2

        # run function
        function_output = cw._write_optimized_vecc_paths_function(max_order)

        # open file
        func_name = "write_optimized_vecc_paths_function_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result

    def test_high_order(self):
        """high order test"""

        # input data
        max_order = 5

        # run function
        function_output = cw._write_optimized_vecc_paths_function(max_order)

        # open file
        func_name = "write_optimized_vecc_paths_function_high_order_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_run_main_w_eqn_code:

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
