# system imports
# import pytest

# local imports
from . import context
import code_dt_equations as cdt


class Test_generate_disconnected_einsum_operands_list:

    def test_basic(self):
        dt_index = 0
        tupl = (1, 1)
        function_output = cdt._generate_disconnected_einsum_operands_list(dt_index, tupl)
        expected_result = 'dt_i, t_i'
        assert function_output == expected_result


class Test_generate_disconnected_einsum_function_call_list:

    def test_basic(self):
        partition = (1, 1)
        order = 2
        function_output = cdt._generate_disconnected_einsum_function_call_list(partition, order)
        expected_result = ["np.einsum('aci, cbj->abij', dt_i, t_i)", "np.einsum('aci, cbj->abij', t_i, dt_i)"]
        assert function_output == expected_result


class Test_generate_optimized_disconnected_einsum_function_call_list:

    def test_basic(self):
        partition = (1, 1)
        order = 2
        function_output = cdt._generate_optimized_disconnected_einsum_function_call_list(
            partition,
            order,
            iterator_name='optimized_einsum'
        )
        expected_result = ['next(optimized_einsum)(dt_i, t_i)', 'next(optimized_einsum)(t_i, dt_i)']
        assert function_output == expected_result


class Test_construct_linked_disconnected_definition:

    def test_basic(self):
        return_string = ''
        order = 2
        function_output = cdt._construct_linked_disconnected_definition(
            return_string,
            order,
            opt_einsum=False,
            iterator_name='optimized_einsum'
        )
        expected_result = str(
            '\n'
            'def _order_2_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'
            '    """Calculate all uniquely linked disconnected terms generated from the wave operator ansatz.\n'
            '    This means for order 5 we include terms such as (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'
            '    But not terms (5), (3, 2), (2, 2, 1)\n'
            '    """\n'
            '    # unpack the `t_args` and \'dt_args\'\n'
            '    t_i, *unusedargs = t_args\n'
            '    dt_i, *unusedargs = dt_args\n'
        )
        assert function_output == expected_result

    def test_einsum(self):
        return_string = ''
        order = 2
        function_output = cdt._construct_linked_disconnected_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )
        expected_result = str(
            '\n'
            'def _order_2_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list):\n'
            '    """Calculate all uniquely linked disconnected terms generated from the wave operator ansatz.\n'
            '    This means for order 5 we include terms such as (4, 1), (3, 1, 1), (2, 1, 1, 1)\n'
            '    But not terms (5), (3, 2), (2, 2, 1), (1, 1, 1, 1, 1)\n'
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'
            '    """\n'
            '    # unpack the `t_args` and \'dt_args\'\n'
            '    t_i, *unusedargs = t_args\n'
            '    dt_i, *unusedargs = dt_args\n'
            '    # make an iterable out of the `opt_path_list`\n'
            '    optimized_einsum = iter(opt_path_list)\n'
        )
        assert function_output == expected_result


class Test_write_linked_disconnected_strings:

    def test_basic(self):
        order = 2
        function_output = cdt._write_linked_disconnected_strings(order, opt_einsum=False)
        expected_result = open(
            "tests/deep/files/test_code_dt_equations/write_linked_disconnected_strings_basic/expected_result.py", "r"
        )
        assert function_output == expected_result.read()

    def test_low_order(self):
        order = 1
        function_output = cdt._write_linked_disconnected_strings(order, opt_einsum=False)
        expected_result = str(
            '\n'
            'def _order_1_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'
            '    """Exists for error checking."""\n'
            '    raise Exception(\n'
            '        "the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"\n'
            '        "which requires a residual of at least 2nd order"\n'
            '    )\n'
        )
        assert function_output == expected_result

    def test_einsum(self):
        order = 2
        function_output = cdt._write_linked_disconnected_strings(order, opt_einsum=True)
        expected_result = open(
            "tests/deep/files/test_code_dt_equations/write_linked_disconnected_strings_einsum/expected_result.py", "r"
        )
        assert function_output == expected_result.read()


class Test_construct_un_linked_disconnected_definition:

    def test_basic(self):
        return_string = ''
        order = 2
        function_output = cdt._construct_un_linked_disconnected_definition(
            return_string,
            order,
            opt_einsum=False,
            iterator_name='optimized_einsum'
        )
        expected_result = str(
            '\n'
            'def _order_2_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'
            '    """Calculate all uniquely un-linked disconnected terms generated from the wave operator ansatz.\n'
            '    This means for order 5 we include terms such as (3, 2), (2, 2, 1)\n'
            '    But not terms (5), (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'
            '    """\n'
            '    # unpack the `t_args` and \'dt_args\'\n'
            '    t_i, *unusedargs = t_args\n'
            '    dt_i, *unusedargs = dt_args\n'
        )
        assert function_output == expected_result

    def test_einsum(self):
        return_string = ''
        order = 2
        function_output = cdt._construct_un_linked_disconnected_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )
        expected_result = str(
            '\n'
            'def _order_2_un_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list):\n'
            '    """Calculate all uniquely un-linked disconnected terms generated from the wave operator ansatz.\n'
            '    This means for order 5 we include terms such as (3, 2), (2, 2, 1)\n'
            '    But not terms (5), (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'
            '    """\n'
            '    # unpack the `t_args` and \'dt_args\'\n'
            '    t_i, *unusedargs = t_args\n'
            '    dt_i, *unusedargs = dt_args\n'
            '    # make an iterable out of the `opt_path_list`\n'
            '    optimized_einsum = iter(opt_path_list)\n'
        )
        assert function_output == expected_result


class Test_write_un_linked_disconnected_strings:

    def test_high_order(self):
        order = 5
        function_output = cdt._write_un_linked_disconnected_strings(order, opt_einsum=False)
        expected_result = open(
            "tests/deep/files/test_code_dt_equations/write_un_linked_disconnected_strings_high_order/expected_result.py", "r"
        )
        assert function_output == expected_result.read()

    def test_high_order_einsum(self):
        order = 5
        function_output = cdt._write_un_linked_disconnected_strings(order, opt_einsum=True)
        expected_result = open(
            "tests/deep/files/test_code_dt_equations/write_un_linked_disconnected_strings_high_order_einsum/expected_result.py", "r"
        )
        assert function_output == expected_result.read()

    def test_low_order(self):
        order = 2
        function_output = cdt._write_un_linked_disconnected_strings(order, opt_einsum=False)
        expected_result = str(
            '\n'
            'def _order_2_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'
            '    """Exists for error checking."""\n'
            '    raise Exception(\n'
            '        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n'
            '        "which requires a residual of at least 4th order"\n'
            '    )\n'
        )
        assert function_output == expected_result


class Test_construct_dt_amplitude_definition:

    def test_basic(self):
        return_string = ''
        order = 2
        function_output = cdt._construct_dt_amplitude_definition(
            return_string,
            order,
            opt_einsum=False,
            iterator_name='optimized_einsum'
        )
        expected_result = str(
            '\n'
            'def _calculate_order_2_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):\n'
            '    """Calculate the derivative of the 2 t-amplitude for use in the calculation of the residuals."""\n'
            '    # unpack the `w_args`\n'
            '    w_i, w_ij, *unusedargs = w_args\n'
        )
        assert function_output == expected_result

    def test_einsum(self):
        return_string = ''
        order = 3
        function_output = cdt._construct_dt_amplitude_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )
        expected_result = str(
            '\n'
            'def _calculate_order_3_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_epsilon, opt_path_list):\n'
            '    """Calculate the derivative of the 3 t-amplitude for use in the calculation of the residuals.\n'
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'
            '    """\n    # unpack the `w_args`\n    w_i, w_ij, w_ijk, *unusedargs = w_args\n'
            '    # make an iterable out of the `opt_path_list`\n'
            '    optimized_einsum = iter(opt_path_list)\n'
        )
        assert function_output == expected_result


class Test_write_dt_amplitude_strings:

    def test_basic(self):
        order = 2
        function_output = cdt._write_dt_amplitude_strings(order, opt_einsum=False)
        expected_result = open(
            "tests/deep/files/test_code_dt_equations/write_dt_amplitude_strings_basic/expected_result.py", "r"
        )
        assert function_output == expected_result.read()

    def test_high_order(self):
        order = 5
        function_output = cdt._write_dt_amplitude_strings(order, opt_einsum=False)
        expected_result = open(
            "tests/deep/files/test_code_dt_equations/write_dt_amplitude_strings_high_order/expected_result.py", "r"
        )
        assert function_output == expected_result.read()

    def test_einsum(self):
        order = 2
        function_output = cdt._write_dt_amplitude_strings(order, opt_einsum=True)
        expected_result = open(
            "tests/deep/files/test_code_dt_equations/write_dt_amplitude_strings_einsum/expected_result.py", "r"
        )
        assert function_output == expected_result.read()

    def test_einsum_high_order(self):
        order = 5
        function_output = cdt._write_dt_amplitude_strings(order, opt_einsum=True)
        expected_result = open(
            "tests/deep/files/test_code_dt_equations/write_dt_amplitude_strings_einsum_high_order/expected_result.py", "r"
        )
        assert function_output == expected_result.read()


class Test_write_master_dt_amplitude_function:

    def test_basic(self):
        order = 2
        function_output = cdt._write_master_dt_amplitude_function(order, opt_einsum=False)
        expected_result = str(
            '\n'
            'def solve_doubles_equations(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):\n'
            '    """Compute the change in the t_ij term (doubles)"""\n'
            '\n'
            '    if not trunc.doubles:\n'
            '        raise Exception(\n'
            '            "It appears that doubles is not true, this cannot be."\n'
            '            "Something went terribly wrong!!!"\n'
            '        )\n'
            '    dt_ij = _calculate_order_2_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args)\n'
            '    return dt_ij\n'
        )
        assert function_output == expected_result

    def test_einsum(self):
        order = 2
        function_output = cdt._write_master_dt_amplitude_function(order, opt_einsum=True)
        expected_result = str(
            '\n'
            'def solve_doubles_equations_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list):\n'
            '    """Compute the change in the t_ij term (doubles)"""\n'
            '\n'
            '    if not trunc.doubles:\n'
            '        raise Exception(\n'
            '            "It appears that doubles is not true, this cannot be."\n'
            '            "Something went terribly wrong!!!"\n'
            '        )\n'
            '    dt_ij = _calculate_order_2_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list)\n'
            '    return dt_ij\n'
        )
        assert function_output == expected_result


class Test_write_optimized_dt_amplitude_paths_function:

    def test_basic(self):
        max_order = 2
        function_output = cdt._write_optimized_dt_amplitude_paths_function(max_order)
        expected_result = str(
            '\n'
            'def compute_optimized_paths(A, N, truncation):\n'
            '    """Calculate optimized paths for the einsum calls up to `highest_order`."""\n'
            '\n'
            '    order_1_list, order_2_list, order_3_list = [], [], []\n'
            '    order_4_list, order_5_list, order_6_list = [], [], []\n'
            '\n'
            '    return [None]\n'
        )
        assert function_output == expected_result


class Test_generate_dt_amplitude_string:

    def test_basic(self):
        max_order = 2
        function_output = cdt.generate_dt_amplitude_string(max_order, s1=75, s2=28)
        expected_result = open(
            "tests/deep/files/test_code_dt_equations/generate_dt_amplitude_string/expected_result.py", "r"
        )
        assert function_output == expected_result.read()


class Test_generate_dt_amplitude_equations_file:

    def test_run_main(self):
        cdt.generate_dt_amplitude_equations_file(2, path="./dt_amplitude_equations.py")

