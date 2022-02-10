# system imports
# import pytest
from pathlib import Path
root_dir = str(Path(__file__).parent)+'\\files\\'
classtest = 'test_code_dt_equations\\'


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
        expected_result = open(
            root_dir+classtest+"construct_linked_disconnected_definition_basic_out.py", "r"
        )
        assert function_output == expected_result.read()

    def test_einsum(self):
        return_string = ''
        order = 2
        function_output = cdt._construct_linked_disconnected_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )
        expected_result = open(
            root_dir+classtest+"construct_linked_disconnected_definition_einsum_out.py", "r"
        )
        assert function_output == expected_result.read()


class Test_write_linked_disconnected_strings:

    def test_basic(self):
        order = 2
        function_output = cdt._write_linked_disconnected_strings(order, opt_einsum=False)
        expected_result = open(
            root_dir+classtest+"write_linked_disconnected_strings_basic_out.py", "r"
        )
        assert function_output == expected_result.read()

    def test_low_order(self):
        order = 1
        function_output = cdt._write_linked_disconnected_strings(order, opt_einsum=False)
        expected_result = open(
            root_dir+classtest+"write_linked_disconnected_strings_low_order_out.py", "r"
        )
        assert function_output == expected_result.read()

    def test_einsum(self):
        order = 2
        function_output = cdt._write_linked_disconnected_strings(order, opt_einsum=True)
        expected_result = open(
            root_dir+classtest+"write_linked_disconnected_strings_einsum_out.py", "r"
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
        expected_result = open(
            root_dir+classtest+"construct_un_linked_disconnected_definition_basic_out.py", "r"
        )
        assert function_output == expected_result.read()

    def test_einsum(self):
        return_string = ''
        order = 2
        function_output = cdt._construct_un_linked_disconnected_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )
        expected_result = open(
            root_dir+classtest+"construct_un_linked_disconnected_definition_einsum_out.py", "r"
        )
        assert function_output == expected_result.read()


class Test_write_un_linked_disconnected_strings:

    def test_high_order(self):
        order = 5
        function_output = cdt._write_un_linked_disconnected_strings(order, opt_einsum=False)
        expected_result = open(
            root_dir+classtest+"write_un_linked_disconnected_strings_high_order_out.py", "r"
        )
        assert function_output == expected_result.read()

    def test_high_order_einsum(self):
        order = 5
        function_output = cdt._write_un_linked_disconnected_strings(order, opt_einsum=True)
        expected_result = open(
            root_dir+classtest+"write_un_linked_disconnected_strings_high_order_einsum_out.py", "r"
        )
        assert function_output == expected_result.read()

    def test_low_order(self):
        order = 2
        function_output = cdt._write_un_linked_disconnected_strings(order, opt_einsum=False)
        expected_result = open(
            root_dir+classtest+"write_un_linked_disconnected_strings_low_order_out.py", "r"
        )
        assert function_output == expected_result.read()


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
        expected_result = open(
            root_dir+classtest+"construct_dt_amplitude_definition_basic_out.py", "r"
        )
        assert function_output == expected_result.read()

    def test_einsum(self):
        return_string = ''
        order = 3
        function_output = cdt._construct_dt_amplitude_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )
        expected_result = open(
            root_dir+classtest+"construct_dt_amplitude_definition_einsum_out.py", "r"
        )
        assert function_output == expected_result.read()


class Test_write_dt_amplitude_strings:

    def test_basic(self):
        order = 2
        function_output = cdt._write_dt_amplitude_strings(order, opt_einsum=False)
        expected_result = open(
            root_dir+classtest+"write_dt_amplitude_strings_basic_out.py", "r"
        )
        assert function_output == expected_result.read()

    def test_high_order(self):
        order = 5
        function_output = cdt._write_dt_amplitude_strings(order, opt_einsum=False)
        expected_result = open(
            root_dir+classtest+"write_dt_amplitude_strings_high_order_out.py", "r"
        )
        assert function_output == expected_result.read()

    def test_einsum(self):
        order = 2
        function_output = cdt._write_dt_amplitude_strings(order, opt_einsum=True)
        expected_result = open(
            root_dir+classtest+"write_dt_amplitude_strings_einsum_out.py", "r"
        )
        assert function_output == expected_result.read()

    def test_einsum_high_order(self):
        order = 5
        function_output = cdt._write_dt_amplitude_strings(order, opt_einsum=True)
        expected_result = open(
            root_dir+classtest+"write_dt_amplitude_strings_einsum_high_order_out.py", "r"
        )
        assert function_output == expected_result.read()


class Test_write_master_dt_amplitude_function:

    def test_basic(self):
        order = 2
        function_output = cdt._write_master_dt_amplitude_function(order, opt_einsum=False)
        expected_result = open(
            root_dir+classtest+"write_master_dt_amplitude_function_basic_out.py", "r"
        )
        assert function_output == expected_result.read()

    def test_einsum(self):
        order = 2
        function_output = cdt._write_master_dt_amplitude_function(order, opt_einsum=True)
        expected_result = open(
            root_dir+classtest+"write_master_dt_amplitude_function_einsum_out.py", "r"
        )
        assert function_output == expected_result.read()


class Test_write_optimized_dt_amplitude_paths_function:

    def test_basic(self):
        max_order = 2
        function_output = cdt._write_optimized_dt_amplitude_paths_function(max_order)
        expected_result = open(
            root_dir+classtest+"write_optimized_dt_amplitude_paths_function_out.py", "r"
        )
        assert function_output == expected_result.read()


class Test_generate_dt_amplitude_string:

    def test_basic(self):
        max_order = 2
        function_output = cdt.generate_dt_amplitude_string(max_order, s1=75, s2=28)
        expected_result = open(
            root_dir+classtest+"generate_dt_amplitude_string_out.py", "r"
        )
        assert function_output == expected_result.read()


class Test_generate_dt_amplitude_equations_file:

    def test_run_main(self):
        cdt.generate_dt_amplitude_equations_file(2, path="./dt_amplitude_equations.py")

