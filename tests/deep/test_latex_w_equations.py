# system imports
# import re
# import pytest
from os.path import abspath, dirname, join
deep_dir = dirname(abspath(__file__))
root_dir = join(deep_dir, 'files')
classtest = 'test_latex_w_equations'
# local imports
from . import context
import latex_w_equations as lw
# import namedtuple_defines as nt

# TODO classname == function maybe in the future?
class Test_Latex_of_W_operators:

    def test_remove_list_item(self):
        function_output = lw.remove_list_item(1, [1, 2, 3])
        expected_result = [2, 3]
        assert function_output == expected_result

    def test_count_items(self):
        lis = [1, 1, 2]
        function_output = lw.count_items(lis)
        expected_result = {1: 2, 2: 1}
        assert function_output == expected_result

    def test_generate_w_prefactor_test_1(self):
        """no prefact"""
        w_dict = {2: 1, 1: 1}
        function_output = lw.generate_w_prefactor(w_dict)
        expected_result = ''
        assert function_output == expected_result

    def test_generate_w_prefactor_test_2(self):
        """return latex"""
        w_dict = {1: 2}
        function_output = lw.generate_w_prefactor(w_dict)
        expected_result = '\\frac{1}{2!}'
        assert function_output == expected_result

    def test_generate_labels_on_w(self):
        order = 2
        function_output = lw.generate_labels_on_w(order, is_excited=True)
        expected_result = [
            lw.w_namedtuple_latex(m=1, n=0),
            lw.w_namedtuple_latex(m=2, n=0),
            lw.w_namedtuple_latex(m=1, n=1)
        ]
        assert function_output == expected_result

    def test_ground_state_w_equations_latex(self):
        max_w_order = 2
        function_output = lw.ground_state_w_equations_latex(max_w_order, path="./ground_state_w_equations.tex")
        func_name = "ground_state_w_equations_latex_out.py"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()
        assert function_output == expected_result

    def test_generate_t_terms_group(self):
        w_ntuple = lw.w_namedtuple_latex(m=2, n=0)
        function_output = lw.generate_t_terms_group(w_ntuple)
        expected_result = [[(2, 0)], [(1, 0), (1, 0)]]
        assert function_output == expected_result

    def test_generate_t_terms_group_zero_order(self):
        w_ntuple = lw.w_namedtuple_latex(m=0, n=0)
        function_output = lw.generate_t_terms_group(w_ntuple)
        expected_result = 'i\\left(\\varepsilon\\right)'
        assert function_output == expected_result

    def test_excited_state_w_equations_latex(self, tmpdir):
        """runs main function and compares it to a reference file"""

        output_path = join(tmpdir, "latex_test_excited_state_w_equations_latex.txt")

        max_w_order = 3
        lw.excited_state_w_equations_latex(max_w_order, output_path)

        with open(output_path, 'r') as fp:
            file_data = fp.read()

        func_name = "thermal_w_equations.txt"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            reference_file_data = fp.read()

        assert file_data == reference_file_data, 'Fail'
