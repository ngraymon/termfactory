#
#   python3 driver.py -t (1-2) (1-4) 1 (1-4) (1-4)  -lhs -c

# system imports
# from logging import exception
from os.path import abspath, dirname, join
import pytest

# local imports
from truncation_keys import TruncationsKeys as tkeys
from code_eT_zhz import generate_eT_zhz_python, _simplify_eT_zhz_python_prefactor

# set the path (`root_dir`) to the files we need to compare against
deep_dir = dirname(abspath(__file__))
root_dir = join(deep_dir, 'files')
classtest = 'test_code_eT_zhz'

class Test_simplify_eT_zhz_python_prefactor:

    def test_disjoint(self):
        """basic test"""

        # input data
        numerator_list = ['1', '2', '3']
        denominator_list = ['4', '5', '6']

        # run function
        function_output = _simplify_eT_zhz_python_prefactor(numerator_list, denominator_list)
        expected_result = (['1', '2', '3'], ['4', '5', '6'])

        assert function_output == expected_result
    
    def test_duplicate_prefactors(self):
        """basic test"""

        # a > b

        # input data
        numerator_list = ['1', '2', '2']
        denominator_list = ['2']

        # run function
        function_output = _simplify_eT_zhz_python_prefactor(numerator_list, denominator_list)
        expected_result = (['1', '2'], [])

        assert function_output == expected_result

        # a < b

        # input data
        numerator_list = ['1', '2']
        denominator_list = ['2', '2', '2']

        # run function
        function_output = _simplify_eT_zhz_python_prefactor(numerator_list, denominator_list)
        expected_result = (['1'], ['2', '2'])

        assert function_output == expected_result

        # a == b

        # input data
        numerator_list = ['1']
        denominator_list = ['1', '2']

        # run function
        function_output = _simplify_eT_zhz_python_prefactor(numerator_list, denominator_list)
        expected_result = ([], ['2'])

        assert function_output == expected_result


def _gen_wrapper_eT_zhz_python(truncations, tmpdir, **kwargs):
    # the 's_taylor_max_order' isn't releveant for this execution pathway

    # f_term_string = "_no_f_terms" if kwargs['remove_f_terms'] else ''
    # gs_string = "ground_state_" if kwargs['only_ground_state'] else ''
    # path = f"./{gs_string}eT_zhz_equations{f_term_string}.py"

    # temporary naming scheme until a better one can be designed
    # also hot band equation generation has not been implemented anyways

    path = (
        "eT_zhz_eqs"
        f"_{kwargs['lhs_rhs']}"
        f"_H({truncations[tkeys.H]})"
        f"_P({truncations[tkeys.P]})"
        f"_T({truncations[tkeys.T]})"
        f"_exp({truncations[tkeys.eT]})"
        f"_Z({truncations[tkeys.CC]})"
        ".py"
    )
    output_path = join(tmpdir, path)
    kwargs['path'] = output_path

    generate_eT_zhz_python(truncations, **kwargs)


class Test_lhs_gen():

    @pytest.fixture(scope="class", params=[1, 2])
    def A(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2, 3, 4])
    def B(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1])
    def C(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2, 3, 4])
    def D(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2, 3, 4])
    def E(self, request):
        return request.param

    @pytest.fixture(scope="class", params=['LHS', 'RHS'])
    def left_right_switch(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def truncations(self, A, B, C, D, E):
        eT_trunc = {
            tkeys.H: A,
            tkeys.CC: B,
            tkeys.T: C,
            tkeys.eT: D,
            tkeys.P: E
        }
        return eT_trunc

    def test_mass_gen(self, tmpdir, truncations, left_right_switch):

        _gen_wrapper_eT_zhz_python(
            truncations,
            tmpdir,
            only_ground_state=True,
            remove_f_terms=False,
            lhs_rhs=left_right_switch
        )
        path = (
            "eT_zhz_eqs"
            f"_{left_right_switch}"
            f"_H({truncations[tkeys.H]})"
            f"_P({truncations[tkeys.P]})"
            f"_T({truncations[tkeys.T]})"
            f"_exp({truncations[tkeys.eT]})"
            f"_Z({truncations[tkeys.CC]})"
            ".py"
        )
        output_path = join(tmpdir, path)
        # output_path = path

        with open(output_path, 'r') as fp:
            file_data = fp.read()

        file_name = join(root_dir, classtest, path)

        with open(file_name, 'r') as fp:
            reference_file_data = fp.read()

        assert file_data == reference_file_data, 'Fail'
