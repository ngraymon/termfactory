#
#   python3 driver.py -t (1-2) (1-4) 1 (1-4) (1-4)  -lhs -c

# system imports
# from logging import exception
from os.path import abspath, dirname, join, basename
import pytest

# local imports
from truncation_keys import TruncationsKeys as tkeys
from latex_eT_zhz import _filter_out_valid_eTz_terms
from code_eT_zhz import (
    generate_eT_zhz_python,
    _simplify_eT_zhz_python_prefactor,
    _write_third_eTz_einsum_python,
    #
    generate_eT_taylor_expansion,
    generate_pruned_H_operator,
    generate_omega_operator,
    generate_z_operator,
)

# set the path (`root_dir`) to the files we need to compare against
deep_dir = dirname(abspath(__file__))
root_dir = join(deep_dir, 'files')
classtest = 'test_code_eT_zhz'


class Test_simplify_eT_zhz_python_prefactor():

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
    def trunc(self, A, B, C, D, E):
        eT_trunc = {
            tkeys.H: A,
            tkeys.CC: B,
            tkeys.T: C,
            tkeys.eT: D,
            tkeys.P: E
        }
        return eT_trunc

    def gen_path(self, truncations, tmpdir, kwargs):
        """ defines the path to the file with the python equations"""

        # f_term_string = "_no_f_terms" if kwargs['remove_f_terms'] else ''
        # gs_string = "ground_state_" if kwargs['only_ground_state'] else ''
        # path = f"./{gs_string}eT_zhz_equations{f_term_string}.py"
        # temporary naming scheme until a better one can be designed

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

        return join(tmpdir, path)

    def test_mass_gen(self, tmpdir, trunc, left_right_switch):
        """ Test hthe f """

        kwargs = {
            'only_ground_state': True,
            'remove_f_terms': False,
            'lhs_rhs': left_right_switch,
        }

        output_path = self.gen_path(trunc, tmpdir, kwargs)

        # add to the kwargs
        kwargs['path'] = output_path

        # do the hard work of generating all the code
        generate_eT_zhz_python(trunc, **kwargs)

        with open(output_path, 'r') as fp:
            file_data = fp.read()

        file_name = join(root_dir, classtest, basename(output_path))

        with open(file_name, 'r') as fp:
            reference_file_data = fp.read()

        assert file_data == reference_file_data, 'Fail'


class Test__write_third_eTz_einsum_python():
    """ The primary purpose here is to run the function
    `_write_third_eTz_einsum_python` with the flag
    `suppress_empty_if_checks` set to `False.
    """

    @pytest.fixture(scope="class", params=['LHS', 'RHS'])
    def lhs_rhs(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1])
    def max_h_rank(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1])
    def max_cc_rank(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1])
    def max_proj_rank(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1])
    def max_T_rank(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def max_exp_taylor_rank(self):
        return 1

    @pytest.fixture(scope="class", params=[True, False])
    def Pop(self, request, max_cc_rank, max_proj_rank):
        return generate_omega_operator(max_cc_rank, max_proj_rank)

    @pytest.fixture(scope="class", params=[True, False])
    def Hop(self, request, max_h_rank):
        return generate_pruned_H_operator(max_h_rank)

    @pytest.fixture(scope="class", params=[True, False])
    def Zop(self, request, max_cc_rank):
        return generate_z_operator(max_cc_rank, only_ground_state=request.param)

    @pytest.fixture(scope="class", params=[True, False])
    def eTop(self, request, max_T_rank, max_exp_taylor_rank):
        return generate_eT_taylor_expansion(max_T_rank, max_exp_taylor_rank)

    @pytest.fixture(scope="class", params=[1, 2])
    def operators(self, Pop, Hop, Zop, eTop):
        return (Pop, Hop, Zop, eTop)

    def test_no_suppression(self, operators, lhs_rhs):
        """ Simple test to get full coverage, need to redesign later """

        # for i, Proj in enumerate(operators[0].operator_list):

        t_term_list = []

        # just pick the highest rank projector
        Proj = operators[0].operator_list[-1]

        master_omega, H, Z, eT_taylor_expansion = operators

        # use the simple eT = 1 truncation and only consider HZ but not eTHZ terms
        zero_eT_term = operators[-1][0]   # only select the first term from eTop
        _filter_out_valid_eTz_terms(Proj, zero_eT_term, H, None, Z, t_term_list, lhs_rhs)

        string = _write_third_eTz_einsum_python(
            Proj.rank, operators, t_term_list, lhs_rhs,
            trunc_obj_name='truncation', b_loop_flag=True, suppress_empty_if_checks=False
        )

        return

    # should be expanded in the future
    # def notdone_test_collect_z_contributions(self):
    #     """ After `collect_z_contributions` is factored out
    #     this should test the function, specifically the

    #         if temp_z_list == []:
    #             return_array.append(z_header_if_string)
    #             return_array.append(f"{tabstr}{tab}pass")

    #     branch

    #     """

    #     @pytest.fixture(scope="function")
    #     def z_pair():
    #         """ x """

    #     def create_hamiltonian_rank_list(Zop, H, master_omega, t_term_list):
    #         """

    #         The full `hamiltonian_rank_list` is a four-deep nested dict.
    #         Each element is accessed like so
    #             hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][z_right.rank][prefactor]

    #         For testing purposes we only need the last two layers
    #             hamiltonian_rank_list[0][0][z_right.rank][prefactor]

    #         """

    #         z_rank_list = []

    #         # fill with empty dictionaries
    #         for i in range(Zop.maximum_rank+1):

    #             z_right_kwargs = {
    #                 'rank': right_z.rank,
    #                 'm': right_z.m,
    #                 # 'm_lhs': 0,
    #                 # 'm_t': [0, ],
    #                 # 'm_h': 0,
    #                 # 'm_l': 0,
    #                 'n': right_z.n,
    #                 # 'n_lhs': 0,
    #                 # 'n_t': [0, ],
    #                 # 'n_h': 0,
    #                 # 'n_l': 0,
    #             }

    #             z_right = disconnected_eT_z_right_operator_namedtuple(**z_right_kwargs)

    #             z_pair = (None, z_right)


    #             z_rank_list[z]
    #             z_rank_list.append(
    #                 dict([(i, {}) for i in range(master_omega.maximum_rank+1)])
    #             )

    #         for term in t_term_list:
    #             # unpack
    #             omega, t_list, h, z_pair, _ = term

    #             max_t_rank = max(t.rank for t in t_list)

    #             z_left, z_right = z_pair

    #             prefactor = '1'

    #             # build with permutations
    #             hamiltonian_rank_list[max(h.m, h.n)].setdefault(
    #                 max_t_rank, {}).setdefault(
    #                     z_right.rank, {}).setdefault(
    #                         prefactor, []
    #             )

    #             string = f"np.einsum('acj, cj -> ac', h[(0,1)], z[(1,0)])"

    #             # append that string to the current list
    #             hamiltonian_rank_list[max(h.m, h.n)][max_t_rank][z_right.rank][prefactor].append(
    #                 string
    #             )


    #         return hamiltonian_rank_list

    #     def test_suppression(self, H, master_omega):
    #         """ x """

    #         return_array = []

    #         hamiltonian_rank_list = create_hamiltonian_rank_list(H, master_omega)

    #         global suppress_empty_if_checks
    #         suppress_empty_if_checks = False

    #         collect_z_contributions(hamiltonian_rank_list, return_array, nof_tabs=0)

    #         del suppress_empty_if_checks

    #         print(return_array)

    #     def test_no_suppression():
    #         """ x """
    #         pass
