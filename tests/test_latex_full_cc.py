# system imports
import re
import pytest
# local imports
from .context import latex_full_cc as fcc
from .context import namedtuple_defines as nt

class Test_Operator_Gens:

    def test_generate_omega_operator(self):
        output=fcc.generate_omega_operator(maximum_cc_rank=2, omega_max_order=3)
        result=nt.omega_namedtuple(maximum_rank=2, operator_list=[nt.general_operator_namedtuple(name='', rank=0, m=0, n=0),
                                                     nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1), 
                                                     nt.general_operator_namedtuple(name='d', rank=1, m=1, n=0), 
                                                     nt.general_operator_namedtuple(name='bb', rank=2, m=0, n=2), 
                                                     nt.general_operator_namedtuple(name='db', rank=2, m=1, n=1), 
                                                     nt.general_operator_namedtuple(name='dd', rank=2, m=2, n=0)])
        assert output==result

    def test_generate_s_operator(self):
        output=fcc.generate_full_cc_hamiltonian_operator(maximum_rank=2)
        result=nt.hamiltonian_namedtuple(maximum_rank=2, operator_list=[fcc.h_operator_namedtuple(rank=0, m=0, n=0),
                                                                        fcc.h_operator_namedtuple(rank=1, m=0, n=1), 
                                                                        fcc.h_operator_namedtuple(rank=2, m=0, n=2), 
                                                                        fcc.h_operator_namedtuple(rank=1, m=1, n=0), 
                                                                        fcc.h_operator_namedtuple(rank=2, m=1, n=1), 
                                                                        fcc.h_operator_namedtuple(rank=2, m=2, n=0)])
        assert output==result

    def test_generate_full_cc_hamiltonian_operator(self):
        output=fcc.generate_s_operator(maximum_cc_rank=2, only_ground_state=False)
        result=fcc.s_operator_namedtuple(maximum_rank=2, operator_list=[nt.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1), 
                                                                        nt.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2), 
                                                                        nt.general_operator_namedtuple(name='s^1', rank=1, m=1, n=0), 
                                                                        nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1), 
                                                                        nt.general_operator_namedtuple(name='s^2', rank=2, m=2, n=0)])
        assert output==result

    def test_generate_s_taylor_expansion(self):
        #kinda messy produces all combinations with lists of lists, gonna come back to this maybe not worth testing
        fcc.generate_s_taylor_expansion(maximum_cc_rank=3, s_taylor_max_order=1, only_ground_state=False)
  
class Test_Validate_Op_Pairings:

    def test_t_joining_with_t_terms(self):
        omega=nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        h=fcc.h_operator_namedtuple(rank=0, m=0, n=0)
        s_list=[nt.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        nof_creation_ops=1
        output=fcc._t_joining_with_t_terms(omega, h, s_list, nof_creation_ops)
        result=False
        assert output==result

        omega=nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        h=fcc.h_operator_namedtuple(rank=0, m=0, n=0)
        s_list=[nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1)]
        nof_creation_ops=1
        output=fcc._t_joining_with_t_terms(omega, h, s_list, nof_creation_ops)
        result=True
        assert output==result

    def test_omega_joining_with_itself(self):
        # (omega.m == 0) or (omega.n == 0) return False
        omega=nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        h=fcc.h_operator_namedtuple(rank=0, m=0, n=0)
        s_list=[nt.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        output=fcc._omega_joining_with_itself(omega, h, s_list)
        assert output==False

        # (omega.n > 0 and h.m > 0) or (omega.m > 0 and h.n > 0) return False
        omega=nt.general_operator_namedtuple(name='db', rank=2, m=1, n=1)
        h=fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_list=[nt.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        output=fcc._omega_joining_with_itself(omega, h, s_list)
        assert output==False

        # for s in s_list: if (omega.n > 0 and s.m > 0) or (omega.m > 0 and s.n > 0): return False
        omega=nt.general_operator_namedtuple(name='db', rank=2, m=1, n=1)
        h=fcc.h_operator_namedtuple(rank=0, m=0, n=0)
        s_list=[nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1)]
        output=fcc._omega_joining_with_itself(omega, h, s_list)
        assert output==False

        # else == True
        omega=nt.general_operator_namedtuple(name='db', rank=2, m=1, n=1)
        h=fcc.h_operator_namedtuple(rank=0, m=0, n=0)
        s_list=[nt.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        output=fcc._omega_joining_with_itself(omega, h, s_list)
        assert output==True

    def test_h_joining_with_itself(self):
        # (h.m == 0) or (h.n == 0): return False
        omega=nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        h=fcc.h_operator_namedtuple(rank=0, m=0, n=0)
        s_list=[nt.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        output=fcc._h_joining_with_itself(omega, h, s_list)
        assert output==False

        # if (omega.n > 0 and h.m > 0) or (omega.m > 0 and h.n > 0): return False
        omega=nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        h=fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_list=[nt.general_operator_namedtuple(name='s^1', rank=1, m=1, n=0)]
        output=fcc._h_joining_with_itself(omega, h, s_list)
        assert output==False

        # for s in s_list: if (omega.n > 0 and s.m > 0) or (omega.m > 0 and s.n > 0): return False
        omega=nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        h=fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_list=[nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1)]
        output=fcc._h_joining_with_itself(omega, h, s_list)
        assert output==False

        # else == True
        omega=nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        h=fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_list=[nt.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        output=fcc._h_joining_with_itself(omega, h, s_list)
        assert output==True

class Test_Gen_Ops:
    
    def test_generate_valid_s_n_operator_permutations(self):
        omega=nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        h=fcc.h_operator_namedtuple(rank=0, m=0, n=0)
        s_series_term=[[nt.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]]
        output=fcc._generate_valid_s_n_operator_permutations(omega,h,s_series_term)
        result=[[nt.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]]
        assert result==output

    def test_generate_all_valid_t_connection_permutations(self):
        omega=nt.general_operator_namedtuple(name='dd', rank=2, m=2, n=0)
        h=fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_term_list=[nt.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2), nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1)]
        output=fcc._generate_all_valid_t_connection_permutations(omega, h, s_term_list, log_invalid=True)
        result=([((0, 0), (0, 1))], [((1, 1), (1, 0)), ((2, 0), (0, 1))])
        assert result==output

    def test_generate_all_omega_h_connection_permutations(self):
        omega=nt.general_operator_namedtuple(name='dd', rank=2, m=2, n=0)
        h=fcc.h_operator_namedtuple(rank=2, m=2, n=0)
        valid_permutations=[[nt.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2), nt.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)]]
        output=fcc._generate_all_omega_h_connection_permutations(omega, h, valid_permutations, found_it_bool=False)
        result=[
                    [
                    fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0), 
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2)
                    ], 
                    [
                    fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1), 
                    fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1)
                    ],
                    [fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),
                    fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0)
                    ]
                ]
        assert result==output

    def test_remove_duplicate_s_permutations(self): #TODO add dupe test
        s_list=[[fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)], [fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)]]
        output=fcc._remove_duplicate_s_permutations(s_list)
        result={(fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)}
        assert result==output

    def test_generate_explicit_connections(self):

        # normal input
        omega=nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        h=fcc.h_operator_namedtuple(rank=2, m=0, n=2)
        unique_s_permutations={(fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0))}
        output=fcc._generate_explicit_connections(omega, h, unique_s_permutations)
        result=[
                    [
                    fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0, 0], n_t=[0, 0]),
                    fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]),
                        (
                        fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0), 
                        fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
                        )
                    ]
                ]
        assert result==output

        # TODO if req, also add raise excepts for if states?
        # invalid term case 1
        # invalid term case 2

    def test_remove_f_zero_terms(self):
        #TODO exclude log replace with raise exceptions
        labeled_permutations=[
                                [
                                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0, 0], n_t=[0, 0]),
                                fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]),
                                    (
                                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0), 
                                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
                                    )
                                ]
                            ]
        output=fcc._remove_f_zero_terms(labeled_permutations)
        assert output==labeled_permutations

    def test_filter_out_valid_s_terms(self):
        #TODO test list mutation
        omega=nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        H=fcc.hamiltonian_namedtuple(
                                    maximum_rank=2,
                                    operator_list=[
                                                    fcc.h_operator_namedtuple(rank=0, m=0, n=0),
                                                    fcc.h_operator_namedtuple(rank=1, m=0, n=1),
                                                    fcc.h_operator_namedtuple(rank=2, m=0, n=2),
                                                    fcc.h_operator_namedtuple(rank=1, m=1, n=0),
                                                    fcc.h_operator_namedtuple(rank=2, m=1, n=1),
                                                    fcc.h_operator_namedtuple(rank=2, m=2, n=0)
                                    ])
        s_series_term=nt.general_operator_namedtuple(name='1', rank=0, m=0, n=0)
        term_list=[]
        total_list=[]
        output=fcc._filter_out_valid_s_terms(omega, H, s_series_term, term_list, total_list, remove_f_terms=True)
        assert output==None

    def test_seperate_s_terms_by_connection(self):
        # TODO add further tests
        total_list=[
                        [
                            fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                            fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]), 
                            (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)
                        ],
                        [
                            fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                            fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]), 
                            (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),)
                        ],
                        [
                            fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]), 
                            fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[1], n_t=[0]), 
                            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),)
                        ]
                    ]
        output=fcc._seperate_s_terms_by_connection(total_list)
        result=([[fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)], [fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]), (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),)], [fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[1], n_t=[0]), (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),)]], [], [])
        assert result==output

class Test_upper_lower_latex_indicies:

    def test_build_h_term_latex_labels_normal(self):
        h=fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        output=fcc._build_h_term_latex_labels(h, condense_offset=0, color=True)
        result='\\bh^'+r'{\blue{}\red{}}_'r'{\blue{i}\red{}}'
        assert output==result

    def test_build_h_term_latex_labels_zero_case(self):
        h=fcc.connected_h_operator_namedtuple(rank=0, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        output=fcc._build_h_term_latex_labels(h, condense_offset=0, color=True)
        result='\\bh_0'
        assert output==result

    def test_build_t_term_latex_labels_zero(self):
        term=fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0)
        offset_dict={'summation_upper': 0, 'summation_lower': 0, 'unlinked': 0}
        output=fcc._build_t_term_latex_labels(term, offset_dict, color=True)
        result='^{}_{}'
        assert output==result

    def test_build_t_term_latex_labels_subscript(self):
        term=fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0)
        offset_dict={'summation_upper': 1, 'summation_lower': 0, 'unlinked': 0}
        output=fcc._build_t_term_latex_labels(term, offset_dict, color=True)
        result='^{}_'+r'{\blue{i}\red{}}'
        assert output==result

    def test_build_t_term_latex_labels_superscript(self):
        term=fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
        offset_dict={'summation_upper': 0, 'summation_lower': 0, 'unlinked': 0}
        output=fcc._build_t_term_latex_labels(term, offset_dict, color=True)
        result=r'^{\blue{i}\red{}}_{}'
        assert output==result

    def test_build_t_term_latex(self):
        s=fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
        output=fcc._build_t_term_latex(s, h=None)
        result=r'\bt^{\blue{}\red{z}}_{}'
        assert output==result

    def test_build_t_term_latex_non_none_h(self): #TODO is this really testing the function?
        s=fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
        h=fcc.connected_h_operator_namedtuple(rank=0, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        output=fcc._build_t_term_latex(s, h)
        result=r'\bt^{\blue{}\red{z}}_{}'
        assert output==result

class Test_latex_writing:

    def test_validate_s_terms_valid(self):
        s_list=(fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)
        output=fcc._validate_s_terms(s_list)
        result=None
        assert output==result

    def test_validate_s_terms_invalid(self):
        s_list=("weeee")
        with pytest.raises(AssertionError):
            fcc._validate_s_terms(s_list)

    def test_generate_linked_common_terms(self):
        term_list=[
                    [
                    fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0], n_t=[1]), 
                    fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]), 
                    (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)
                    ]
                ]
        output=fcc._generate_linked_common_terms(term_list)
        result=[[fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)]]
        assert output==result

    def test_prepare_condensed_terms(self):
        #link condesed
        term_list=[
                    [
                    fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0], n_t=[1]),
                    fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=1, m_t=[0], n_t=[0]),
                    (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)
                    ], 
                    [
                    fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[1], n_t=[0]),
                    fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]),
                    (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1),)
                    ]
                ]
        output=fcc.prepare_condensed_terms(term_list, linked_condense=True, unlinked_condense=False)
        result=[[fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)], [fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1)]]
        assert output==result

        #unlink condensed
        term_list=[
                    [
                    fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2], n_t=[0]),
                    fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]), 
                    (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),)
                    ],
                    [
                    fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]),
                    fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]),
                    (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0))],
                    [
                    fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]),
                    fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 0]),
                    (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2), fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0))]]
        output=fcc.prepare_condensed_terms(term_list, linked_condense=False, unlinked_condense=True)
        result=fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2)
        assert output==result

        #omega rank 3
        term_list=[
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1], n_t=[2]),
                        fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]),
                        (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                        fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                        fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 2]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=2, n_h=0, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                        fcc.connected_h_operator_namedtuple(rank=3, m=0, n=3, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 3]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=3, n_h=0, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                        fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 0]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                        fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 1]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=1, n_h=1, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                        fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 2]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=2, n_h=1, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                        fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[0, 2], n_t=[0, 0]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                        fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=0, n_o=0, m_t=[0, 2], n_t=[0, 1]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=1, n_h=2, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                        fcc.connected_h_operator_namedtuple(rank=3, m=3, n=0, m_o=0, n_o=0, m_t=[0, 3], n_t=[0, 0]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=0, n_h=3, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                        fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0, 0], n_t=[0, 1, 1]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                        fcc.connected_h_operator_namedtuple(rank=3, m=0, n=3, m_o=0, n_o=0, m_t=[0, 0, 0], n_t=[0, 1, 2]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                            fcc.connected_namedtuple(m_h=2, n_h=0, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                        fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=0, m_t=[0, 1, 0], n_t=[0, 0, 1]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                        fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=0, n_o=0, m_t=[0, 0, 1], n_t=[0, 1, 1]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                            fcc.connected_namedtuple(m_h=1, n_h=1, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                        fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=0, n_o=0, m_t=[0, 1, 0], n_t=[0, 0, 2]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                            fcc.connected_namedtuple(m_h=2, n_h=0, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                        fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[0, 1, 1], n_t=[0, 0, 0]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                            fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                        fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=0, n_o=0, m_t=[0, 1, 1], n_t=[0, 0, 1]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                            fcc.connected_namedtuple(m_h=1, n_h=1, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                        fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=0, n_o=0, m_t=[0, 2, 0], n_t=[0, 0, 1]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0),
                            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                        ),
                    ],
                    [
                        fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                        fcc.connected_h_operator_namedtuple(rank=3, m=3, n=0, m_o=0, n_o=0, m_t=[0, 1, 2], n_t=[0, 0, 0]),
                        (
                            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                            fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                            fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0),
                        ),
                    ],
                ]
        output=fcc.prepare_condensed_terms(term_list, linked_condense=True, unlinked_condense=False)
        result=[[fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1)]]
        assert output==result

    def test_simplify_full_cc_python_prefactor(self):
        numerator_list=[]
        denominator_list=['2!', '2!']
        output=fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        result=([], ['2!', '2!'])
        assert output==result

        numerator_list=['2!']
        denominator_list=['2!']
        output=fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        result=([], [])
        assert output==result

        numerator_list=['2!']
        denominator_list=['3!']
        output=fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        result=(['2!'], ['3!'])
        assert output==result

        numerator_list=['2!', '2!', '2!']
        denominator_list=['2!', '2!']
        output=fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        result=(['2!'], [])
        assert output==result

        numerator_list=['2!', '2!']
        denominator_list=['2!', '2!', '2!']
        output=fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        result=([], ['2!'])
        assert output==result

    def test_build_latex_prefactor(self):
        # single h
        h=fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0])
        t_list=(fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)
        output=fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        result=''
        assert output==result

        #t_list len>1
        h=fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1])
        t_list=(fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0))
        output=fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        result=r'\frac{1}{2!2!}'
        assert output==result

        # x>1 case
        h=fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1])
        t_list=(fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0))
        output=fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        result=r'\frac{1}{2!}'
        assert output==result

        # h.m>1 case and h.n > 1 case
        h=fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[2], n_t=[0])
        t_list=(fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0),)
        output=fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        result=r'\frac{1}{2!}'
        assert output==result

        # case where numerator == '1' and denominator == '1'
        h=fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        t_list=(fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),)
        output=fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        result=''
        assert output==result

    def test_linked_condensed_adjust_t_terms(self):
        # basic
        common_linked_factor_list=[
                                    [
                                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)], 
                                    [fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)]
                                ]
        h=fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0])
        t_list=(fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)
        output=fcc._linked_condensed_adjust_t_terms(common_linked_factor_list, h, t_list)
        result=(1, [], {'summation_upper': 0, 'summation_lower': 0, 'unlinked': 2})
        assert output==result

        #case where 0 == h.m_o == h.n_o
        common_linked_factor_list=[
                                    [
                                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)], 
                                    [fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)]
                                ]
        h=fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 0])
        t_list=(fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0), fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0))
        output=fcc._linked_condensed_adjust_t_terms(common_linked_factor_list, h, t_list)
        result=(0, [], {'summation_lower': 0, 'summation_upper': 0, 'unlinked': 2})
        assert output==result

    def test_creates_f_prefactor(self):
        omega=fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0])
        h=fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0])
        output=fcc._creates_f_prefactor(omega, h)
        result=False
        assert output==result

    def test_creates_fbar_prefactor(self):
        omega=fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0])
        h=fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0])
        output=fcc._creates_fbar_prefactor(omega, h)
        result=False
        assert output==result

    def test_make_latex(self):
        #basic
        rank=0
        term_list=[]
        output=fcc._make_latex(rank, term_list, linked_condense=False, unlinked_condense=False, print_prefactors=True, color=False)
        result="()"
        assert output==result

        #linked_condense 
        rank=1
        term_list=[[fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=0, m_t=[0], n_t=[1]), fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)]]
        output=fcc._make_latex(rank, term_list, linked_condense=False, unlinked_condense=True, print_prefactors=True, color=True)
        result=r'(\disconnected{\bh_0})\bt^{\blue{}\red{z}}_{}'
        assert output==result

        #unlink_condense
        rank=2
        term_list=[[fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0], n_t=[1]), fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=0),)]]
        output=fcc._make_latex(rank, term_list, linked_condense=True, unlinked_condense=False, print_prefactors=True, color=True)
        result=r'(\bar{f}\bh^{\blue{}\red{x}}_{\blue{}\red{}})\bt^{\blue{}\red{zy}}_{}'
        assert output==result

        #_creates_f_prefactor(omega, h)==True
        rank=1
        term_list=[[fcc.connected_omega_operator_namedtuple(rank=1, m=1, n=0, m_h=1, n_h=0, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=1, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)], [fcc.connected_omega_operator_namedtuple(rank=1, m=1, n=0, m_h=0, n_h=0, m_t=[1], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]), (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1),)], [fcc.connected_omega_operator_namedtuple(rank=1, m=1, n=0, m_h=0, n_h=0, m_t=[1], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[1], n_t=[0]), (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1),)]]
        output=fcc._make_latex(rank, term_list, linked_condense=False, unlinked_condense=False, print_prefactors=True, color=True)
        result=r'(f\bh^{\blue{}\red{}}_{\blue{}\red{z}} + \bh^{\blue{}\red{}}_{\blue{i}\red{}}\bt^{\blue{i}\red{}}_{\blue{}\red{z}} + \bh^{\blue{i}\red{}}_{\blue{}\red{}}\bt^{}_{\blue{i}\red{z}})'
        assert output==result

        #_creates_f_prefactor(omega, h)==False
        rank=2
        term_list=[[fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=2, n_h=0, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=2, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)]]
        output=fcc._make_latex(rank, term_list, linked_condense=False, unlinked_condense=False, print_prefactors=True, color=True)
        result=r'(f^{2}\bh^{\blue{}\red{}}_{\blue{}\red{zy}})'
        assert output==result

        #_creates_fbar_prefactor(omega, h)==False
        rank=2
        term_list=[[fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=2, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=2, n_o=0, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)], [fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0], n_t=[1]), fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=1, n_o=0, m_t=[0], n_t=[1]), (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0),)], [fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=0, m_t=[0, 0], n_t=[1, 1]), fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]), (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0))]]
        output=fcc._make_latex(rank, term_list, linked_condense=False, unlinked_condense=False, print_prefactors=True, color=True)
        result=r'(\bar{f}^{2}\bh^{\blue{}\red{zy}}_{\blue{}\red{}} + \bar{f}\bh^{\blue{}\red{z}}_{\blue{i}\red{}}\bt^{\blue{i}\red{y}}_{} + \frac{1}{2!2!}\bh^{\blue{}\red{}}_{\blue{ij}\red{}}\bt^{\blue{i}\red{z}}_{}\bt^{\blue{j}\red{y}}_{})'
        assert output==result

        #long running line else catch

        #format the term list for [3,3,3,3]
