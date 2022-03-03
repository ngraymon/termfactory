from logging import root
from . import context
import latex_full_cc as fcc
import code_residual_equations as cre
import latex_eT_zhz as et
import latex_zhz as zhz

class write_cc_einsum_python_from_list_single_unique_key:

    t_term_list = [
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0], n_t=[1]),
            fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]),
            (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=0, m_t=[0, 0], n_t=[1, 1]),
            fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 0]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=0, m_t=[0, 0], n_t=[1, 1]),
            fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0, 0], n_t=[1, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=1, n_o=0, m_t=[0, 0], n_t=[0, 1]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
            )
        ]
    ]

    output = [
        'if truncation.singles:',
        '    R += 1/(factorial(2)) * (',
        "        np.einsum('ac, cdz, dby -> abzy', h_args[(0, 0)], t_args[(1, 0)], t_args[(1, 0)]) +",
        "        np.einsum('ac, cdy, dbz -> abzy', h_args[(0, 0)], t_args[(1, 0)], t_args[(1, 0)])",
        '    )',
        '',
        'if truncation.at_least_linear:',
        '    if truncation.singles:',
        '        R += (',
        "            np.einsum('acz, cby -> abzy', h_args[(1, 0)], t_args[(1, 0)]) +",
        "            np.einsum('acz, cby -> abzy', h_args[(1, 0)], t_args[(1, 0)])",
        '        )',
        '        R += 1/(factorial(2)) * (',
        "            np.einsum('aciz, cdy, dbi -> abzy', h_args[(1, 1)], t_args[(1, 0)], t_args[(1, 0)]) +",
        "            np.einsum('aciz, cdi, dby -> abzy', h_args[(1, 1)], t_args[(1, 0)], t_args[(1, 0)])",
        '        )',
        '    if truncation.doubles:',
        '        R += 1/(factorial(2)) * (',
        "            np.einsum('aci, cdz, dbiy -> abzy', h_args[(0, 1)], t_args[(1, 0)], t_args[(2, 0)]) +",
        "            np.einsum('aci, cdiy, dbz -> abzy', h_args[(0, 1)], t_args[(2, 0)], t_args[(1, 0)])",
        '        )',
        '',
        'if truncation.at_least_quadratic:'
    ]


class generate_residual_data:

    expected_result = (
        [
            ['h_0', 'h_{k_1} * w^{k_1}'],
            ['h_0 * w^{i_1}', 'h_{k_1} * w^{i_1,k_1}', 'h^{i_1}']
        ],
        [
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
                )
            ],
            [
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
        ]
    )


class write_residual_function_string_high_order_h_and_w:

    res_list = [
        cre.residual_term(
            prefactor='(1/2)',
            h=cre.h_namedtuple(max_i=0, max_k=0),
            w=cre.w_namedtuple(max_i=2, max_k=0, order=2)
        ),
        cre.residual_term(
            prefactor='(3/6)',
            h=cre.h_namedtuple(max_i=0, max_k=1),
            w=cre.w_namedtuple(max_i=2, max_k=1, order=3)
        ),
        cre.residual_term(
            prefactor='(6/24)',
            h=cre.h_namedtuple(max_i=0, max_k=2),
            w=cre.w_namedtuple(max_i=2, max_k=2, order=4)
        ),
        cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=1, max_k=0),
            w=cre.w_namedtuple(max_i=1, max_k=0, order=1)
        ),
        cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=1, max_k=1),
            w=cre.w_namedtuple(max_i=1, max_k=1, order=2)
        ),
        cre.residual_term(
            prefactor='(1/2)',
            h=cre.h_namedtuple(max_i=2, max_k=0),
            w=cre.w_namedtuple(max_i=0, max_k=0, order=0)
        )
    ]


class generate_python_code_for_residual_functions:

    t_list = [
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
            )
        ],
        [
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
    ]


class generate_eT_taylor_expansion:

    expansion = [
        et.general_operator_namedtuple(name='1', rank=0, m=0, n=0),
        [
            et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
            et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)
        ],
        [
            [
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)
            ],
            [
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)
            ],
            [
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)
            ],
            [
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)
            ]
        ],
        [
            [
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)
            ],
            [
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)
            ],
            [
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)
            ],
            [
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)
            ],
            [
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)
            ],
            [
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)
            ],
            [
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)
            ],
            [
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)
            ]
        ]
    ]


class prepare_condensed_terms:

    large_term_list = [
        [
            fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1], n_t=[2]),
            fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]),
            (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1), ),
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


class make_latex:

    long_catch_term_list = [  # file flag
        [
            fcc.connected_omega_operator_namedtuple(
                rank=2, m=1, n=1, m_h=1, n_h=1, m_t=[0], n_t=[0]
            ),
            fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=1, n_o=1, m_t=[0], n_t=[0]),
            (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),),
        ],
        [
            fcc.connected_omega_operator_namedtuple(
                rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0], n_t=[1]
            ),
            fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=1, m_t=[0], n_t=[1]),
            (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0),),
        ],
        [
            fcc.connected_omega_operator_namedtuple(
                rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[1], n_t=[0]
            ),
            fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=1, n_o=0, m_t=[0], n_t=[1]),
            (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1),),
        ],
        [
            fcc.connected_omega_operator_namedtuple(
                rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0], n_t=[1]
            ),
            fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=1, m_t=[1], n_t=[0]),
            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=1, n_o=0),),
        ],
        [
            fcc.connected_omega_operator_namedtuple(
                rank=2, m=1, n=1, m_h=1, n_h=1, m_t=[0], n_t=[0]
            ),
            fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=1, n_o=1, m_t=[0], n_t=[1]),
            (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),),
        ],
        [
            fcc.connected_omega_operator_namedtuple(
                rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[1], n_t=[0]
            ),
            fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=1, n_o=0, m_t=[1], n_t=[0]),
            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1),),
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=1, n_h=1, m_t=[0], n_t=[0]),
            fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=1, n_o=1, m_t=[1], n_t=[0]),
            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),),
        ],
        [
            fcc.connected_omega_operator_namedtuple(
                rank=2, m=1, n=1, m_h=0, n_h=0, m_t=[1, 0], n_t=[0, 1]
            ),
            fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]),
            (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)),
        ],
        [
            fcc.connected_omega_operator_namedtuple(
                rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0, 0], n_t=[0, 1]
            ),
            fcc.connected_h_operator_namedtuple(rank=3, m=0, n=3, m_o=0, n_o=1, m_t=[0, 0], n_t=[1, 1]),
            (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)),
        ],
        [
            fcc.connected_omega_operator_namedtuple(
                rank=2, m=1, n=1, m_h=0, n_h=0, m_t=[1, 0], n_t=[0, 1]
            ),
            fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=0, m_t=[1, 0], n_t=[0, 1]),
            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)),
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=0, m_t=[0, 1], n_t=[1, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=0, m_t=[1, 0], n_t=[0, 1]),
            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=1, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1)),
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0, 0], n_t=[1, 0]),
            fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=0, n_o=1, m_t=[1, 0], n_t=[0, 1]),
            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=1, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)),
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[0, 1], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=1, n_o=0, m_t=[0, 0], n_t=[1, 1]),
            (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1)),
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0, 0], n_t=[0, 1]),
            fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=0, n_o=1, m_t=[1, 0], n_t=[0, 1]),
            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)),
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=0, m_t=[1, 0], n_t=[0, 1]),
            fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[1, 1], n_t=[0, 0]),
            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=0, n_h=1, m_o=1, n_o=0)),
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[1, 0], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=1, n_o=0, m_t=[1, 0], n_t=[0, 1]),
            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)),
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[0, 1], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=1, n_o=0, m_t=[1, 0], n_t=[0, 1]),
            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1)),
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0, 0], n_t=[0, 1]),
            fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=0, n_o=1, m_t=[1, 1], n_t=[0, 0]),
            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=0, n_h=1, m_o=1, n_o=0)),
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[0, 1], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=3, m=3, n=0, m_o=1, n_o=0, m_t=[1, 1], n_t=[0, 0]),
            (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1)),
        ],
    ]


class write_cc_latex_from_lists_high_rank_case:

    fully = [
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=2, n_h=0, m_t=[0], n_t=[0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=2, m_t=[0], n_t=[0]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=1, n_h=0, m_t=[1], n_t=[0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=1, m_t=[0], n_t=[1]),
            (
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1),
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=1, n_h=0, m_t=[1], n_t=[0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=1, m_t=[1], n_t=[0]),
            (
                fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1),
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]),
            (
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=0, m_t=[1, 0], n_t=[0, 1]),
            (
                fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[1, 1], n_t=[0, 0]),
            (
                fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1),
                fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1)
            )
        ]
    ]

    linked = [
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=1, n_h=0, m_t=[1], n_t=[0]),
            fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=1, m_t=[0], n_t=[0]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1),
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 0]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1),
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=1, n_h=0, m_t=[1, 0], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=1, m_t=[0, 0], n_t=[0, 1]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 0]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1),
                fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=1, n_h=0, m_t=[1, 0], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=1, m_t=[0, 1], n_t=[0, 0]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1),
                fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0)
            )
        ]
    ]

    unlinked = [
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2], n_t=[0]),
            fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 2]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),
                fcc.connected_namedtuple(m_h=2, n_h=0, m_o=0, n_o=0)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 0]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),
                fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 1]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),
                fcc.connected_namedtuple(m_h=1, n_h=1, m_o=0, n_o=0)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[0, 2], n_t=[0, 0]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),
                fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0)
            )
        ]
    ]


class test_long_line_splitting:

    terms = [
        [
            et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=0, m_r=0, n_r=1),
            (et.disconnected_t_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=0, m_r=0, n_r=0),),
            et.connected_eT_h_z_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_t=[0], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=0),
            (None, et.disconnected_eT_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=1, n_lhs=0, m_t=(0,), n_t=(0,), m_h=0, n_h=0, m_l=0, n_l=0)),
            1
        ],
        [
            et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=0, m_r=0, n_r=1),
            (et.disconnected_t_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=0, m_r=0, n_r=0),),
            et.connected_eT_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_t=[0], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=1),
            (None, et.connected_eT_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=1, n_lhs=0, m_t=(0,), n_t=(0,), m_h=1, n_h=0, m_l=0, n_l=0)),
            1
        ],
        [
            et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=1, m_r=0, n_r=0),
            (et.disconnected_t_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=0, m_r=0, n_r=0),),
            et.connected_eT_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=1, n_lhs=0, m_t=[0], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=0),
            (None, et.disconnected_eT_z_right_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_t=(0,), n_t=(0,), m_h=0, n_h=0, m_l=0, n_l=0)),
            1
        ],
        [
            et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=1, m_r=0, n_r=0),
            (et.disconnected_t_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=0, m_r=0, n_r=0),),
            et.connected_eT_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=1, n_lhs=0, m_t=[0], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=1),
            (None, et.connected_eT_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_t=(0,), n_t=(0,), m_h=1, n_h=0, m_l=0, n_l=0)),
            1
        ],
        [
            et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=0, m_r=0, n_r=1),
            (et.disconnected_t_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=0, m_r=0, n_r=1),),
            et.connected_eT_h_z_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_t=[0], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=0),
            (None, et.disconnected_eT_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=1, n_lhs=0, m_t=(1,), n_t=(0,), m_h=0, n_h=0, m_l=0, n_l=0)),
            1
        ],
        [
            et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=1, m_r=0, n_r=0),
            (et.disconnected_t_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=0, m_r=0, n_r=1),),
            et.connected_eT_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=1, n_lhs=0, m_t=[0], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=0),
            (None, et.disconnected_eT_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_t=(1,), n_t=(0,), m_h=0, n_h=0, m_l=0, n_l=0)),
            1
        ],
        [
            et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=0, m_r=0, n_r=1),
            (et.connected_t_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=1, m_r=0, n_r=0),),
            et.connected_eT_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_t=[1], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=0),
            (None, et.disconnected_eT_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=1, n_lhs=0, m_t=(0,), n_t=(0,), m_h=0, n_h=0, m_l=0, n_l=0)),
            1
        ],
        [
            et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=1, m_r=0, n_r=0),
            (et.disconnected_t_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=0, m_r=0, n_r=1),),
            et.connected_eT_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=1, n_lhs=0, m_t=[0], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=1),
            (None, et.connected_eT_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_t=(1,), n_t=(0,), m_h=1, n_h=0, m_l=0, n_l=0)),
            1
        ],
        [
            et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=0, m_r=0, n_r=1),
            (et.connected_t_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=1, m_r=0, n_r=0),),
            et.connected_eT_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_t=[1], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=1),
            (None, et.connected_eT_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=1, n_lhs=0, m_t=(0,), n_t=(0,), m_h=1, n_h=0, m_l=0, n_l=0)),
            1
        ],
        [
            et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=1, m_r=0, n_r=0),
            (et.connected_t_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=1, m_r=0, n_r=0),),
            et.connected_eT_h_z_operator_namedtuple(rank=2, m=2, n=0, m_lhs=1, n_lhs=0, m_t=[1], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=0),
            (None, et.disconnected_eT_z_right_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_t=(0,), n_t=(0,), m_h=0, n_h=0, m_l=0, n_l=0)),
            1
        ]
    ]


class _prepare_fourth_z_latex:
    zero_lhs_op_nt= zhz.connected_lhs_operator_namedtuple(rank=0, m=0, n=0, m_l=0, n_l=0, m_h=0, n_h=0, m_r=0, n_r=0)
    zero_connected_h_z_op_nt = zhz.connected_h_z_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=0)
    zero_disconnected_z_r_nt = zhz.disconnected_z_right_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=0)
    zero_disconnected_z_l_nt = zhz.disconnected_z_left_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=0, n_r=0)
    term_list_short = [
        [
            zero_lhs_op_nt,
            zero_connected_h_z_op_nt,
            [
                zero_disconnected_z_l_nt,
                zero_disconnected_z_r_nt
            ]
        ],
        [
            zero_lhs_op_nt,
            zero_connected_h_z_op_nt,
            [
                zhz.disconnected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=0, n_r=1),
                zhz.disconnected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=1, n_l=0)
            ]
        ],
        [
            zero_lhs_op_nt,
            zero_connected_h_z_op_nt,
            [
                zhz.disconnected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=1, n_r=0),
                zhz.disconnected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=1)
            ]
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=1),
            [
                zero_disconnected_z_l_nt,
                zhz.connected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=0)
            ]
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=0),
                zero_disconnected_z_r_nt
            ]
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=1, n_r=0),
            [
                zero_disconnected_z_l_nt,
                zhz.connected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=0)
            ]
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=0),
                zero_disconnected_z_r_nt
            ]
        ]
    ]
    term_list_long = [
        [
            zero_lhs_op_nt,
            zero_connected_h_z_op_nt,
            [
                zero_disconnected_z_l_nt,
                zero_disconnected_z_r_nt,
            ],
        ],
        [
            zero_lhs_op_nt,
            zero_connected_h_z_op_nt,
            [
                zhz.disconnected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=0, n_r=1),
                zhz.disconnected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=1, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zero_connected_h_z_op_nt,
            [
                zhz.disconnected_z_left_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=0, n_r=2),
                zhz.disconnected_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=2, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zero_connected_h_z_op_nt,
            [
                zhz.disconnected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=1, n_r=0),
                zhz.disconnected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=1),
            ],
        ],
        [
            zero_lhs_op_nt,
            zero_connected_h_z_op_nt,
            [
                zhz.disconnected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=1, n_r=1),
                zhz.disconnected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=1, n_l=1),
            ],
        ],
        [
            zero_lhs_op_nt,
            zero_connected_h_z_op_nt,
            [
                zhz.disconnected_z_left_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=2, n_r=0),
                zhz.disconnected_z_right_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=2),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=1),
            [
                zero_disconnected_z_l_nt,
                zhz.connected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=1),
            [
                zhz.disconnected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=0, n_r=1),
                zhz.connected_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=1, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=0),
                zero_disconnected_z_r_nt,
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=1),
            [
                zhz.disconnected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=1, n_r=0),
                zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=1),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=1),
                zhz.disconnected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=1, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=1, n_r=0),
                zhz.disconnected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=1),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=2),
            [
                zero_disconnected_z_l_nt,
                zhz.connected_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=2, n_h=0, m_l=0, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=1),
            [
                zhz.connected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=0),
                zhz.connected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=1),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=1),
                zhz.connected_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=1, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_l=0, n_l=2, m_r=0, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=2, n_h=0, m_r=0, n_r=0),
                zero_disconnected_z_r_nt,
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=1),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=1, n_r=0),
                zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=1),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=1, n_r=0),
            [
                zero_disconnected_z_l_nt,
                zhz.connected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=0),
                zero_disconnected_z_r_nt,
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=1, n_r=0),
            [
                zhz.disconnected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=0, n_r=1),
                zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=1, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=1),
                zhz.disconnected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=1, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=1, n_r=0),
            [
                zhz.disconnected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=1, n_r=0),
                zhz.connected_z_right_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=1),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=1, n_r=0),
                zhz.disconnected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=1),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=1, n_r=1),
            [
                zero_disconnected_z_l_nt,
                zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=1, m_l=0, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=1),
            [
                zhz.connected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=0),
                zhz.connected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=1),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=1),
                zhz.connected_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=1, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=1, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=0),
                zhz.connected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=1, n_l=1, m_r=0, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=1, m_r=0, n_r=0),
                zero_disconnected_z_r_nt,
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=1),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=1, n_r=0),
                zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=1),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=1, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=1),
                zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=1, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=1, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=1, n_r=0),
                zhz.connected_z_right_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=1),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=2, n_r=0),
            [
                zero_disconnected_z_l_nt,
                zhz.connected_z_right_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=2, m_l=0, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=1, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=0),
                zhz.connected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_l=2, n_l=0, m_r=0, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=2, m_r=0, n_r=0),
                zero_disconnected_z_r_nt,
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=1, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=1),
                zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=1, n_l=0),
            ],
        ],
        [
            zero_lhs_op_nt,
            zhz.connected_h_z_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=1, n_r=0),
            [
                zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=1, n_r=0),
                zhz.connected_z_right_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=1),
            ],
        ],
    ]
