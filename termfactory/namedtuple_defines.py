""" NAMED TUPLES DEFINITIONS """

# system imports
from collections import namedtuple

# third party imports

# local imports


# ----------------------------------------------------------------------------------------------- #
# the building blocks for the h & w components of each residual term are stored in named tuples
# ----------------------------------------------------------------------------------------------- #

# tuple for a general operator as described on page 1, eq 1
general_operator_namedtuple = namedtuple('operator', ['name', 'rank', 'm', 'n'])

# connected_namedtuple = namedtuple('connected', ['name', 'm', 'n'])
# linked_namedtuple = namedtuple('linked', ['name', 'm', 'n'])
# unlinked_namedtuple = namedtuple('unlinked', ['name', 'm', 'n'])

omega_namedtuple = namedtuple('Omega', ['maximum_rank', 'operator_list'])
hamiltonian_namedtuple = namedtuple('hamiltonian', ['maximum_rank', 'operator_list'])

"""rather than just lists and dictionaries using namedtuples makes the code much more concise
we can write things like `h.max_i` instead of `h[0]` and the label of the member explicitly
describes what value it contains making the code more readable and user friendly """
h_namedtuple = namedtuple('h_term', ['max_i', 'max_k'])
w_namedtuple = namedtuple('w_term', ['max_i', 'max_k', 'order'])

# ----------------------------------------------------------------------------------------------- #
# --------------------------------------  CC OPERATORS  ----------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


# namedtuples for the t amplitudes
connected_namedtuple = namedtuple('connected', ['m_h', 'n_h', 'm_o', 'n_o'])
disconnected_namedtuple = namedtuple('disconnected', ['m_h', 'n_h', 'm_o', 'n_o'])


# ----------------------------------------------------------------------------------------------- #
# --------------------------------------  W OPERATORS  ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


# used in building the code W operators
t_term_namedtuple = namedtuple('t_term_namedtuple', ['string', 'order', 'shape'])

# building the latex W operators
w_namedtuple_latex = namedtuple('w_latex', ['m', 'n'])
t_namedtuple_latex = namedtuple('t_namedtuple_latex', ['m', 'n'])
