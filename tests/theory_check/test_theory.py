"""
This test would be for the sole purpose of generated `*.tex` output files that are compared to
files already present in the `latex` directory which have been verified as being theoretically correct.

This way any research can go and look at those `pdf` files and convince themselves that the generated terms are correct.
They can manually compile the `.tex` files inside the `latex` folder and the pytest makes sure that the generated `.tex` output
exactly matches what is already present in the `latex` folder.


"""


# system imports
from os.path import abspath, dirname, join
theory_dir = dirname(abspath(__file__))
root_dir = join(theory_dir, 'latex')

# local imports
from . import context


# put code here

