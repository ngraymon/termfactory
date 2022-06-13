from os import pardir
from os.path import abspath, join, dirname
import sys

# import the path to the pibronic package
my_directory = dirname(__file__)
sys.path.insert(0, abspath(join(my_directory, pardir, pardir, 'termfactory')))
