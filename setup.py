"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
https://docs.python.org/3/distutils/setupscript.html#distutils-installing-scripts
"""
import subprocess
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import sys



# def read_md(filename):
#     """use Pandoc magic to convert Markdown to RST for uploading to PyPi"""
#     here = path.abspath(path.dirname(__file__))
#     with open(path.join(here, filename), encoding='utf-8') as f:
#         try:
#             from pypandoc import convert_text, download_pandoc
#             download_pandoc()
#             return convert_text(f.readlines(), 'rst', format="md")
#         except ImportError:
#             print("warning: pypandoc module not found, could not convert Markdown to RST")
#             return f.read()


# import pypandoc
# pypandoc.download_pandoc()
# long_description_rst = pypandoc.convert_file('README.md', 'rst')
long_description_md = open('README.md').read()

VERSION = (
            1,
            0,
            0
            )


def get_version(version=None):
    """Return version (X.Y[.Z]) from VERSION."""
    parts = 2 if version[2] == 0 else 3
    return '.'.join(str(x) for x in version[:parts])


version = get_version(VERSION)

REQUIRED_PYTHON = (3, 9)

EXCLUDE_FROM_PACKAGES = [
                         'docs',
                         'tests'
                        ]


def setup_package():
    """ wapper for setup() so that we can check if julia is installed first """

    setup_info = dict(
        name='termfactory',
        version=version,
        python_requires='~={}.{}'.format(*REQUIRED_PYTHON),
        url='https://github.com/ngraymon/termfactory',
        author='Neil Raymond',
        author_email='neil.raymond@uwaterloo.ca',
        description='A python package for generating LaTeX and python code for evaluating residual terms',
        long_description=long_description_md,
        long_description_content_type='text/markdown',
        license='MIT',
        packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
        include_package_data=True,
        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        classifiers=[
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Chemistry',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.9'
        ],
        keywords='coupled_cluster quantum_mechanics chemistry vibronic',
        # https://packaging.python.org/en/latest/requirements.html
        install_requires=[
            'numpy>=1.18.*',
        ],
        extras_require={
            #'dev': ['check-manifest'],
            'test': ['coverage', 'pytest']
        },
    )

    setup(**setup_info)


if __name__ == '__main__':
    setup_package()
