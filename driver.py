# system imports
import os
import sys
import argparse

# third party imports

# local imports
from truncation_keys import TruncationsKeys as tkeys
from truncations import save_trunc_to_JSON, load_trunc_from_JSON


def dump_all_stdout_to_devnull():
    sys.stdout = open(os.devnull, 'w')


def prepare_parsed_arguments():
    """ Wrapper for argparser setup """

    # formatclass = argparse.RawDescriptionHelpFormatter
    # formatclass = argparse.RawTextHelpFormatter
    formatclass = argparse.ArgumentDefaultsHelpFormatter  # I liked this the best
    # formatclass = argparse.MetavarTypeHelpFormatter

    # parse the arguments
    parser = argparse.ArgumentParser(description="Code/Latex Generator", formatter_class=formatclass)
    # ----- a args -------
    parser.add_argument('-l', '--log_path', type=str, default='default_logging_file.txt', metavar='/path/filename.txt', help='path to log file')
    parser.add_argument('-s', '--stdlog', action='store_true', help='provide if you want the log to be piped to stdout')
    # parser.add_argument('-q', '--quiet', action='store_true', help='provide if you want to suppress all output/logging')

    # ----- a args -------
    parser.add_argument('-t', type=tuple, default=(1,1,1,1), help="Provided Truncations")
    parser.add_argument('-a', '--ansatz', type=str, default='full fcc', help="Specify Ansatz")
    # parser.add_argument('-gs', '--ground_state', type=bool, default=True, help="Only ground state?")
    parser.add_argument('-es', '--excited_state', type=bool, default=True, help="Only ground state?")
    parser.add_argument('-rf', '--remove_f_terms', type=bool, default=False, help="Choose to remove f terms")

    # ----- a args -------
    # parser.add_argument('--root', type=str, default="./", help="Path to the root directory where file load/save will occur") #change to workdir save tex here too?
    # parser.add_argument('-st', action='store_true', help='enables save file')
    # parser.add_argument('-lt', action='store_true', help='enables load file')
    parser.add_argument('-p', '--path', type=str, default="trunc.json", help="filename of load/save file")

    return parser.parse_args()


def _make_trunc(input_tuple):
    # temp, makes a fcc ENUM
    name = tkeys.key_list_type(input_tuple)

    if name == 'fcc':
        key_list = tkeys.fcc_key_list()
    elif name == 'eTz':
        key_list = tkeys.eTz_key_list()
    elif name == 'zhz':
        key_list = tkeys.zhz_key_list()
    else:
        raise Exception(f'bad keylist name, wtf\n{input_tuple = }')

    trunc = dict([(k, v) for k, v in zip(key_list, input_tuple)])

    return trunc


if (__name__ == '__main__'):
    """ x """
    pargs = prepare_parsed_arguments()

    # process logging parameters
    import log_conf
    if pargs.stdlog is True:
        log = log_conf.get_stdout_logger()
    else:
        logging_output_filename = str(pargs.log_path)
        log = log_conf.get_filebased_logger(logging_output_filename)

    # process execution parameters (truncation and/or keyword args / flags)
    if pargs.t and pargs.path:
        raise Exception('User provided raw truncation values AND path to truncation file. Unclear what they want to do')

    # if the user provides a tuple on the command line
    elif pargs.t:
        trunc = _make_trunc(pargs.t)

    # the user provides a path to a JSON file containing the truncation values
    elif pargs.path:
        assert os.path.isfile(pargs.path), 'invalid path to JSON file'
        trunc = load_trunc_from_JSON(pargs.path)
        # here you load the file
    else:
        raise Exception('user provided no truncation values')

    default_kwargs = {
        'only_ground_state': True,
        'remove_f_terms': False,
        'ansatz': 'eT_z_t ansatz'
    }

    if pargs.es:
        default_kwargs['only_ground_state'] = False

    if pargs.rf:
        default_kwargs['remove_f_terms'] = True

    ansatz_list = ['eT_z_t ansatz', 'z_t ansat', 'full cc']

    if pargs.a:
        assert pargs.a in ansatz_list, f'bad ansatz {pargs.a = }\n should be\n{ansatz_list = }'
        default_kwargs['ansatz'] = pargs.a

    print(pargs)
    # dump_all_stdout_to_devnull()   # calling this removes all prints / logs from stdout
    # log.setLevel('CRITICAL')

    import _glue_
    # load test
    _glue_._generate_latex(trunc, **default_kwargs)
