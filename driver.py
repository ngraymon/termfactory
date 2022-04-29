# system imports
import os
import sys
import argparse

# third party imports

# local imports
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
    parser.add_argument('-l', '--log_path', type=str, default='default_logging_file.txt', metavar='/path/filename.txt', help='path to log file')
    parser.add_argument('-s', '--stdlog', action='store_true', help='provide if you want the log to be piped to stdout')
    # parser.add_argument('-q', '--quiet', action='store_true', help='provide if you want to suppress all output/logging')
    parser.add_argument('-t', type=tuple, default=(1,1,1,1), help="Provided Truncations")
    parser.add_argument('-a', '--ansatz', type=str, default='full fcc', help="Specify Ansatz")
    parser.add_argument('-ogs', '--only_ground_state', type=bool, default=True, help="Only ground state?")
    parser.add_argument('-rf', '--remove_f_terms', type=bool, default=False, help="Choose to remove f terms")
    parser.add_argument('-path', type=str, default="./", help="Path to the root directory where file load/save will occur") #change to workdir save tex here too?
    parser.add_argument('-st', action='store_true', help='enables save file')
    parser.add_argument('-lt', action='store_true', help='enables load file')
    parser.add_argument('-fn', '--filename', type=str, default="trunc.json", help="filename of load/save file")

    return parser.parse_args()


if (__name__ == '__main__'):
    import log_conf

    pargs = prepare_parsed_arguments()

    if pargs.stdlog is True:
        log = log_conf.get_stdout_logger()
    else:
        logging_output_filename = str(pargs.log_path)
        log = log_conf.get_filebased_logger(logging_output_filename)

    print(pargs)
    # dump_all_stdout_to_devnull()   # calling this removes all prints / logs from stdout
    # log.setLevel('CRITICAL')

    import _glue_

    # need to make class meathod for mapping trunc to enum, ask neil desired input meathods
    default_kwargs = {
        'only_ground_state': True,
        'ansatz': 'eT_z_t ansatz' 
    }
    # load test
    if pargs.lt is True:
        trunc=load_trunc_from_JSON(pargs.path+pargs.filename)
        _glue_._generate_latex(trunc, **default_kwargs)
    else:
        _glue_._generate_latex(_glue_._make_trunc(pargs.t, **default_kwargs)) # TODO map rest of pargs
