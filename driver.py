# system imports
import os
import sys
import argparse

# third party imports

# local imports


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

    return parser.parse_args()


if (__name__ == '__main__'):
    import log_conf

    pargs = prepare_parsed_arguments()

    if pargs.stdlog is True:
        log = log_conf.get_stdout_logger()
    else:
        logging_output_filename = str(pargs.log_path)
        log = log_conf.get_filebased_logger(logging_output_filename)

    # dump_all_stdout_to_devnull()   # calling this removes all prints / logs from stdout
    # log.setLevel('CRITICAL')

    import _glue_

    _glue_.main()
