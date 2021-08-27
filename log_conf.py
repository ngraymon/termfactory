import logging

# -----------------------------------------------------------
# LOGGING PREPERATIONS
# -----------------------------------------------------------
# predefined levels for logging
# CRITICAL 50
# ERROR    40
# WARNING  30
# logging.FLOW = 25
# INFO     20
# DEBUG    10
# NOTSET   0
# -----------------------------------------------------------

# add names
# logging.addLevelName(logging.FLOW, "FLOW")
# "[%(asctime)-13s] [%(levelname)s] %(funcName)s: %(message)s"


class MyLogger(logging.Logger):
    def flow(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.FLOW):
            self._log(logging.FLOW, message, args, **kwargs)


def get_stdout_logger(*args, **kwargs):
    logging.setLoggerClass(MyLogger)

    if kwargs == {}:
        kwargs['format'] = "[%(asctime)-13s] [%(levelname)s] %(funcName)s: %(message)s"
        kwargs['datefmt'] = ["%m/%d/%Y %I:%M:%S %p ", "%d %I:%M:%S "][1]
        kwargs['level'] = [logging.INFO, logging.DEBUG][0]

    logging.basicConfig(**kwargs)
    return logging.getLogger(__name__)


def get_filebased_logger(filename, *args, **kwargs):
    logging.setLoggerClass(MyLogger)

    if kwargs == {}:
        kwargs['format'] = "[%(asctime)-13s] [%(name)s] [%(levelname)-8s] (%(lineno)s): %(funcName)s\n%(message)s"
        kwargs['datefmt'] = ["%m/%d/%Y %I:%M:%S %p ", "%d %I:%M:%S "][1]
        kwargs['level'] = [logging.INFO, logging.DEBUG][0]

    kwargs.update({
        'filename': filename,
        'filemode': 'w',
    })

    print(kwargs)
    logging.basicConfig(**kwargs)
    return logging.getLogger(__name__)


def setLevelCritical(log):
    log.setLevel(logging.CRITICAL)


def setLevelError(log):
    log.setLevel(logging.ERROR)


def setLevelWarning(log):
    log.setLevel(logging.WARNING)


def setLevelFlow(log):
    log.setLevel(logging.FLOW)


def setLevelInfo(log):
    log.setLevel(logging.INFO)


def setLevelDebug(log):
    log.setLevel(logging.DEBUG)
