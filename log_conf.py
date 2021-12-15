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

line_length = 120


class HeaderAdapter(logging.LoggerAdapter):
    """ The purpose is to allow for logging with short messages
    that standout in the resulting logs to delineate larger sections of logs
    """
    def process(self, msg, kwargs):
        ss = (line_length - len(msg)) // 2
        ls2 = '-'*line_length
        spacer = '-'*ss
        return f"\n{ls2}\n{spacer} {msg} {spacer}\n{ls2}\n", kwargs


class SubHeaderAdapter(logging.LoggerAdapter):
    """ The purpose is to allow for logging with short messages
    that standout in the resulting logs to delineate larger sections of logs
    """
    def process(self, msg, kwargs):
        ss = (line_length - len(msg)) // 2
        spacer = '-'*ss
        return f"\n{spacer} {msg} {spacer}\n", kwargs


class MyLogger(logging.Logger):
    def flow(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.FLOW):
            self._log(logging.FLOW, message, args, **kwargs)


def get_stdout_logger(*args, **kwargs):
    logging.setLoggerClass(MyLogger)

    if kwargs == {}:
        kwargs['format'] = "[%(asctime)-13s] [%(name)s] [{levelname:^8}] [{module:^8}] (%(lineno)s): %(funcName)s(): %(message)s"
        kwargs['datefmt'] = ["%m/%d/%Y %I:%M:%S %p ", "%d %I:%M:%S "][1]
        kwargs['level'] = [logging.INFO, logging.DEBUG][0]

    logging.basicConfig(**kwargs)
    return logging.getLogger(__name__)


def get_filebased_logger(filename, *args, **kwargs):
    logging.setLoggerClass(MyLogger)

    if kwargs == {}:

        # old % style
        if False:
            kwargs['format'] = "[%(asctime)-13s] [%(name)s] [%(levelname)-8s] [%(module)-8s] (%(lineno)s): %(funcName)s(): %(message)s"
            kwargs['datefmt'] = ["%m/%d/%Y %I:%M:%S %p ", "%d %I:%M:%S ", "%I:%M:%S "][-1]
            kwargs['level'] = [logging.INFO, logging.DEBUG][0]
        # trying out the f string log formatting
        else:
            # kwargs['format'] = "[{asctime:<13s}] [{name:s}] [{levelname:^10s}] [{module:^8}] ({lineno}): {funcName}():{message}"
            kwargs['format'] = "{module:<12s}: {funcName:<30s}:({lineno:<4d}): {message:s}"
            kwargs['datefmt'] = ["%m-%d-%Y %I:%M:%S %p", "%d %I:%M:%S", "%I:%M:%S"][-1]
            kwargs['level'] = [logging.INFO, logging.DEBUG][0]
            kwargs['style'] = '{'

    kwargs.update({
        'filename': filename,
        'filemode': 'w',
    })

    # apply the configuration parameters
    logging.basicConfig(**kwargs)

    # create and return a logging instance
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
