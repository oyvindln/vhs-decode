import logging
import sys


def init_logging(outfile_name, columns=80):
    """ Sets up a logger with a logfile, status line, and stderr output """

    statusWritten = False

    logger = logging.getLogger("lddecode")
    logger.setLevel(logging.DEBUG)

    class StreamHandlerR(logging.StreamHandler):
        def __init__(self):
            super().__init__()

        def emit(self, record):
            nonlocal statusWritten

            if statusWritten:
                print()
                super().emit(record)
                sys.stdout.flush()
                statusWritten = False
            else:
                super().emit(record)

    logger_stderr = StreamHandlerR()
    logger_stderr.setLevel(logging.INFO)

    logger.addHandler(logger_stderr)

    if outfile_name is not None:
        # Delete old logfile if it exists
        try:
            os.unlink(outfile_name)
        except Exception:
            pass

        logger_file = logging.FileHandler(outfile_name)
        logger_file.setLevel(logging.DEBUG)

        logger_fileformatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger_file.setFormatter(logger_fileformatter)

        logger.addHandler(logger_file)

    def status(line):
        nonlocal statusWritten

        print(line + " " * (columns - len(line)), end="\r")
        logger.debug(line)
        statusWritten = True

    logger.status = status

    return logger


def logging_test():
    import os
    import time
    import tempfile

    fname = os.path.join(tempfile.gettempdir(), "logtest.log")
    print("Writing to " + fname, file=sys.stderr)

    print("Logging test beginning:")
    logger = init_logging(fname)

    logger.debug("logger.debug")
    logger.status("Status line 1 - a bit longer than 2")
    time.sleep(1)
    logger.status("Status line 2")
    logger.info("logger.info")
    logger.status("Status line 3")
    logger.info("end info")

    print("Logfile output:")
    with open(fname, "r") as fp:
        for line in fp.readlines():
            print(line, end="")


if __name__ == "__main__":
    # execute only if run as a script, for testing
    logging_test()
