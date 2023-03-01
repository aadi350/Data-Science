# import logging
# import logging.config

# import module
# import yaml

# # only the FIRST call to logging.basicConfig() actually does anything, since it's meant as a one-off simple configuration facility


# # with open("logging.yaml", "r") as f:
# #     loggingconfig = yaml.load(f, Loader=yaml.FullLoader)

# # logging.config.fileConfig("logging.conf")

# # # default level is WARNING

# # logger = logging.getLogger("main")
# # logger.debug("test debug")


# class FileLoggingHandler(logging.Handler):
#     def __init__(self, filename):
#         logging.Handler.__init__(self)
#         self.filename = filename

#     def emit(self, record):
#         message = self.format(record)
#         with open(self.filename, "a") as f:
#             f.write(message + "\n")


# module.do_something()

# logger = logging.getLogger("my_logger")
# logger.setLevel(logging.DEBUG)

# file_handler = FileLoggingHandler("logfile.log")
# file_handler.setLevel(logging.DEBUG)

# logger.addHandler(file_handler)

# logger.debug("This is a debug message")
# logger.info("This is an info message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")
# logger.critical("This is a critical message")
import logging


def fprint(msg, )

print_fancy("test")

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("My_app")
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

logger.debug("debug message")
logger.info("info message")
logger.warning("warning message")
logger.error("error message")
logger.critical("critical message")
