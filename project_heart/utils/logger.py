import logging


class Logger():
    def __init__(self, debug=False, logLevel=None, logName="_", **kwargs):
        self.logger = logging.getLogger(logName)
        if not logLevel is None:
            self.logger.setLevel(logLevel)
        elif debug:
            self.logger.setLevel(logging.DEBUG)

    def dlog(self, msg, *args, **kwargs):
        return self.logger.debug(msg, *args, **kwargs)

    def ilog(self, msg, *args, **kwargs):
        return self.logger.info(msg, *args, **kwargs)

    def wlog(self, msg, *args, **kwargs):
        return self.logger.warning(msg, *args, **kwargs)

    def elog(self, msg, *args, **kwargs):
        return self.logger.error(msg, *args, **kwargs)

    def clog(self, msg, *args, **kwargs):
        return self.logger.critical(msg, *args, **kwargs)
