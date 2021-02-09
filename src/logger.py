import logging.config


def set_logger():
    """
    Initialize the logger according to the logger configuration file
    :return:
    """
    logging.config.fileConfig('configuration/logs.conf',
                              defaults={'logfilename': 'logs/logs.out'},
                              disable_existing_loggers=False)

