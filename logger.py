
import logging
import datetime
import config
import os

def set_logger(args, log_path='./logs'):
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger_name = f"{args.dataset}_{args.model}_{args.batch}_{time}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(log_path, logger_name + '.log')
    fh = logging.FileHandler(log_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def log_args(logger, args):
    args = vars(args)
    logger.info("=" * 30 + " args " + "=" * 30)
    for k in args.keys():
        logger.info(f"{k}: {args[k]}")
    logger.info("=" * 30 + " End args " + "=" * 30)


def log_config(logger):
    logger.info("=" * 30 + " config " + "=" * 30)
    # Get the names of global variables in the config module
    global_var_names = [var for var in dir(config) if not var.startswith('_')]

    # Get the values of the global variables
    global_var_values = {var: getattr(config, var) for var in global_var_names}

    for name, value in global_var_values.items():
        logger.info(f"{name}: {value}")
    logger.info("=" * 30 + " End config " + "=" * 30)


def log_testResult(logger, test_result_dict):
    logger.info('--------------------- test results-------------------------------')
    logger.info('acc:' + str(test_result_dict['acc']) + '  prec:' + str(test_result_dict['prec']) +
                '  rec:' + str(test_result_dict['rec']) + '  f1:' + str(test_result_dict['f1']))
