import json
from bunch import Bunch
import os

cur_dir = os.path.dirname(__file__)

ADV = {'0': 19636, '10': 27296, '50': 68986, '100': 123279, '500': 566479, '1000': 1125550, '2000': 2245938}


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    base_path = os.path.join(cur_dir, '../experiments')
    config.summary_dir = os.path.join(base_path, config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join(base_path, config.exp_name, "checkpoint/")
    config.ckpt_best = os.path.join(base_path, config.exp_name, 'ckpt_best/')
    config.train_data_len = ADV[str(config.adv)]

    return config
