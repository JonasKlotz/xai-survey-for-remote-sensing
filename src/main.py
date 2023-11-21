import os
import yaml

from src.training.train import train
from src.xai.explain import generate_explanations
from xai.metrics.evaluate_explanation_methods import evaluate_explanation_methods

CONFIGPATH = "/home/jonasklotz/Studys/MASTERS/XAI/config"

def parse_config(config_path):
    # read all files in config_path
    file_names = os.listdir(config_path)
    configs = {}

    for file_name in file_names:
        if file_name.endswith('.yaml') or file_name.endswith('.yml'):
            with open(os.path.join(config_path, file_name)) as file:
                loaded_cfg = yaml.load(file, Loader=yaml.FullLoader)
                configs[file_name[:-11]] = loaded_cfg

    return configs


def main():
    configs = parse_config(CONFIGPATH)
    general_config = configs['general']

    if general_config['training']:
        train(configs['training'])

    if general_config['explanations']:
        generate_explanations(configs['explanations'])
        # for expl in general_config['explanations

    if general_config['evaluations']:
        evaluate_explanation_methods(configs['evaluations'])




if __name__ == '__main__':
    main()