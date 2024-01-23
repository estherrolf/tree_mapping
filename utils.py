# imports
import yaml

def get_project_dir():
    with open('project_dir.yaml', 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    return cfg['project_dir']
