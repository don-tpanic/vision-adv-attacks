import yaml
import pandas as pd

def load_config(config_version="config_1"):
    """
    Load the configuration file.
    """
    config_path = f"configs/{config_version}.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def decode_class_id_to_description(class_id):
    """
    Based on `imagenet_1k_labels.csv`, find the 
    corresponding `description` for a given `idx`.
    """
    labels = pd.read_csv("imagenet_1k_labels.csv")
    return labels.loc[labels["idx"] == class_id, "description"].values[0]