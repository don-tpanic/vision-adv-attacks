import pandas as pd

def decode_class_id_to_description(class_id):
    """
    Based on `imagenet_1k_labels.csv`, find the 
    corresponding `description` for a given `idx`.
    """
    labels = pd.read_csv("imagenet_1k_labels.csv")
    return labels.loc[labels["idx"] == class_id, "description"].values[0]