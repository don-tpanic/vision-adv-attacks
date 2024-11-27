from torchvision import models

def load_model(model_name, device):
    if model_name == "VGG16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    model.eval()
    return model
