import torch
from torchvision import models
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, ViTForImageClassification

def load_model(model_name, device):
    if model_name == "VGG16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    elif model_name == "google/vit-base-patch16-224":
        model = ViTForImageClassification.from_pretrained(model_name).to(device)
    else:
        raise ValueError("Model not supported")
    model.eval()
    return model

def preprocess_image(model_name, img_path, device, imagenet_means, imagenet_stds):
    if model_name == "VGG16":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_means, std=imagenet_stds)
        ])
        img = Image.open(img_path).convert("RGB")
        return transform(img).unsqueeze(0).to(device)

    elif model_name == "google/vit-base-patch16-224":
        preprocesser = AutoImageProcessor.from_pretrained(model_name)
        img = Image.open(img_path)
        return preprocesser(img, return_tensors="pt")['pixel_values'].to(device)

    else:
        raise ValueError("Model not supported")
    
def get_model_prediction(img, model):
    output = model(img)
    if hasattr(output, "logits"):
        output = output.logits
    pred_id = output.max(1, keepdim=True)[1].item()
    prob = torch.nn.functional.softmax(output, dim=1)[0, pred_id].item()
    return pred_id, prob