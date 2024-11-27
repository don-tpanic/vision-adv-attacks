import yaml
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from utils import plotting, models

def load_config(config_version="config_1"):
    config_path = f"configs/{config_version}.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def preprocess_image(img_path, device, imagenet_means, imagenet_stds):
    """
    Preprocess the image for the model.
    Args:
        - img_path: Path to the input image.
    Returns:
        - Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_means, std=imagenet_stds)
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def iterative_fgsm_attack(model, original_img, target_label, epsilon, alpha, n_iters, device, criterion,
                          imagenet_lower_bound, imagenet_upper_bound):
    """
    Perform an Iterative FGSM (I-FGSM) attack.
    Args:
        - model: The neural network model
        - original_img: Input image tensor
        - target_label: Target misclassification label
        - epsilon: Maximum perturbation
        - alpha: Step size for each iteration
        - n_iters: Number of perturbation steps
    Returns:
        - perturbed_img: The adversarial image
        - cumulative_noise: The total perturbation added
        - history: Dictionary containing loss and prediction at each iteration
    """
    perturbed_img = original_img.clone().detach().to(device)
    perturbed_img.requires_grad = True

    # Initialize cumulative perturbation
    cumulative_noise = torch.zeros_like(original_img).to(device)

    # Log metrics
    history = {
        'loss': [],
        'pred_class_id': []
    }

    for i in range(n_iters):
        output = model(perturbed_img)

        # Compute loss for `target` class
        target_tensor = torch.tensor([target_label], dtype=torch.long, device=device)
        loss = criterion(output, target_tensor)

        model.zero_grad()
        loss.backward()

        # Compute perturbation, a small change to
        # the original image, in the direction that
        # minimizes loss for target class
        target_grad = perturbed_img.grad.data
        perturbation = alpha * -target_grad.sign()
        cumulative_noise += perturbation

        # Clamp total perturbation to be bounded
        # and apply it to the original image
        cumulative_noise = torch.clamp(cumulative_noise, -epsilon, epsilon)
        perturbed_img = original_img + cumulative_noise

        # Ensure perturbed image is within ImageNet bounds
        perturbed_img = torch.clamp(
            perturbed_img, imagenet_lower_bound, imagenet_upper_bound)

        # Detach and re-enable gradient computation for the next iteration
        perturbed_img = perturbed_img.detach()
        perturbed_img.requires_grad = True

        # Record loss and prediction
        with torch.no_grad():
            output = model(perturbed_img)
            loss_value = criterion(output, target_tensor).item()
            pred_class_id = output.max(1, keepdim=True)[1].item()
            history['loss'].append(loss_value)
            history['pred_class_id'].append(pred_class_id)

        print(f"Iter {i+1}/{n_iters} | Loss: {loss_value:.4f} | Pred class id: {pred_class_id}")

    return perturbed_img, cumulative_noise, history

def main(args):
    # Load configurations
    config = load_config(args.config)
    img_path = args.img_path
    target_class_id = args.target_class_id
    epsilon = config['epsilon']
    alpha = config['alpha']
    n_iters = config['n_iters']
    imagenet_means = config['imagenet_means']
    imagenet_stds = config['imagenet_stds']
    imagenet_lower_bound = config['imagenet_lower_bound']
    imagenet_upper_bound = config['imagenet_upper_bound']
    model_name = config['model_name']
    fig_dir = config['fig_dir']

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # Load model
    model = models.load_model(model_name, device=device)

    # Load and preprocess the image
    original_img = preprocess_image(img_path, device, imagenet_means, imagenet_stds)

    # Get original model predictions
    with torch.no_grad():
        output = model(original_img)
        original_pred_id = output.max(1, keepdim=True)[1].item()
        original_prob = torch.nn.functional.softmax(output, dim=1)[0, original_pred_id].item()

    # Perform IFGSM
    perturbed_img, noise, history = iterative_fgsm_attack(
        model,
        original_img,
        target_class_id,
        epsilon,
        alpha,
        n_iters,
        device,
        criterion,
        imagenet_lower_bound,
        imagenet_upper_bound
    )

    # Classify the perturbed image
    with torch.no_grad():
        output_perturbed = model(perturbed_img)
        perturbed_pred_id = output_perturbed.max(1, keepdim=True)[1].item()
        perturbed_prob = torch.nn.functional.softmax(output_perturbed, dim=1)[0, perturbed_pred_id].item()

    # Classify the noise (TODO: make sense or not given the range of noise might be OOD)
    with torch.no_grad():
        noise_output = model(noise)
        noise_pred_id = noise_output.max(1, keepdim=True)[1].item()
        noise_prob = torch.nn.functional.softmax(noise_output, dim=1)[0, noise_pred_id].item()
    
    # Visualize the images and save the plot
    plotting.visualize_images(
        original_img, noise, perturbed_img, 
        original_pred_id, noise_pred_id, perturbed_pred_id, 
        [original_prob, noise_prob, perturbed_prob],
        imagenet_means, imagenet_stds,
        img_path, target_class_id,
        fig_dir
    )

    # Plot loss and predicted class over iterations
    plotting.plot_metrics(
        img_path, target_class_id,
        history, 
        fig_dir
    )
    print(f"Original Prediction: {original_pred_id} ({original_prob*100:.2f}%)")
    print(f"Noise Prediction: {noise_pred_id} ({noise_prob*100:.2f}%)")
    print(f"Perturbed Prediction: {perturbed_pred_id} ({perturbed_prob*100:.2f}%)")
    print(f"Target Class ID: {target_class_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative FGSM attack")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config_1", 
        help="Path to the config file"
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="data/imagenet_1k_val_white/n03908618/ILSVRC2012_val_00001265.JPEG",
        help="Path to the input image"
    )
    parser.add_argument(
        "--target_class_id",
        type=int,
        default=283,  # ImageNet class: Persian_cat
        help="Target class ID for the attack"
    )
    args = parser.parse_args()
    main(args)