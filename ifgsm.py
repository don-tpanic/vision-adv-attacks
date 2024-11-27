import yaml
import argparse
import torch
import torch.nn as nn
from torchvision import  models
from utils import plotting, models, data

def iterative_fgsm_attack(model, 
                          original_img, target_label, epsilon, alpha, 
                          n_iters, device, criterion,
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
        - device: Device to run the model on
        - criterion: Loss function
        - imagenet_lower_bound: Lower bound for ImageNet pixel values
        - imagenet_upper_bound: Upper bound for ImageNet pixel values

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
        if hasattr(output, "logits"):  # for huggingface models
            output = output.logits

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
            if hasattr(output, "logits"):  # for huggingface models
                output = output.logits
            loss_value = criterion(output, target_tensor).item()
            pred_class_id = output.max(1, keepdim=True)[1].item()
            history['loss'].append(loss_value)
            history['pred_class_id'].append(pred_class_id)

        print(f"Iter {i+1}/{n_iters} | Loss: {loss_value:.4f} | "\
              f"Pred class id: {pred_class_id} | Pred class: {data.decode_class_id_to_description(pred_class_id)}")

    return perturbed_img, cumulative_noise, history

def main(args):
    # Load configurations
    config = data.load_config(args.config)
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

    # Load model and preprocess image
    model = models.load_model(model_name, device=device)
    original_img = models.preprocess_image(
        model_name, img_path, device, imagenet_means, imagenet_stds)
    
    # Get original model predictions
    original_pred_id, original_prob = models.get_model_prediction(original_img, model)

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
    perturbed_pred_id, perturbed_prob = models.get_model_prediction(perturbed_img, model)

    # Classify the noise (TODO: make sense or not given the range of noise might be OOD)
    noise_pred_id, noise_prob = models.get_model_prediction(noise, model)
    
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
    print(f"Target Class: {data.decode_class_id_to_description(target_class_id)}")

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