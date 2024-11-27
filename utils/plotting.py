import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(history, save_dir="figs"):
    """
    Plot the loss and predicted class over iterations.

    Args:
        - history: Dictionary containing 'loss' and 'pred_class_id' lists.
        - save_dir: Directory to save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_iters = len(history['loss'])

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(n_iters), history['loss'], marker='o')
    plt.title("Loss Over Iterations")
    plt.xlabel("Num. Iterations")
    plt.ylabel("CE")
    plt.grid(True)

    # Plot Predicted class
    plt.subplot(1, 2, 2)
    plt.scatter(range(n_iters), history['pred_class_id'], marker='*')
    plt.title("Predicted class over iterations")
    plt.xlabel("Num. Iterations")
    plt.ylabel("Class ID")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attack_metrics.png"))

def visualize_images(
        original_img, noise_img, perturbed_img, 
        original_label, noise_label, perturbed_label, 
        probs,
        save_dir
    ):
    os.makedirs(save_dir, exist_ok=True)

    # Convert tensors to numpy arrays
    original_img = original_img.cpu().detach().numpy().squeeze()
    noise_img = noise_img.cpu().detach().numpy().squeeze()
    perturbed_img = perturbed_img.cpu().detach().numpy().squeeze()

    # De-normalize original and perturbed images (channel-wise)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    original_img = std * original_img + mean
    perturbed_img = std * perturbed_img + mean

    # Clip values to [0, 1] for valid RGB range
    original_img = np.clip(original_img, 0, 1)
    perturbed_img = np.clip(perturbed_img, 0, 1)

    # Normalize noise to [0, 1] for visualization
    noise_img = np.clip(noise_img, -1, 1)
    noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min())

    # Transpose from (C, H, W) to (H, W, C) for plotting
    original_img = original_img.transpose(1, 2, 0)
    noise_img = noise_img.transpose(1, 2, 0)
    perturbed_img = perturbed_img.transpose(1, 2, 0)

    # Plot the images
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    
    ax[0].imshow(original_img)
    ax[0].set_title(f"Original\nClass: {original_label}\nProb: {probs[0]:.2f}")
    ax[0].axis("off")
    
    ax[1].imshow(noise_img)
    ax[1].set_title(f"Noise\nClass: {noise_label}\nProb: {probs[1]:.2f}")
    ax[1].axis("off")
    
    ax[2].imshow(perturbed_img)
    ax[2].set_title(f"Perturbed\nClass: {perturbed_label}\nProb: {probs[2]:.2f}")
    ax[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attack_visualization.png"))
    plt.close()

    # Save the final pertubed image individually
    plt.imshow(perturbed_img)
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, "perturbed_image.png"))
