# An implementation of iterative fast gradient sign method for creating targeted adversarial images.

## Description
The original Fast Gradient Sign Method (FGSM), introduced in this [paper](https://arxiv.org/abs/1412.6572), provides an efficient approach to generating adversarial images for vision models. FGSM perturbs the pixel values of an input image in the direction that reduces the likelihood of the image being correctly classified. This direction is determined by the sign of the gradient of the error with respect to the input image.

While FGSM is effective, it does not guarantee that the adversarial image will lead to a specific target misclassification; the misclassified class is not predetermined. This repo implements a variation of FGSM that addresses this limitation by enabling control over the target misclassification. Instead of maximizing the classification error for the original class, this approach minimizes the classification error for the desired target class.

## Example output
![analysis_ILSVRC2012_val_00001049_to_283](https://github.com/user-attachments/assets/96c12e63-dc5c-42ec-82e3-71961c706de6)

## Quickstart

### Work with the repo locally:
```
git clone git@github.com:don-tpanic/vision-adv-attacks.git
```

### Install dependencies:
```
conda env create -f environment.yml
```

### Usage example:
Run ifgsm attack using default configuration on an example image and target class id:
```
python ifgsm.py --config config_1 \
                --img_path data/imagenet_1k_val_white/n03908618/ILSVRC2012_val_00001265.JPEG \
                --target_class_id 283
```
Explanation of parameters:
* `--config`: specifies the configuration file for the attack (e.g., learning rate, number of iterations). See [example](https://github.com/don-tpanic/vision-adv-attacks/blob/main/configs/config_1.yaml).
* `--img_path`: path to the input image on which the attack will be applied. In theory, this approach should work out of the box for any images. In this example, I used images from the ImageNet 1K validation set (hence the particular directory names).
* `--target_class_id`: the class ID you want the model to misclassify the input as (283 corresponds to "Persian cat" in ImageNet).

## Repo structure
* `data/`: directory of source images. For example, for ImageNet 1K vailidation set images, place your images as: `data/imagenet_1k_val_white/<wordnetID>/<fname>.JPEG`

* `configs/`: directory of configuration files. Currently support: `epsilon` (added noise bound), `alpha` (learning rate), `n_iters` (number of iterations of adding noise), `model_name` (the model to be attacked on; currently support torchvision and huggingface ViT models).

* `figs/`: directory of generated figures.
  - `analysis_*.png` are diagnostics. Such as how attack progresses over iterations.
  - `final_*.png` are the perturbed adversarial images.

* `utils/`: directory of utility functions such as preprocessing and plotting.

