# vision-adv-attacks

## To work with the repo locally:
```
git clone git@github.com:don-tpanic/vision-adv-attacks.git
```

## Install dependencies:
```
conda env create -f environment.yml
```

## Usage example:
Run ifgsm attack using default configuration on an example image and target class id:
```
python ifgsm.py --config config_1 --img_path data/imagenet_1k_val_white/n03908618/ILSVRC2012_val_00001265.JPEG --target_class_id 283
```
