# PairGAN

Code for [The Benefits of Pairwise Discriminators for Adversarial Training](https://arxiv.org/abs/2002.08621)

## Prerequisites
- preprocess_cat_dataset.py requires opencv
- Running Models requires Pytorch (written with v1.4)
- Calculating FID requires Tensorflow (written with v2.0)

## Preprocess CAT dataset
This part requires setting_up_script.sh and preprocess_cat_dataset.py. Adopted from https://github.com/AlexiaJM/relativistic-f-divergences

Detailed explanations can be found in these files

## Run the code
**Sample Commands**
- for PairGAN:

```bash
python PairGAN.py --dataset cat --dataroot "datapath" --cuda --outf "outpath" --manualSeed 1 --niter 1500 --imageSize 64 --save_freq_epoch 15
```

- for PacPairGAN2:

This version is to address the severe mode collapse problem in CAT for 256x256 (there are only about 2k images).
```bash
python PacPairGAN2.py --dataset cat --dataroot "datapath" --cuda --outf "outpath" --manualSeed 1 --niter 3500 --imageSize 256 --save_freq_epoch 35
```

- for standard GAN:

```bash
python GAN.py --dataset cat --dataroot "datapath" --cuda --outf "outpath" --manualSeed 1 --niter 1500 --imageSize 64 --save_freq_epoch 15
```

- for Pac standard GAN 2:

```bash
python PacGAN2.py --dataset cat --dataroot "datapath" --cuda --outf "outpath" --manualSeed 1 --niter 3500 --imageSize 256 --save_freq_epoch 35
```

## Calculate FID score
Adopted from https://github.com/bioinf-jku/TTUR

The code to pre-calculate the FID stats for a given dataset is precalc_stats.py, which is run automatically for the CAT dataset in setting_up_script.sh

To calculate FID, use calculate_fid.sh, which will generate images (50k by default) first, and then calculate FID based on these images.

## Citation
Please cite our work if you find it useful in your research:
```bibtex
@article{tong2020benefits,
  title={The Benefits of Pairwise Discriminators for Adversarial Training},
  author={Shangyuan Tong and Timur Garipov and Tommi Jaakkola},
  journal={arXiv preprint arXiv:2002.08621},
  year={2020}
}
```
