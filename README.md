# DeepDeblur-PyTorch

This is a pytorch implementation of our research. Please refer to our CVPR 2017 paper for details:

Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring
[[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)]
[[supplementary](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Nah_Deep_Multi-Scale_Convolutional_2017_CVPR_supplemental.zip)]
[[slide](https://drive.google.com/file/d/1sj7l2tGgJR-8wTyauvnSDGpiokjOzX_C/view?usp=sharing)]

If you find our work useful in your research or publication, please cite our work:
```
@InProceedings{Nah_2017_CVPR,
  author = {Nah, Seungjun and Kim, Tae Hyun and Lee, Kyoung Mu},
  title = {Deep Multi-Scale Convolutional Neural Network for Dynamic Scene Deblurring},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {July},
  year = {2017}
}
```

Original Torch7 implementaion is available [here](https://github.com/SeungjunNah/DeepDeblur_release).

## Dependencies

* python 3 (tested with anaconda3)
* PyTorch 1.6
* tqdm
* imageio
* scikit-image
* numpy
* matplotlib
* readline

Please refer to [this issue](https://github.com/SeungjunNah/DeepDeblur-PyTorch/issues/5#issuecomment-651177352) for the versions.

## Datasets

* GOPRO_Large: [link](https://seungjunnah.github.io/Datasets/gopro)
* REDS: [link](https://seungjunnah.github.io/Datasets/reds)

## Usage examples

* Preparing dataset

Before running the code, put the datasets on a desired directory. By default, the data root is set as '~/Research/dataset'  
See: [src/option.py](src/option.py)
```python
group_data.add_argument('--data_root', type=str, default='~/Research/dataset', help='dataset root location')
```
Put your dataset under ```args.data_root```.

The dataset location should be like:
```bash
# GOPRO_Large dataset
~/Research/dataset/GOPRO_Large/train/GOPR0372_07_00/blur_gamma/....
# REDS dataset
~/Research/dataset/REDS/train/train_blur/000/...
```

* Example commands

```bash
# single GPU training
python main.py --n_GPUs 1 --batch_size 8 # save the results in default experiment/YYYY-MM-DD_hh-mm-ss
python main.py --n_GPUs 1 --batch_size 8 --save_dir GOPRO_L1  # save the results in experiment/GOPRO_L1

# adversarial training
python main.py --n_GPUs 1 --batch_size 8 --loss 1*L1+1*ADV
python main.py --n_GPUs 1 --batch_size 8 --loss 1*L1+3*ADV
python main.py --n_GPUs 1 --batch_size 8 --loss 1*L1+0.1*ADV

# train with GOPRO_Large dataset
python main.py --n_GPUs 1 --batch_size 8 --dataset GOPRO_Large
# train with REDS dataset (always set --do_test false)
python main.py --n_GPUs 1 --batch_size 8 --dataset REDS --do_test false --milestones 100 150 180 --end_epoch 200

# save part of the evaluation results (default)
python main.py --n_GPUs 1 --batch_size 8 --dataset GOPRO_Large --save_results part
# save no evaluation results (faster at test time)
python main.py --n_GPUs 1 --batch_size 8 --dataset GOPRO_Large --save_results none
# save all of the evaluation results
python main.py --n_GPUs 1 --batch_size 8 --dataset GOPRO_Large --save_results all
```

```bash
# multi-GPU training (DataParallel)
python main.py --n_GPUs 2 --batch_size 16
```

```bash
# multi-GPU training (DistributedDataParallel), recommended for the best speed
# single command version (do not set ranks)
python launch.py --n_GPUs 2 main.py --batch_size 16

# multi-command version (type in independent shells with the corresponding ranks, useful for debugging)
python main.py --batch_size 16 --distributed true --n_GPUs 2 --rank 0 # shell 0
python main.py --batch_size 16 --distributed true --n_GPUs 2 --rank 1 # shell 1
```

```bash
# single precision inference (default)
python launch.py --n_GPUs 2 main.py --batch_size 16 --precision single

# half precision inference (faster and requires less memory)
python launch.py --n_GPUs 2 main.py --batch_size 16 --precision half

# half precision inference with AMP
python launch.py --n_GPUs 2 main.py --batch_size 16 --amp true
```

```bash
# optional mixed-precision training
# mixed precision training may result in different accuracy
python main.py --n_GPUs 1 --batch_size 16 --amp true
python main.py --n_GPUs 2 --batch_size 16 --amp true
python launch.py --n_GPUs 2 main.py --batch_size 16 --amp true
```

```bash
# Advanced usage examples 
# using launch.py is recommended for the best speed and convenience
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large --milestones 500 750 900 --end_epoch 1000 --save_results none
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large --milestones 500 750 900 --end_epoch 1000 --save_results part
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large --milestones 500 750 900 --end_epoch 1000 --save_results all
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large --milestones 500 750 900 --end_epoch 1000 --save_results all --amp true

python launch.py --n_GPUs 4 main.py --dataset REDS --milestones 100 150 180 --end_epoch 200 --save_results all --do_test false
python launch.py --n_GPUs 4 main.py --dataset REDS --milestones 100 150 180 --end_epoch 200 --save_results all --do_test false --do_validate false
```

```bash
# Commands used to generate the below results
python launch.py --n_GPUs 2 main.py --dataset GOPRO_Large --milestones 500 750 900 --end_epoch 1000
python launch.py --n_GPUs 4 main.py --dataset REDS --milestones 100 150 180 --end_epoch 200 --do_test false
```

For more advanced usage, please take a look at src/option.py

## Results

* Single-precision training results

Dataset | GOPRO_Large | REDS
:--:|:--:|:--:
PSNR | 30.40 | 32.89
SSIM | 0.9018 | 0.9207
Download | [link](https://drive.google.com/file/d/1-wGC6s2D2ba-PSV60AeHf48HtYd9JkQ4/view?usp=sharing) | [link](https://drive.google.com/file/d/1aSPgVsNcPNqeGPn0Y2uGmgIwaIn5Njkv/view?usp=sharing)

* Mixed-precision training results

Dataset | GOPRO_Large | REDS | REDS (GOPRO_Large pretrained)
:--:|:--:|:--:|:--:
PSNR| 30.42 | 32.95 | 33.13
SSIM| 0.9021 | 0.9209 | 0.9237
Download | [link](https://drive.google.com/file/d/1TgiiiB-4lwWIIy8c-oSSkIy5g4GvDBKB/view?usp=sharing) | [link](https://drive.google.com/file/d/10hH5vtfGUUpy8jLvIBRCBqRoEhWRO1va/view?usp=sharing) | [link](https://drive.google.com/file/d/1YV6uhGLDBbvaiWN2_cYgUhYakmvLMAM9/view?usp=sharing)

Mixed-precision training uses less memory and is faster, especially on NVIDIA Turing-generation GPUs.
Loss scaling technique is adopted to cope with the narrow representation range of fp16.
This could improve/degrade accuracy.

* Inference speed on RTX 2080 Ti (resolution: 1280x720)

Inference in half precision has negligible effect on accuracy while it requires less memory and computation time.
type | FP32 | FP16
:--:|:--:|:--:
fps | 1.06 | 3.03
time (s) | 0.943 | 0.330

## Demo

To use the trained models, download files, unzip, and put them under DeepDeblur-PyTorch/experiment
* [GOPRO_L1](https://drive.google.com/file/d/1AfZhyUXEA8_UdZco9EdtpWjTBAb8BbWv/view?usp=sharing)
* [REDS_L1](https://drive.google.com/file/d/1UwFNXnGBz2rCBxhvq2gKt9Uhj5FeEsa4/view?usp=sharing)
* [GOPRO_L1_amp](https://drive.google.com/file/d/1ZcP3l2ZXj-C6yrDge5d3UxcaAKRN725w/view?usp=sharing)
* [REDS_L1_amp](https://drive.google.com/file/d/1do_HOjVFj2AYTX4BbwQ0enELRWtzhW6F/view?usp=sharing)
* [REDS_L1_amp_pretrained](https://drive.google.com/file/d/1BkEgUrFtOSymVnaADfptOvqfNOYiD3J1/view?usp=sharing)

```bash
python main.py --save_dir SAVE_DIR --demo true --demo_input_dir INPUT_DIR_NAME --demo_output_dir OUTPUT_DIR_NAME
# SAVE_DIR is the experiment directory where the parameters are saved (GOPRO_L1, REDS_L1)
# SAVE_DIR is relative to DeepDeblur-PyTorch/experiment
# demo_output_dir is by default SAVE_DIR/results
# image dataloader looks into DEMO_INPUT_DIR, recursively

# example
# single GPU (GOPRO_Large, single precision)
python main.py --save_dir GOPRO_L1 --demo true --demo_input_dir ~/Research/dataset/GOPRO_Large/test/GOPR0384_11_00/blur_gamma
# single GPU (GOPRO_Large, amp-trained model, half precision)
python main.py --save_dir GOPRO_L1_amp --demo true --demo_input_dir ~/Research/dataset/GOPRO_Large/test/GOPR0384_11_00/blur_gamma --precision half
# multi-GPU (REDS, single precision)
python launch.py --n_GPUs 2 main.py --save_dir REDS_L1 --demo true --demo_input_dir ~/Research/dataset/REDS/test/test_blur --demo_output_dir OUTPUT_DIR_NAME
# multi-GPU (REDS, half precision)
python launch.py --n_GPUs 2 main.py --save_dir REDS_L1 --demo true --demo_input_dir ~/Research/dataset/REDS/test/test_blur --demo_output_dir OUTPUT_DIR_NAME --precision half
```

## Differences from the original code

The default options are different from the original paper.
* RGB range is [0, 255]
* L1 loss (without adversarial loss. Usage possible. See above examples)
* Batch size increased to 16.
* Distributed multi-gpu training is recommended.
* Mixed-precision training enabled. Accuracy not guaranteed.
* SSIM function changed from MATLAB to python

## SSIM issue

There are many different SSIM implementations.  
In this repository, SSIM metric is based on the following function:
```python
from skimage.metrics import structural_similarity
ssim = structural_similarity(ref_im, res_im, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
```
`SSIM` class in [src/loss/metric.py](src/loss/metric.py) supports PyTorch.  
SSIM function in MATLAB is not correct if applied to RGB images. See [this issue](https://github.com/SeungjunNah/DeepDeblur_release/issues/51) for details.
