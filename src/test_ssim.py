#from loss import PSNR, SSIM, SSIM_v2, SSIM_v3
from loss.metric import PSNR, SSIM
import torch

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

n, c, h, w = 1, 3, 32, 32
shape = (n, c, h, w)
data_range = 255

im1 = (torch.rand(shape) * data_range).round()
#im2 = (torch.rand(shape) * data_range).round()

#im1 = (torch.rand(shape) * data_range/2 + 127).round()
#im2 = (torch.rand(shape) * data_range/2 + 127).round()

im2 = im1.clone()/2 + (torch.rand(shape) * data_range/2).round()

#im1 = im1.double()
#im2 = im2.double()

im1 = im1.cuda()
im2 = im2.cuda()


psnr = PSNR(im1, im2)
ssim = SSIM(im1, im2)

print('PSNR: {}'.format(psnr))
print('SSIM: {}'.format(ssim))

ssim_half = SSIM(im1.half(), im2.half())
print('SSIM_h: {}'.format(ssim_half))



#print('SSIM_v2: {}'.format(ssim_v2))
#print('SSIM_v3: {}'.format(ssim_v3))

img1 = im1[0].permute(1, 2, 0).cpu().numpy()
img2 = im2[0].permute(1, 2, 0).cpu().numpy()

print()

psnr_s = peak_signal_noise_ratio(img1, img2, data_range=data_range)
ssim_s = structural_similarity(img1, img2, multichannel=True, gaussian_weights=True, use_sample_covariance=False, data_range=data_range)

print('PSNR_s: {}'.format(psnr_s))
print('SSIM_s: {}'.format(ssim_s))
