# from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
from torch import nn

def _expand(img):
    if img.ndim < 4:
        img = img.expand([1] * (4-img.ndim) + list(img.shape))

    return img

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, im1, im2, data_range=None):
        # tensor input, constant output

        if data_range is None:
            data_range = 255 if im1.max() > 1 else 1

        se = (im1-im2)**2
        se = _expand(se)

        mse = se.mean(dim=list(range(1, se.ndim)))
        psnr = 10 * (data_range**2/mse).log10().mean()

        return psnr

class SSIM(nn.Module):
    def __init__(self, device_type='cpu', dtype=torch.float32):
        super(SSIM, self).__init__()

        self.device_type = device_type
        self.dtype = dtype      # SSIM in half precision could be inaccurate

        def _get_ssim_weight():
            truncate = 3.5
            sigma = 1.5
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
            nch = 3

            weight = torch.Tensor([-(x - win_size//2)**2/float(2*sigma**2) for x in range(win_size)]).exp().unsqueeze(1)
            weight = weight.mm(weight.t())
            weight /= weight.sum()
            weight = weight.repeat(nch, 1, 1, 1)

            return weight

        self.weight = _get_ssim_weight().to(self.device_type, dtype=self.dtype, non_blocking=True)

    def forward(self, im1, im2, data_range=None):
        """Implementation adopted from skimage.metrics.structural_similarity
        Default arguments set to multichannel=True, gaussian_weight=True, use_sample_covariance=False
        """

        im1 = im1.to(self.device_type, dtype=self.dtype, non_blocking=True)
        im2 = im2.to(self.device_type, dtype=self.dtype, non_blocking=True)

        K1 = 0.01
        K2 = 0.03
        sigma = 1.5

        truncate = 3.5
        r = int(truncate * sigma + 0.5)  # radius as in ndimage
        win_size = 2 * r + 1

        im1 = _expand(im1)
        im2 = _expand(im2)

        nch = im1.shape[1]

        if im1.shape[2] < win_size or im1.shape[3] < win_size:
            raise ValueError(
                "win_size exceeds image extent.  If the input is a multichannel "
                "(color) image, set multichannel=True.")

        if data_range is None:
            data_range = 255 if im1.max() > 1 else 1

        def filter_func(img):   # no padding
            return nn.functional.conv2d(img, self.weight, groups=nch).to(self.dtype)
            # return torch.conv2d(img, self.weight, groups=nch).to(self.dtype)

        # compute (weighted) means
        ux = filter_func(im1)
        uy = filter_func(im2)

        # compute (weighted) variances and covariances
        uxx = filter_func(im1 * im1)
        uyy = filter_func(im2 * im2)
        uxy = filter_func(im1 * im2)
        vx = (uxx - ux * ux)
        vy = (uyy - uy * uy)
        vxy = (uxy - ux * uy)

        R = data_range
        C1 = (K1 * R) ** 2
        C2 = (K2 * R) ** 2

        A1, A2, B1, B2 = ((2 * ux * uy + C1,
                        2 * vxy + C2,
                        ux ** 2 + uy ** 2 + C1,
                        vx + vy + C2))
        D = B1 * B2
        S = (A1 * A2) / D

        # compute (weighted) mean of ssim
        mssim = S.mean()

        return mssim
