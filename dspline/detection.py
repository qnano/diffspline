"""
Simple spot detection pipeline using correlation with PSF
"""
import torch
from typing import List, Tuple, Union, Optional, Final
from torch import Tensor
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import tifffile

ROIInfoDType = np.dtype([
    ('id', '<i4'), ('score', '<f4'), ('x', '<i4'), ('y', '<i4'), ('z', '<i4')
])


class SpotDetector(torch.nn.Module):
    def __init__(self, detection_threshold,
                 image_shape, psf,
                 spots_per_frame,
                 roisize=None,
                 bg_filter_size=None,
                 max_filter_size=None):

        if len(psf.shape) == 2:
            psf = psf[None]

        assert psf.shape[1] % 2 == 0 and psf.shape[2] % 2 == 0
        assert image_shape[0] == image_shape[1]

        super().__init__()

        pad_h = (image_shape[0] - psf.shape[1]) // 2
        pad_w = (image_shape[1] - psf.shape[2]) // 2
        self.input_shape = image_shape

        if bg_filter_size is None:
            bg_filter_size = psf.shape[-1] * 2

        if max_filter_size is None:
            max_filter_size = psf.shape[-1]

        if roisize is None:
            roisize = psf.shape[-1]

        self.roisize = roisize

        max_filter_pad = max_filter_size // 2 - 1
        self.max_filter = torch.nn.MaxPool2d(max_filter_size, stride=1, padding=max_filter_pad)

        self.psf_padded = torch.nn.functional.pad(psf, (pad_h, pad_h, pad_w, pad_w), 'constant', 0)
        self.psf_padded = torch.fft.fftshift(self.psf_padded, dim=(1, 2))
        self.psf_fd = torch.flip(torch.fft.fft2(self.psf_padded), dims=(1, 2))

        H, W = image_shape
        bg_filter = (torch.arange(H)[:, None] - H // 2) ** 2 + (
                    torch.arange(W)[None, :] - W // 2) ** 2 < bg_filter_size ** 2
        bg_filter = bg_filter.float() / (np.pi * bg_filter_size ** 2)
        self.bg_filter_fd = torch.fft.fft2(torch.fft.fftshift(bg_filter.to(psf.device)))

        self.detection_threshold = detection_threshold
        self.spots_per_frame = spots_per_frame
        self.device = psf.device

    def detect(self, image):
        """
        outputs:
        (roi_y, roi_x, intensity)
        """
        assert image.shape == self.input_shape
        image_fd = torch.fft.fft2(image)
        conv_psf = torch.fft.ifft2(image_fd[None] * self.psf_fd).real
        conv_bg = torch.fft.ifft2(image_fd * self.bg_filter_fd).real
        result = conv_psf / (conv_bg[None] + 1)
        result, z = result.max(0)
        max_result = self.max_filter(result[None])[0]

        i = 0  # sorry for hack  ## lol
        while len(max_result) < len(result):
            max_result = torch.nn.functional.pad(max_result, (1 - i, i, 1 - i, i), 'constant', 0.0)
            i = 1 - i

        mask = ((max_result == result) & (result > self.detection_threshold)).float() * result

        values, indices = torch.topk(mask.flatten(), self.spots_per_frame)

        # from ui.array_view import array_view
        # array_view(np.array([mask.cpu().numpy(), image.cpu().numpy()]))

        # imgsize = self.input_shape[0] * self.input_shape[1]
        # z = torch.div(indices, imgsize, rounding_mode='floor') # annoying warnings..
        # indices -= z * imgsize

        x = indices % self.input_shape[1]
        y = torch.div(indices, self.input_shape[0], rounding_mode='floor')  # annoying warnings..
        z = z.flatten()[indices]

        sel = values > self.detection_threshold
        return torch.stack((z, y, x), -1)[sel], values[sel]

    def forward(self, image):
        center, intensity = self.detect(image)
        roipos = center[:, 1:] - self.roisize // 2

        sel = ((roipos[:, 0] >= 0) & (roipos[:, 0] < self.input_shape[0] - self.roisize) &
               (roipos[:, 1] >= 0) & (roipos[:, 1] < self.input_shape[1] - self.roisize))

        roipos = roipos[sel].long()
        rois = self.extract(roipos, image[None])[0]

        return roipos, intensity[sel], rois

    def extract(self, roipos, stack):
        r = torch.arange(self.roisize, device=stack.device).long()
        Y = roipos[:, 0, None, None] + r[None, :, None]
        X = roipos[:, 1, None, None] + r[None, None, :]
        return stack[:, Y, X]


def load_gain_offset(gain_fn, offset_fn):
    print(f'estimating gain from light {gain_fn} and dark {offset_fn} frames')
    light = tifffile.imread(gain_fn)
    offset = tifffile.imread(offset_fn)

    assert len(light.shape) == 3
    assert len(offset.shape) == 3
    assert np.array_equal(light.shape[1:], offset.shape[1:])

    offset = np.mean(offset, 0)
    sig = light - offset
    v = np.var(sig, 0)
    m = np.mean(sig, 0)

    gain = m / v
    gain[gain == 0] = np.mean(gain)
    print(f'mean camera gain: {np.mean(gain):.2f} ADU/photons offset: {np.mean(offset):.2f}', flush=True)
    return gain, offset


# @torch.jit.script
def extract_rois(frames, roipos: Tensor, roisize: int):
    r = torch.arange(roisize, device=frames.device, dtype=torch.long)
    I = torch.arange(len(frames), device=frames.device, dtype=torch.long)
    Y = roipos[:, 0, None, None, None] + r[None, None, :, None]
    X = roipos[:, 1, None, None, None] + r[None, None, None, :]
    I = I[None, :, None, None]
    return frames[I, Y, X]


def detect_spots_in_movie(detector, camera_calib, movie_iterator, sumframes, totalframes,
                          output_fn, batch_size=10000):
    dev = detector.device

    with open(output_fn, "wb") as f, tqdm.tqdm(total=totalframes) as pb:
        numframes = 0
        numrois = 0
        nsummed = 0

        batch_info = []
        batch_rois = []

        rois_in_batch = 0

        def save_rois():
            nonlocal numrois, rois_in_batch
            np.save(f, np.concatenate(batch_info), allow_pickle=False)
            np.save(f, np.concatenate(batch_rois), allow_pickle=False)
            numrois += len(rois_info)
            rois_in_batch = 0
            batch_info.clear()
            batch_rois.clear()

        framebuf = None
        for i, img in enumerate(movie_iterator):
            img = torch.tensor(img.astype(np.float32), device=detector.device)
            img = camera_calib(img)

            if framebuf is None:
                framebuf = torch.zeros((sumframes, img.shape[0], img.shape[1]),
                                       dtype=img.dtype, device=dev)

            framebuf[nsummed] = img
            nsummed += 1

            if nsummed == sumframes:
                roipos, scores, sum_rois = detector.forward(framebuf.sum(0))
                rois = extract_rois(framebuf, roipos, detector.roisize)
                nsummed = 0

                roipos = roipos.cpu().numpy()
                rois = rois.cpu().numpy()

                rois_info = np.zeros((len(sum_rois)), dtype=ROIInfoDType)
                rois_info['id'] = i // sumframes
                rois_info['score'] = scores.cpu()
                rois_info['z'] = 0
                rois_info['y'] = roipos[:, 0]
                rois_info['x'] = roipos[:, 1]

                batch_info.append(rois_info)
                batch_rois.append(rois)

                numrois += len(sum_rois)
                rois_in_batch += len(sum_rois)

                if rois_in_batch >= batch_size:
                    save_rois()

            numframes += 1
            pb.set_description(f"#detected spots: {numrois}")
            pb.update()

        if rois_in_batch > 0:
            save_rois()

        return numrois, numframes


def end_of_file(f):
    curpos = f.tell()
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(curpos, 0)
    return curpos == file_size


def load_rois_iterator(rois_fn, maxrois=None):
    """
    Load rois sequentially so we can deal with very large datasets
    """
    with open(rois_fn, "rb") as f:
        total = 0
        while not end_of_file(f):
            rois_info = np.load(f, allow_pickle=True)
            pixels = np.load(f, allow_pickle=True)

            if maxrois is not None:
                if len(pixels) + total >= maxrois:
                    rem = maxrois - total
                    yield rois_info[:rem], pixels[:rem]
                    return

            total += len(pixels)
            yield rois_info, pixels


def load_rois(rois_fn, maxrois=None):
    rois_info = []
    pixels = []
    for ri, px in load_rois_iterator(rois_fn, maxrois):
        rois_info.append(ri)
        pixels.append(px)

    return np.concatenate(rois_info), np.concatenate(pixels)


if __name__ == '__main__':
    from gaussian_psf import Gaussian2DAstigmaticPSF
    from ui.array_view import array_view

    gauss3D_calib = [
        [1.0, -0.12, 0.2, 0.1],
        [1.05, 0.15, 0.19, 0]]

    N = 40
    W = 256
    intensity = 500
    bg = 50
    sigma = 1.5
    roisize = 14
    psf_model = Gaussian2DAstigmaticPSF(W, torch.tensor(gauss3D_calib))

    dev = torch.device('cuda')  # cuda gives a 15x speedup (RTX 2080 vs i7)

    spots = torch.rand(size=(N, 4), device=dev) * torch.tensor([W, W, 1, intensity], device=dev)
    spots[:, 2] -= 0.5
    params = torch.cat((spots, torch.zeros((N, 1), device=dev)), -1)
    image = psf_model.forward(params)[0].sum(0) + bg
    sample = torch.poisson(image)

    plt.figure()
    plt.imshow(sample.cpu().numpy())

    x = torch.tensor([
        [roisize / 2, roisize / 2, -0.5, 1, 0],
        [roisize / 2, roisize / 2, 0, 1, 0],
        [roisize / 2, roisize / 2, 0.5, 1, 0]])

    detection_template = Gaussian2DAstigmaticPSF(roisize, torch.tensor(gauss3D_calib)).forward(x)[0]

    # array_view(detection_template)

    sd = SpotDetector(1.2, (W, W), detection_template.to(dev), 20, bg_filter_size=30, max_filter_size=10)
    # sd = torch.jit.script(sd)

    cornerpos, intensity, rois = sd.forward(sample)

    array_view(rois, title='Detected ROIs')

    # plt.figure()
    # plt.imshow(im[0].cpu().numpy())
