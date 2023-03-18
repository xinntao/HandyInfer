import cv2
import os
import os.path as osp
import torch
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

from handyinfer.utils.percentile import Percentile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def load_file_from_url(url, model_dir=None, progress=True, file_name=None, save_dir=None):
    """Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    if save_dir is None:
        save_dir = os.path.join(ROOT_DIR, model_dir)
    os.makedirs(save_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(save_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


class TensorGrayR():

    def __init__(self):
        self.gray_r_map = torch.linspace(1, 0, 256)
        self.gray_r_map = self.gray_r_map.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
        self.percentile = Percentile()

    def to_gray_r(self,
                  value,
                  vmin=None,
                  vmax=None,
                  invalid_val=-99,
                  invalid_mask=None,
                  background_color=128,
                  dtype='float32'):
        """Converts a depth map to a gray revers image.
        Args:
            value (torch.Tensor): Input depth map. Shape: (b, 1, H, W).
            All singular dimensions are squeezed
            vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used.
            Defaults to None.
            vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used.
            Defaults to None.
            invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'.
            Defaults to -99.
            invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
            background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels.
            Defaults to (128, 128, 128).
        Returns:
            tensor.Tensor, dtype - float32 if dtype == 'float32 or unit8: gray reverse depth map. shape (b, 1, H, W)
        """
        # Percentile can only process the first dimension
        self.gray_r_map = self.gray_r_map.to(value.device)
        n, c, h, w = value.shape
        value = value.reshape(n, c, h * w).permute(2, 0, 1)

        if invalid_mask is None:
            invalid_mask = value == invalid_val
        mask = torch.logical_not(invalid_mask)

        # normaliza
        vmin_vmax = self.percentile(value[mask], [2, 85])
        vmin = vmin_vmax[0] if vmin is None else vmin
        vmax = vmin_vmax[1] if vmax is None else vmax

        value[:, vmin == vmax] = value[:, vmin == vmax] * 0.
        value[:, vmin != vmax] = (value[:, vmin != vmax] - vmin[vmin != vmax]) / (
            vmax[vmin != vmax] - vmin[vmin != vmax])

        value[invalid_mask] = torch.nan

        diff = torch.abs(self.gray_r_map - value)
        min_ids = torch.argmin(diff, dim=0)  # [h*w, n, c]

        min_ids[invalid_mask] = background_color
        min_ids = min_ids.reshape(h, w, n, c).permute(2, 3, 0, 1)

        if dtype == 'float32':
            min_ids = min_ids.type(value.dtype) / 255.0  # [0,1]

        return min_ids
