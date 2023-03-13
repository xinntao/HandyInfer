import torch

from handyinfer.utils import load_file_from_url
from .inspyrenet_arch import InSPyReNet_SwinB

__all__ = ['InSPyReNet_SwinB']


def init_saliency_detection_model(model_name, half=False, device='cuda', model_rootpath=None):
    if model_name == 'inspyrenet':
        model = InSPyReNet_SwinB()
        model_url = 'https://huggingface.co/Xintao/HandyInfer/resolve/main/models/saliency_detection_InSpyReNet_SwinB.pth'  # noqa: E501
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='handyinfer/weights', progress=True, file_name=None, save_dir=model_rootpath)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    # model = model.to(device)
    if device == 'cuda':
        model = model.cuda()
    return model
