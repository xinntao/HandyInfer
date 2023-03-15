import torch

from handyinfer.utils import load_file_from_url
from .DPT_BEiT_L_384_arch import DPTDepthModel
from .midas import MidasCore
from .zoedepth_arch import ZoeDepth

__all__ = ['ZoeDepth']


def init_depth_estimation_model(model_name, device='cuda', model_rootpath=None, img_size=[384, 512]):
    if model_name == 'ZoeD_N':

        # "DPT_BEiT_L_384"
        midas = DPTDepthModel(
            path=None,
            backbone='beitl16_384',
            non_negative=True,
        )

        core = MidasCore(midas, freeze_bn=True, img_size=img_size)
        core.set_output_channels('DPT_BEiT_L_384')
        model = ZoeDepth(core)
        model_url = 'https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='handyinfer/weights', progress=True, file_name=None, save_dir=model_rootpath)
    model.load_state_dict(torch.load(model_path)['model'], strict=True)
    model.eval()
    model = model.to(device)
    return model
