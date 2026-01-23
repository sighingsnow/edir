from .rzen_model import RzenModel
from .qwen2vl_model import Qwen2VLModel

def load_rzen(model_name_or_path: str, **kwargs):
    device = kwargs.get('device', 'cuda')
    return RzenModel(model_name_or_path, device=device, **kwargs)

def load_qwen2vl(model_name_or_path: str, **kwarge):
    return Qwen2VLModel(model_name_or_path, **kwargs)

def load_model(model_id, model_name_or_path: str, **kwargs):
    if 'rzen' in model_id:
        return RzenModel(model_name_or_path, **kwargs)
    elif 'ops' in model_id or 'gme' in model_id:
        return Qwen2VLModel(model_name_or_path, **kwargs)
    else:
        raise ValueError(f'Model {model_id} not supported')