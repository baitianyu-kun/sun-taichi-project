from mmcv import Config, DictAction

from mmpose.apis.webcam import WebcamExecutor
from mmpose.apis.webcam.nodes import model_nodes

def set_device(cfg: Config, device: str):
    device = device.lower()
    assert device == 'cpu' or device.startswith('cuda:')
    for node_cfg in cfg.executor_cfg.nodes:
        if node_cfg.type in model_nodes.__all__:
            node_cfg.update(device=device)
    return cfg

cfg = Config.fromfile('web_cam_test_config.py')
cfg = set_device(cfg, 'cuda:0')
executor = WebcamExecutor(**cfg.executor_cfg)
executor.run()