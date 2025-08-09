from easydict import EasyDict
import yaml
import importlib
from bert import MODELS
import torch


def register_module_from_config(cfg, registry):
    """
    动态加载类并注册到注册表中。
    Args:
        cfg (dict): 配置字典，包含类的模块路径和类名。
        registry (:obj:`Registry`): 注册表对象。
    """
    if 'module' not in cfg or 'class' not in cfg:
        raise KeyError('cfg must contain "module" and "class" keys')

    module_name = cfg['module']
    class_name = cfg['class']

    # 动态导入模块
    module = importlib.import_module(module_name)

    # 获取类
    cls = getattr(module, class_name)

    # 注册类到注册表
    registry.register_module(module=cls)

def build_model_from_cfg(cfg, **kwargs):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT):
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    if 'module' in cfg and 'class' in cfg:
        register_module_from_config(cfg, MODELS)
    return MODELS.build(cfg, **kwargs)

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config

def get_config():
    config = cfg_from_yaml_file("/home/qq/2/x/PointTransformer.yaml")
    return config

def model_builder(config):
    model = build_model_from_cfg(config)
    return model


def load_model(base_model, ckpt_path):
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict = True)

    return


def test_net(config):
    base_model = model_builder(config.model)
    # load checkpoints
    load_model(base_model, "/home/qq/2/x/PointTransformer_ModelNet1024points.pth")  # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    return base_model

    #test(base_model, test_dataloader, args, config, logger=logger)

conf = get_config()


