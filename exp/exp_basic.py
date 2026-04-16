import importlib
import os
import torch


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_registry = {
            'Autoformer': 'models.Autoformer',
            'DLinear': 'models.DLinear',
            'FEDformer': 'models.FEDformer',
            'Informer': 'models.Informer',
            'PatchTST': 'models.PatchTST',
            'iTransformer': 'models.iTransformer',
            'CAMEL': 'models.CAMEL',
            'PhaseFormer': 'models.PhaseFormer',
            'MixLinear': 'models.MixLinear',
            'FreqCycle': 'models.FreqCycle',
            'stgcn': 'models.stgcn',
            'gwn': 'models.gwn',
            'astgcn': 'models.astgcn',
            'pdformer': 'models.pdformer',
        }
        self.model_dict = {args.model: self._load_model_module(args.model)}
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _load_model_module(self, model_name):
        if model_name not in self.model_registry:
            supported = ', '.join(self.model_registry.keys())
            raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")
        return importlib.import_module(self.model_registry[model_name])

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
