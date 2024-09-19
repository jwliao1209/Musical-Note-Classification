import torch
import numpy as np
from torchvision.transforms import Compose


class BaseTransform(object):
    def __init__(self, keys, **kwargs):
        self.keys = keys
        self._parse_var(**kwargs)

    def __call__(self, data, **kwargs):
        for key in self.keys:
            if key in data:
                data[key] = self._process(data[key], **kwargs)
            else:
                raise KeyError(f"{key} is not a key in data.")

        return data

    def _parse_var(self, **kwargs):
        pass

    def _process(self, single_data, **kwargs):
        NotImplementedError

    def _update_prob(self, cur_ep, total_ep):
        pass


class LoadMelSpec(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(LoadMelSpec, self).__init__(keys, **kwargs)

    def _process(self, single_data, **kwargs):
        single_data = np.load(single_data)
        return single_data.reshape(1, *single_data.shape)


class ToTensord(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ToTensord, self).__init__(keys, **kwargs)

    def _process(self, single_data, **kwargs):
        return torch.tensor(single_data)


def get_transforms():
    return Compose([
        LoadMelSpec(keys=['mel_spec']),
        ToTensord(keys=['mel_spec', 'label']),
    ])
