import copy
from torch.utils.data import Dataset


class AudiosDataset(Dataset):
    def __init__(self, data_list, transform=None):
        super(AudiosDataset, self).__init__()
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        data = copy.deepcopy(self.data_list[i])
        return self.transform(data) if self.transform is not None else data
