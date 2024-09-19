from src.utils import read_json
from src.transform import get_transforms

from src.dataset import AudiosDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    train_data = read_json("dataset/train.json")
    transforms = get_transforms()
    train_set = AudiosDataset(train_data, transform=transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=1
    )
