from base import BaseDataLoader
from data_loader.datasets.egan_dataset import EGANDataset
from data_loader.datasets.retinex_dataset import RetinexNetDataset
from data_loader.datasets.egan2_dataset import EGAN2Dataset
from data_loader.datasets.semi_dataset import SemiDataset


class EGANDataLoader(BaseDataLoader):
    def __init__(self, batch_size, fine_size, num_workers, training, shuffle=True, validation_split=0.0):
        self.dataset = EGANDataset(fine_size, training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class EGAN2DataLoader(BaseDataLoader):
    def __init__(self, batch_size, fine_size, num_workers, training, shuffle=True, validation_split=0.0):
        assert batch_size == 1 and fine_size == 0
        self.dataset = EGAN2Dataset(fine_size, training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class RetinexNetDataLoader(BaseDataLoader):
    def __init__(self, batch_size, fine_size, num_workers, training, shuffle=True, validation_split=0.0):
        self.dataset = RetinexNetDataset(fine_size, training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class SemiDataLoader(BaseDataLoader):
    def __init__(self, batch_size, fine_size, num_workers, training, shuffle=True, validation_split=0.0):
        self.dataset = SemiDataset(fine_size, training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
