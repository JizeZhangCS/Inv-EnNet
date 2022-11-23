import os
import random
import torch
from torchvision import transforms
from base.base_dataset import BaseDataset


to_tensor = transforms.Compose([transforms.ToTensor()])


class EGAN2Dataset(BaseDataset):
    def __init__(self, fine_size, training, dataset_name='EGAN'):
        super(EGAN2Dataset, self).__init__()
        self.dataset_base_dir = "../datasets/"
        self.dataset_dir_list, self.low_dir, self.high_dir = self.get_path(dataset_name, training=True)
        self.training = training
        self.fine_size = fine_size

        self.low_img_list = []
        self.high_img_list = []
        self.low_name_list = []
        self.high_name_list = []

        dataset_dir = self.dataset_dir_list[0]
        low_name = os.listdir(self.dataset_base_dir + dataset_dir + self.low_dir)[0]
        high_name = os.listdir(self.dataset_base_dir + dataset_dir + self.high_dir)[0]

        self.low_name_list.append(dataset_dir + self.low_dir + low_name)
        self.low_img_list.append(self.read_img(self.dataset_base_dir + dataset_dir + self.low_dir + low_name))
        self.high_name_list.append(dataset_dir + self.low_dir + high_name)
        self.high_img_list.append(self.read_img(self.dataset_base_dir + dataset_dir + self.high_dir + high_name))

        self.dataset_size = len(self.low_name_list)

    def __getitem__(self, idx):
        low_img = self.low_img_list[idx]

        if self.training:
            for i in range(3):
                try:
                    high_img = random.choice(self.high_img_list)
                    low_img = self.rand_square_crop(low_img, self.fine_size)
                    high_img = self.rand_square_crop(high_img, self.fine_size)
                    return to_tensor(low_img), to_tensor(high_img)
                except OSError:
                    print("No." + str(i) + "failed on reading " + self.low_name_list[idx])

        return to_tensor(low_img), self.low_name_list[idx]

    def __len__(self):
        return self.dataset_size

    def ret_visuals(self):
        idx = random.randint(0, self.dataset_size-1)
        return torch.unsqueeze(to_tensor(self.low_img_list[idx]), 0), \
            torch.unsqueeze(to_tensor(self.high_img_list[idx]), 0)
