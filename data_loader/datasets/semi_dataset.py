import os
import random
import torch
from torchvision import transforms
from base.base_dataset import BaseDataset


to_tensor = transforms.Compose([transforms.ToTensor()])


class SemiDataset(BaseDataset):
    def __init__(self, fine_size, training, dataset_name='RetinexNet'):
        super(SemiDataset, self).__init__()
        self.dataset_base_dir = "../datasets/"
        self.dataset_dir_list, self.low_dir, self.gt_dir = self.get_path(dataset_name, training)
        self.unpaired_dataset_dir_list, _, self.high_dir = self.get_path("EGAN", training)

        self.training = training
        self.fine_size = fine_size

        self.img_name_list = []
        self.high_name_list = []
        self.low_dataset = []
        self.gt_dataset = []
        self.high_dataset = []

        for dataset_dir in self.dataset_dir_list:
            for gt_name in os.listdir(self.dataset_base_dir + dataset_dir + self.gt_dir):
                if gt_name[-4:] in [".png", ".jpg", ".bmp"]:
                    self.img_name_list.append(dataset_dir + self.gt_dir + gt_name)
                    self.low_dataset.append(self.read_img(self.dataset_base_dir + dataset_dir + self.low_dir + gt_name))
                    if training:
                        self.gt_dataset.append(self.read_img(self.dataset_base_dir + dataset_dir + self.gt_dir + gt_name))

        if training:
            for dataset_dir in self.unpaired_dataset_dir_list:
                for high_name in os.listdir(self.dataset_base_dir + dataset_dir + self.high_dir):
                    if high_name[-4:] == ".png":
                        self.high_name_list.append(self.dataset_base_dir + dataset_dir + self.high_dir + high_name)
                        self.high_dataset.append(
                            self.read_img(self.dataset_base_dir + dataset_dir + self.high_dir + high_name))

        self.dataset_size = len(self.low_dataset)
        # assert len(self.low_dataset) == self.dataset_size

    def __getitem__(self, idx):
        if self.training:
            low_img, gt_img = self.paired_rand_square_crop((self.low_dataset[idx], self.gt_dataset[idx]), self.fine_size)
            high_img = random.choice(self.high_dataset)
            high_img = self.rand_square_crop(high_img, self.fine_size)
            return to_tensor(low_img), to_tensor(gt_img), to_tensor(high_img)

        # return [(to_tensor(self.low_dataset[idx]), to_tensor(self.gt_dataset[idx])), self.img_name_list[idx]]
        return to_tensor(self.low_dataset[idx]), self.img_name_list[idx]

    def __len__(self):
        return self.dataset_size

    def ret_visuals(self):
        idx = random.randint(0, self.dataset_size-1)
        return torch.unsqueeze(to_tensor(self.low_dataset[idx]), 0), \
               torch.unsqueeze(to_tensor(self.gt_dataset[idx]), 0)
