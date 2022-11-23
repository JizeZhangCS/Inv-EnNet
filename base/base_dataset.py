from abc import abstractmethod
import time
import random
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def ret_visuals(self):
        raise NotImplementedError()

    @staticmethod
    def paired_rand_square_crop(img_list, square_size):
        if square_size == 0:
            return img_list

        width, height = img_list[0].size
        width_rand = random.randint(0, width - square_size)
        height_rand = random.randint(0, height - square_size)

        out_list = []
        for img in img_list:
            out_list.append(img.crop((width_rand, height_rand, width_rand + square_size, height_rand + square_size)))
        return out_list
    
    @staticmethod
    def rand_square_crop(img, square_size):
        if square_size == 0:
            return img
        
        width, height = img.size
        width_rand = random.randint(0, width - square_size)
        height_rand = random.randint(0, height - square_size)

        return img.crop((width_rand, height_rand, width_rand + square_size, height_rand + square_size))

    @staticmethod
    def read_img(img_pth):
        img = None
        for try_time in range(3):
            try:
                with open(img_pth, 'rb') as img_file:
                    temp = Image.open(img_file)
                    img = temp.copy()
            except IOError:
                print('fail No.' + str(try_time) + ' when loading: ' + img_pth)
                time.sleep(1)
                continue
            break
        assert img is not None
        return img

    @staticmethod
    def get_path(dataset, training):
        if dataset == "RetinexNet":
            if training:
                dataset_dir_list = ["RetinexNet/LOLdataset/our485", "RetinexNet/RAISE"]
                low_dir = "/low/"
                high_dir = "/high/"
            else:
                dataset_dir_list = ["RetinexNet/LOLdataset/eval15/low", "DICM", "LIME", "NPE", "MEF", "VV"]
                low_dir = "/"
                high_dir = "/"

        elif dataset == "RAISE":
            if training:
                dataset_dir_list = ["RetinexNet/RAISE"]
            else:
                raise NotImplementedError()
            low_dir = "/low/"
            high_dir = "/high/"

        elif dataset == "EGAN":
            if training:
                dataset_dir_list = ["EnlightenGAN/final_dataset"]
                low_dir = "/trainA/"
                high_dir = "/trainB/"
            else:
                dataset_dir_list = ["DICM", "LIME", "NPE", "MEF", "VV"]
                low_dir = '/'
                high_dir = None

        else:
            raise NotImplementedError()

        return dataset_dir_list, low_dir, high_dir
