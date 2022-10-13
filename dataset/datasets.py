import glob
import os
import numpy as np
import cv2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        if transforms_ is not None:
            self.transform = transforms.Compose(transforms_)
        else:
            self.transform = None
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        # print(os.path.join(root, mode),len(self.files))
        
        # if mode == "train":
        #     self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
        # if mode == "train":
        #     # self.files.extend(sorted(glob.glob(os.path.join("/hdd8T_3/zhangzr/ProcessedDataset/v8_uc_all_0.001_300/", "train") + "/*.*")))
        #     self.files.extend(sorted(glob.glob(os.path.join("/hdd8T_3/zhangzr/ProcessedDataset/v8_uc_img2img_0_0/", "train") + "/*.*")))
        # if mode == "test":
        #     self.files.extend(sorted(glob.glob(os.path.join("/workspace/zhangzr/project/Spider_Dataset/CoCo_img2img/", "test") + "/*.*")))

    def __getitem__(self, index):
        

        img_array_ = np.load(self.files[index % len(self.files)])
        img_array = np.nan_to_num(img_array_)

        c, h, w = img_array.shape
        img_A = img_array[:,:,0:int(w/2)].transpose(1,2,0)
        img_B = img_array[0:2,:,int(w/2):w].transpose(1,2,0)

        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

