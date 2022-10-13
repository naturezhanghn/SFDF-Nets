import os 
from PIL import Image
import numpy as np


# Converts a Tensor into a Numpy array
  # |imtype|: the desired type of the converted numpy array
def tensor2array( image_tensor, imtype=np.uint8, normalize=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2array(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)

def saveImg(img, save_dir, type, name, Gray=True):
    fname, fext = name.split('.')
    imgPath = os.path.join(save_dir, "%s_%s.%s" % (fname, type, fext))
    img_array = tensor2array(img.data[0])
    image_pil = Image.fromarray(img_array)
    if Gray:
        image_pil.convert('L').save(imgPath)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
    else:
        image_pil.save(imgPath)
