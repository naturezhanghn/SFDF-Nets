from torch import nn
import torch
import torchvision.models as models
import numpy as np

# from skimage.measure import structural_similarity as ssim
# Loss functions
class PerceptualLoss():
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model
		
	def __init__(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
			
	def get_loss(self, fakeIm, realIm):
		fakeIm_ = torch.cat((fakeIm, fakeIm,fakeIm), 1)
		realIm_ = torch.cat((realIm, realIm,realIm), 1)
		f_fake = self.contentFunc.forward(fakeIm_)
		f_real = self.contentFunc.forward(realIm_)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss
'''
# 低版本python
class FFTLoss(nn.Module):
    # for torch version >= 1.7
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 =  nn.L1Loss()
    
    def get_loss(self, x, y):
        # fft_x = torch.fft.fftn(x,dim=(2,3))
        # fft_y = torch.fft.fftn(y,dim=(2,3))
        fft_x = torch.rfft(x, signal_ndim=2, onesided=False)
        fft_y = torch.rfft(y, signal_ndim=2, onesided=False)

        diff_real = fft_x[:, :, :, :, 0] - fft_y[:, :, :, :, 0]
        diff_imaginary = fft_x[:, :, :, :, 1] - fft_y[:, :, :, :, 1]
        diff_ = torch.sqrt(diff_real ** 2 + diff_imaginary ** 2)
        diff =  torch.pow(torch.abs(diff_),0.1)
        # return self.mse(diff, torch.zeros(diff.size()).cuda())
        return self.l1(diff, torch.zeros(diff.size()).cuda())

def fft2_show(img):
    img_fft = torch.rfft(img, signal_ndim=2, onesided=False)
    img_fft_real = img_fft[:, :, :, :, 0]
    img_fft_imag = img_fft[:, :, :, :, 1]
    diff_ = torch.sqrt(img_fft_real ** 2 + img_fft_imag ** 2)
    return  torch.pow(torch.abs(diff_),0.1)
'''

# 高版本python
class FFTLoss(nn.Module):
    # for torch version >= 1.7
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def get_loss(self, x, y):
        fft_x = torch.fft.fftn(x,dim=(2,3))
        fft_y = torch.fft.fftn(y,dim=(2,3))
        diff_ = (fft_x - fft_y)
        diff =  torch.pow(torch.abs(diff_),0.1)
        return self.mse(diff, torch.zeros(diff.size()).cuda())




