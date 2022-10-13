import torch
import numpy as np


#
def freq2imgnormal(img_freq):
    fft_abs = torch.pow(img_freq[:,0:1,:,:], 10)
    fft_angle = img_freq[:,1:2,:,:] * 2 * np.pi
    img_fft = ( fft_abs * torch.cos(fft_angle) + 1j * fft_abs * torch.sin(fft_angle) ) 
    img_normal = torch.abs(torch.fft.ifftn(img_fft,dim=(2,3))) 
    img_normal_mask = torch.zeros_like(img_normal)
    img_normal_mask[:,:,3:253,3:253] = img_normal[:,:,3:253,3:253]
    img = (img_normal_mask - torch.min(img_normal_mask))/(torch.max(img_normal_mask) - torch.min(img_normal_mask))
    return img
# def freq2imgnormal(img_freq):
#     fft_abs = torch.pow(img_freq[:,0:1,:,:], 10)
#     fft_angle = img_freq[:,1:2,:,:] * 2 * np.pi
#     img_fft = ( fft_abs * torch.cos(fft_angle) + 1j * fft_abs * torch.sin(fft_angle) ) 
#     img_normal = torch.abs(torch.fft.ifftn(img_fft,dim=(2,3))) 
#     img_normal_mask = torch.zeros_like(img_normal)
#     img_crop = img_normal[:,:,3:253,3:253]
#     img_crop = (img_crop - torch.min(img_crop))/(torch.max(img_crop) - torch.min(img_crop))
#     img_normal_mask[:,:,3:253,3:253] = img_crop
#     return img_normal_mask

def freq2img(img_freq):
    fft_abs = torch.pow(img_freq[:,0:1,:,:], 10)
    fft_angle = img_freq[:,1:2,:,:] * 2 * np.pi
    img_fft = (fft_abs * torch.cos(fft_angle) + 1j * fft_abs * torch.sin(fft_angle))
    # print(img_fft.shape)
    img_normal = torch.abs(torch.fft.ifftn(img_fft,dim=(2,3))) 
    return img_normal

#
def freq2img1(img_freq):
    fft_abs = torch.pow(img_freq[:,0:1,:,:], 10)
    fft_angle = img_freq[:,1:2,:,:] * 2 * np.pi
    img_fft = ( fft_abs * torch.cos(fft_angle) + 1j * fft_abs * torch.sin(fft_angle) )
    img_normal = torch.abs(torch.fft.ifftn(img_fft,dim=(2,3))) 
    img_normal_1 = (img_normal - torch.min(img_normal))/(torch.max(img_normal) - torch.min(img_normal) )
    return img_normal_1

# 
def freq2img1_edge(img_freq):
    fft_abs = torch.pow(img_freq[:,0:1,:,:], 10)
    fft_angle = img_freq[:,1:2,:,:] * 2 * np.pi
    img_fft = ( fft_abs * torch.cos(fft_angle) + 1j * fft_abs * torch.sin(fft_angle) )
    img_normal = torch.abs(torch.fft.ifftn(img_fft,dim=(2,3))) 
    # img_normal_1 = (img_normal - torch.min(img_normal))/(torch.max(img_normal) - torch.min(img_normal) )
    img_normal_mask = torch.zeros_like(img_normal)
    img_normal_mask[:,:,3:253,3:253] = img_normal[:,:,3:253,3:253]
    img = (img_normal_mask - torch.min(img_normal_mask))/(torch.max(img_normal_mask) - torch.min(img_normal_mask))
    return img

def freq2img3(img_freq):
    img = torch.zeros([img_freq.size()[0],3,256,256])
    # img = img_freq[:,0:3,:,:]
    for i in [0,2,4]:
        fft_abs = torch.pow(img_freq[:,i:i+1,:,:], 10)
        fft_angle = img_freq[:,i+1:i+2,:,:] * 2 * np.pi
        img_fft = fft_abs * torch.cos(fft_angle) + 1j * fft_abs * torch.sin(fft_angle)
        img_normal = torch.abs(torch.fft.ifftn(img_fft,dim=(2,3)))
        img_normal_3 = (img_normal - torch.min(img_normal))/(torch.max(img_normal) - torch.min(img_normal) )
        # print(img_normal_3.shape)
        img[:,int(i/2):int(i/2)+1,:,:] =  img_normal_3 
    return img

# def freq2img_crop_normal(img_freq):
#     fft_abs = torch.pow(img_freq[:,0:1,:,:], 10)
#     fft_angle = img_freq[:,1:2,:,:] * 2 * np.pi
#     img_fft = ( fft_abs * torch.cos(fft_angle) + 1j * fft_abs * torch.sin(fft_angle) )
#     img_normal = torch.abs(torch.fft.ifftn(img_fft,dim=(2,3))) 
#     img_normal_crop = img_normal[:,:,3:253,3:253]
#     img = (img_normal_crop - torch.min(img_normal_crop))/(torch.max(img_normal_crop) - torch.min(img_normal_crop) )
#     return img

# def freq2imgcrop(img_freq):
#     fft_abs = torch.pow(img_freq[:,0:1,:,:], 10)
#     fft_angle = img_freq[:,1:2,:,:] * 2 * np.pi
#     img_fft = ( fft_abs * torch.cos(fft_angle) + 1j * fft_abs * torch.sin(fft_angle) )
#     img_normal = torch.abs(torch.fft.ifftn(img_fft,dim=(2,3)))  
#     img_normal_crop = img_normal[:,:,3:253,3:253]
#     return img_normal_crop
    
# def fft2_show(img):
#     img_fft = torch.fft.fftn(img,dim=(2,3))
#     return  torch.pow(torch.abs(img_fft),0.1)

def fft2_show_abs_normal(img):
    img_fft = torch.fft.fftn(img,dim=(2,3))
    img_abs = torch.pow(torch.abs(img_fft),0.1)
    return  (img_abs - torch.min(img_abs))/(torch.max(img_abs) - torch.min(img_abs) )

def fft2_show_angle_normal(img):
    img_fft = torch.fft.fftn(img,dim=(2,3))
    img_angle = torch.angle(img_fft)
    angle_normal = ((img_angle + 2 * np.pi) % (2 * np.pi) / (2*np.pi))
    return  angle_normal

def img2fft2(img):
    img_fft = torch.fft.fftn(img,dim=(2,3))
    img_abs = torch.pow(torch.abs(img_fft),0.1)
    img_angle = torch.angle(img_fft)
    abs_normal = (img_abs - torch.min(img_abs))/(torch.max(img_abs) - torch.min(img_abs) )
    angle_normal = ((img_angle + 2 * np.pi) % (2 * np.pi) / (2*np.pi))
    return  torch.cat((abs_normal, angle_normal),1)    


