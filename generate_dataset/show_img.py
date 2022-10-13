import cv2
import numpy as np
import os
import glob

os.makedirs("./imageshow",exist_ok=True)
os.makedirs("./orinshow",exist_ok=True)
files_val = sorted(glob.glob("./val/*.*"))
files_test = sorted(glob.glob("./test/*.*"))
files_train = sorted(glob.glob("./train/*.*"))
files_UCtest = sorted(glob.glob("./UCtest/*.*"))
files = files_train + files_val + files_test + files_UCtest
# files = files_UCtest + files_train + files_val + files_test
# files = files_test
files_valall = sorted(glob.glob("./valall/*.*"))
files_UCtestall = sorted(glob.glob("./UCtestall/*.*"))
files_testall = sorted(glob.glob("./testall/*.*"))
files2 = files_valall + files_UCtestall + files_testall  
for file in files2:
    name = file.split("/")[-2] + file.split("/")[-1].split(".")[0]
    image_array = np.load(file)
    print(name)

    img_abs = (image_array[0,:,:256])
    img_angle = (image_array[1,:,:256])
    
    img_fft = np.power(img_abs,10)*np.cos(img_angle*2*np.pi)+ 1j*np.power(img_abs,10)*np.sin(img_angle*2*np.pi)
    img_ = np.abs(np.fft.fft2(img_fft))
    img = 255 * (img_ - np.min(img_ )) / (np.max(img_ ) - np.min(img_ ))

    img_abs_ori = (image_array[0,:,256:512])
    img_angle_ori = (image_array[1,:,256:512])

    img_fft_ori = np.power(img_abs_ori,10)*np.cos(img_angle_ori*2*np.pi)+ 1j*np.power(img_abs_ori,10)*np.sin(img_angle_ori*2*np.pi)
    img_ori = np.abs(np.fft.fft2(img_fft_ori))
    img_ori = 255 * (img_ori- np.min(img_ori )) / (np.max(img_ori ) - np.min(img_ori ))

    img_out = np.hstack((img_abs*255 , img_angle*255  ,  img , img_ori))
#     cv2.imwrite("./orinshow/"+name+"img_out.png",img_out)

for file in files:
    name = file.split("/")[-2] + file.split("/")[-1].split(".")[0]
    image_array = np.load(file)

    print(name)
    img_abs = (image_array[4,:,0:256])
    img_angle = (image_array[5,:,0:256])
    # cv2.imwrite("./imageshow/"+name+"img_abs.png",img_abs*255)
    # cv2.imwrite("./imageshow/"+name+"img_angle.png",img_angle*255)

    img_fft = np.power(img_abs,10)*np.cos(img_angle*2*np.pi)+ 1j*np.power(img_abs,10)*np.sin(img_angle*2*np.pi)
    img_ = np.abs(np.fft.fft2(img_fft))
    img = 255 * (img_ - np.min(img_ )) / (np.max(img_ ) - np.min(img_ ))
    # cv2.imwrite("./imageshow/"+name+"img.png",img)

    img_abs_ori = (image_array[4,:,256:512])
    img_angle_ori = (image_array[5,:,256:512])

    img_fft_ori = np.power(img_abs_ori,10)*np.cos(img_angle_ori*2*np.pi)+ 1j*np.power(img_abs_ori,10)*np.sin(img_angle_ori*2*np.pi)
    img_ori = np.abs(np.fft.fft2(img_fft_ori))
    img_ori = 255 * (img_ori- np.min(img_ori )) / (np.max(img_ori ) - np.min(img_ori ))
    # cv2.imwrite("./imageshow/"+name+"img_ori.png",img_ori)
    img_out = np.hstack((img_abs*255 , img_angle*255  ,  img , img_ori))
    cv2.imwrite("./imageshow/"+name+"img_out.png",img_out)

