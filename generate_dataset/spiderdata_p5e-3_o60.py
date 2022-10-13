import numpy as np
import cv2
import os

#Set path
DATASET1 = r'/data/zhangzr/dotav1/test/'
DATASET2 = r'/data/zhangzr/UCmerceLandUse/'
DATASETs = [DATASET1,DATASET2 ]
# DATASETs = [DATASET2 ]
SAVE = r"/data2/zhangzr/SpiderDataset/spiderdata_p5e-3_o60/"
os.makedirs(SAVE,exist_ok=True)
os.makedirs(SAVE + "/val", exist_ok=True)
os.makedirs(SAVE + "/test", exist_ok=True)
os.makedirs(SAVE + "/train", exist_ok=True)


# Image size
h = 256
w = h  
# Distance of target
z =  80e+3
# Wave length
waves_length = np.array([1065, 1094, 1123, 1151, 1180, 1208, 
               1237, 1265, 1296, 1322, 1351, 1378, 
               1409, 1437, 1465, 1494, 1520, 1550])*1e-9
# Length of the baseline
B = np.array( [0.72, 1.44, 2.16, 2.88, 3.60, 4.32, 5.76, 7.20, 9.36, 12.23, 15.84, 20.88 ])*1e-3;

u = np.zeros(len(waves_length)*len(B))
i = 0
for wave in waves_length:
    for B_i in B:
        u[i] = B_i / (wave * z)
        i = i + 1
pixel_size =51.2e-2/h
R = np.sort(u) / pixel_size

# Spider signal distribution
def mask(rot_angle,t=1,h=h,w=w,R=R):
    spider_mask = np.zeros([h,w])
    arm = 37*t
    theta = np.linspace(0, 2*np.pi/arm*(arm-1), arm) + rot_angle
    for i in theta:
        for j in R:
            spider_mask[round(j*np.cos(i))][round(j*np.sin(i))] = 1
    return spider_mask
mask1 = mask(0)
mask2 = mask(3.2432/360*2*np.pi)
mask3 = mask(2*3.2432/360*2*np.pi)
cv2.imwrite(SAVE+"mask1.png",mask1*255)
cv2.imwrite(SAVE+"mask2.png",mask2*255)
cv2.imwrite(SAVE+"mask3.png",mask3*255)
spider_masks = [mask1,mask2,mask3]

all_masks = mask(0,3)
cv2.imwrite(SAVE+"all_masks.png",all_masks*255)

def spider(img_gray,ind,dataset_divide="train", P_nosiy = 0.005, OPD = 60,h = h,w = w,all_mask = False):
    # fft and fftshift
    fft_img = np.fft.fft2(img_gray) 
    fft_abs = np.abs(fft_img)
    angles = np.angle( fft_img )
    
    # reference data 
    fft_abs_normal = np.power(np.abs(fft_abs) / np.max(np.abs(fft_abs)) ,0.1).astype(np.float32)
    fft_angle_normal = ((angles + 2 * np.pi) % (2 * np.pi) / (2*np.pi)).astype(np.float32)
    
    # add noisy
    fft_abs_noise = np.abs(fft_abs/np.max(fft_abs) + P_nosiy * np.random.normal(0,1,[h,w]) )
    fft_angle_noise = angles +  np.random.normal(0,1,[h,w]) * OPD / (1200) * 2 * np.pi 
    
    fft_abs_noise[0][0] = 1
    fft_angle_noise[0][0] = 0
    
    fft_abs_normal_ = np.power(np.abs(fft_abs_noise) / np.max(np.abs(fft_abs_noise)) ,0.1).astype(np.float32)
    fft_angle_normal_ = ((fft_angle_noise + 2 * np.pi) % (2 * np.pi) / (2*np.pi)).astype(np.float32)
    
    # print(fft_abs_normal , fft_angle_normal)
    cntc = 0
    if all_mask:
        array_out = np.zeros([6,h,w*2]).astype(np.float32)
        fft_abs_spider = fft_abs_normal_ * all_masks
        fft_angle_spider = fft_angle_normal_ * all_masks 
        fft_abs_spider[0][0] = 1
        fft_angle_spider[0][0] = 0

        array_abs = np.hstack(( fft_abs_spider, fft_abs_normal )).astype(np.float32)
        array_angle = np.hstack(( fft_angle_spider, fft_angle_normal )).astype(np.float32)

        array_out[0, :, :] = array_abs
        array_out[1, :, :] = array_angle

        name =  str(ind).zfill(6) + ".npy" 
        os.makedirs(os.path.join(SAVE ,dataset_divide),exist_ok=True)
        np.save(os.path.join(SAVE ,dataset_divide , name), array_out)

    else:
        array_out = np.zeros([6,h,w*2]).astype(np.float32)
        for spider_mask in spider_masks :
            # spider sample
            fft_abs_spider = fft_abs_normal_ * spider_mask 
            fft_angle_spider = fft_angle_normal_ * spider_mask 
            
            fft_abs_spider[0][0] = 1
            fft_angle_spider[0][0] = 0
            
            # fft_abs_spider = np.fft.fftshift(fft_abs_spider)
            # fft_angle_spider = np.fft.fftshift(fft_angle_spider)
            # fft_abs_normal = np.fft.fftshift(fft_abs_normal)
            # fft_angle_normal = np.fft.fftshift(fft_angle_normal)

            array_abs = np.hstack(( fft_abs_spider, fft_abs_normal )).astype(np.float32)
            array_angle = np.hstack(( fft_angle_spider, fft_angle_normal )).astype(np.float32)
            
            array_out[cntc, :, :] = array_abs
            array_out[cntc + 1, :, :] = array_angle
            cntc = cntc + 2
    
        name =  str(ind).zfill(6) + ".npy" 
        os.makedirs(os.path.join(SAVE ,dataset_divide),exist_ok=True)
        np.save(os.path.join(SAVE ,dataset_divide , name), array_out)


def main():
    ind_val = 1
    ind_test = 1
    ind_train = 1
    ind_UCtest = 1
    num_divide_datateset = 0
    a = 0
    
    # Get image file path 
    for DATASET in DATASETs:
        img_dirs_name = os.listdir(DATASET)
        for dir_name in img_dirs_name:
            if dir_name.split(".")[-1] == "zip":
                print(dir_name)
                continue
            img_dir_path = os.path.join(DATASET, dir_name)
            imgs_name = sorted(os.listdir(img_dir_path))
            print(len(imgs_name),dir_name)


            # Get img path
            for img_name in imgs_name:
                
                if img_name.split(".")[-1] == "zip":
                    print(dir_name)
                    continue
                print(img_name)
                
                img_path = os.path.join(DATASET, dir_name,img_name)
                image_read = cv2.imread(img_path )
                
                if len(image_read) == None:
                    continue  
                
                H ,W ,C = image_read.shape
                min_len = min([H,W])
                flag = 1
                for y in [W/2 ,W/4 ,3/4*W]:
                    for x in [H/2 ,H/4 ,3/4*H]:
                        if min_len <= 4*h+2:
                            flag = 1
                            if H == W:
                                image = image_read
                            elif H == min_len:
                                image = image_read[:, int(W/2 - H/2) : int(W/2 + H/2) ,:]
                            elif W == min_len:
                                image = image_read[ int(H/2 - W/2) : int(H/2 + W/2) ,:,:]
                        else:
                            flag = 0
                            image = image_read[int(x-h//2):int(x+h//2),int(y-w//2):int(y+w//2),:]
                        image_resize = cv2.resize(image,(h,w),interpolation=cv2.INTER_CUBIC)
                        
                        # Convert to Gray
                        if (image_resize.shape[2] == 3):
                            img_gray_ = cv2.cvtColor(image_resize,cv2.COLOR_RGB2GRAY)
                        else:
                            img_gray_ = image_resize 
                        img_gray = 255.0 * (img_gray_ - np.min(img_gray_) ) / (np.max(img_gray_) - np.min(img_gray_))
                        
                        judge =  np.var(img_gray)
                        if  DATASET != DATASET2 and (judge <= 500 or judge >= 3000):
                            a = a + 1
                            print(a)
                            continue
                                          
                        num_divide_datateset +=1 

                        if DATASET == DATASET2:
                            spider(img_gray,ind_UCtest,dataset_divide="UCtest")
                            spider(img_gray,ind_UCtest,dataset_divide="UCtestall",all_mask = True) # dirty image for compare (all )
                            ind_UCtest += 1
                        elif (num_divide_datateset %10  == 0)  :
                            spider(img_gray,ind_val,dataset_divide="val")
                            spider(img_gray,ind_val,dataset_divide="valall",all_mask = True) # dirty image for compare
                            ind_val += 1
                        elif (num_divide_datateset %10  == 1) :
                            spider(img_gray,ind_test,dataset_divide="testall",all_mask = True) # dirty image for compare
                            spider(img_gray,ind_test,dataset_divide="test")
                            ind_test += 1
                        else:
                            spider(img_gray,ind_train,dataset_divide="train")
                            ind_train += 1 
                        if flag == 1:
                            break
                    if flag == 1:
                        break
    print(ind_val ,ind_test ,ind_train )
    print(a)

main()

