import numpy as np 

##############get mask####################################################
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
def define_mask(rot_angle,h=h,w=w,R=R,t=3):
    spider_mask = np.zeros([h,w])
    arm = 37*t
    theta = np.linspace(0, 2*np.pi/arm*(arm-1), arm) + rot_angle
    for i in theta:
        for j in R:
            # spider_mask(ceil(h/2+j*cos(i)),ceil(w/2+j*sin(i))) = 1
            spider_mask[round(j*np.cos(i))][round(j*np.sin(i))] = 1
    return spider_mask


def fft_3sample_addnoise(fft_real,mask,P_nosiy=0.005,OPD=60):

    fft_abs_noise   = np.power( np.power(fft_real[0:1,:,:],10) + P_nosiy * np.random.normal(0,1,[1,h,w]) ,0.1 ) * np.expand_dims(mask, axis=0)
    fft_angle_noise = ( fft_real[1:2,:,:] + np.random.normal(0,1,[1,h,w]) * OPD / (1200)  )* np.expand_dims(mask, axis=0)
    
    # # add noisy
    # fft_abs_noise = np.abs(fft_abs/np.max(fft_abs) + P_nosiy * np.random.normal(0,1,[h,w]) )
    # fft_angle_noise = angles +  np.random.normal(0,1,[h,w]) * OPD / (1200) * 2 * np.pi 
    
    fft_abs_noise[0][0] = 1
    fft_angle_noise[0][0] = 0
    
    # fft_abs_spider= np.power(np.abs(fft_abs_noise) / np.max(np.abs(fft_abs_noise)) ,0.1).astype(np.float32)
    # fft_angle_spider = ((fft_angle_noise + 2 * np.pi) % (2 * np.pi) / (2*np.pi)).astype(np.float32)

    # fft_abs_spider[0][0] = 1
    # fft_angle_spider[0][0] = 0
    return np.concatenate((fft_abs_noise ,fft_angle_noise), axis=0)
