import os
import glob
from statistics import pvariance
import sys
import cv2
sys.path.append(sys.path[0].split("predict")[0])

import torchvision.transforms as transforms
from model.UNet_Nested import *
import numpy as np
from utils.psnr_ssim import *
from utils.img_process_function import *
from utils.img_save_function import saveImg
from utils.lr_strategy import *
from mask import *
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DP = False

# pval = ["5e-3", "5e-2"]
# oval = ["0","6" , "60"]
start = time.clock() #推理计时开始
pval = ["5e-3"]
oval = [ "60"]
print(pval,oval)
for pi in pval:
    for oi in oval:
        # name = "spiderdata_p0_o6"
        name = "spiderdata_p%s_o%s"%(pi,oi)
        print(name)
        generator_model_path = "/workspace/zhangzr/project/oe2022/SFDFNets_Fuse/FFuse/saved_models/%s/best_generator.pth" %(name)
        Fmodel_model_path =    "/workspace/zhangzr/project/oe2022/SFDFNets_Fuse/Fmodel/saved_models/%s/best_Fmodel.pth"%(name)
        # Fmodel_model_path = "/workspace/zhangzr/project/Z2_SFDFNets/Fmodel/saved_models/spiderdata_p5e-3_o60/Fmodel_300.pth"
        Smodel_model_path =    "/workspace/zhangzr/project/oe2022/SFDFNets_Fuse/Smodel/saved_models/%s/best_Smodel.pth"%(name)

        # datapath = "/workspace/zhangzr/dataset/spiderdata_p5e-3_o60/val"
        datapath = "/data2/zhangzr/SpiderDataset/%s/UCtest"%(name)
        # datapath = "/data2/zhangzr/SpiderDataset/spiderdata_p5e-3_o60/UCtest"
        # datapath = "/workspace/zhangzr/dataset/spiderdata_p5e-3_o60/UCtest"
        savename = datapath.split("/")[-2] +"_"+ datapath.split("/")[-1] 

        cuda = True if torch.cuda.is_available() else False
        device = torch.device("cuda" if cuda else "cpu")

        # mask = define_mask(0)

        os.makedirs("images/%s" % savename, exist_ok=True)
        os.makedirs("freq/%s" % savename, exist_ok=True)
        # Initialize generator and discriminator
        generator = UNet_Nested(in_channels=4, out_channels=2)
        # Set up a pretrained model 
        Fmodel = UNet_Nested(in_channels=6, out_channels=2)
        Smodel = UNet_Nested(in_channels=3, out_channels=1)
        if cuda:
            generator = generator.cuda()
            Fmodel    = Fmodel.cuda()
            Smodel    = Smodel.cuda()
            
        if name == "spiderdata_p5e-3_o0" or DP == True:
            generator = torch.nn.DataParallel(generator)
            # Fmodel = torch.nn.DataParallel(Fmodel)
            # Smodel = torch.nn.DataParallel(Smodel)
        
        generator.load_state_dict(torch.load(generator_model_path))
        Fmodel.load_state_dict(torch.load(Fmodel_model_path))
        Smodel.load_state_dict(torch.load(Smodel_model_path))
        


        import xlsxwriter as xw
        workbook   = xw.Workbook("images/%s.xlsx" % (savename) )
        worksheet1 = workbook.add_worksheet("evaluation result")
        worksheet1.activate()  # 激活表
        title      = [ '', 'num','PSNR_Fuse','SSIM_Fuse',"PSNR_Smodel", "SSIM_Smodel" ,"PSNR_Fmodel", "SSIM_Fmodel","PSNR_Spider", "SSIM_Spider"] 
        worksheet1.write_row('A1', title)  

        path_sorted = sorted(glob.glob(datapath + "/*.*"))
        transform   = transforms.Compose([transforms.ToTensor(),])
        path_fullnoise_sorted = sorted(glob.glob(datapath+"all" + "/*.*"))

        i = 2
        for j,img_path in enumerate(path_sorted):
            print(j)
            num        = img_path.split("/")[-1].split(".")[0]
            img_array_ = np.load(img_path)
            img_array  = np.nan_to_num(img_array_)

            # saveImg( np.abs(img_array)  , "freq/%s/"% (savename),num,"%s.png"% ( "abs"), Gray=True)
            # saveImg( ((np.angle(img_array) + 2*np.pi) %  2*np.pi) /2*np.pi , "freq/%s/"% (savename),num,"%s.png"% ( "angle"), Gray=True)
            c, h, w    = img_array.shape
            cv2.imwrite( "freq/%s/"%(savename)+num+"%s.png" % ( "abs_spider"  ) , (img_array[0,:,0:w//2])*255.0 )
            cv2.imwrite( "freq/%s/"% (savename)+num+"%s.png"% ( "angle_spider") , (img_array[1,:,0:w//2])*255.0 )
            cv2.imwrite( "freq/%s/"%(savename)+num+"%s.png" % ( "abs_ground_truth"  ) , (img_array[0,:,w//2:w])*255.0 )
            cv2.imwrite( "freq/%s/"% (savename)+num+"%s.png"% ( "angle_ground_truth") , (img_array[1,:,w//2:w])*255.0 )

            fft_spider = img_array[:,:,0:w//2].transpose(1,2,0)
            fft_real   = img_array[0:2,:,w//2:w].transpose(1,2,0)
            fft_spider = transform(fft_spider).unsqueeze(0).to(device)
            fft_real   = transform(fft_real).unsqueeze(0).to(device)

            real_img   = freq2imgnormal( fft_real ).to(device)[:,:,3:253,3:253]
            saveImg( real_img  , "images/%s/"% (savename),num,"%s.png"% ( "Orin_img"), Gray=True)

            Fout       = Fmodel(fft_spider)
            Fout_img   = freq2imgnormal(Fout)[:,:,3:253,3:253]
            spider_img = freq2img3(fft_spider).to(device)

            # print(Fout.shape,real_img.shape)

            saveImg( Fout[:,0:1,:,:], "freq/%s/"% (savename),"abs_Fmodel","%s.png"% (num ), Gray=True)
            saveImg( Fout[:,1:2,:,:], "freq/%s/"% (savename),"angle_Fmodel","%s.png"% (num ), Gray=True)

            img_array_fullnoise  = np.load(path_fullnoise_sorted[j])
            fft_spider_fullnoise = img_array_fullnoise[:,:,0:w//2].transpose(1,2,0)
            fft_spider_fullnoise = transform(fft_spider_fullnoise).unsqueeze(0).to(device)
            spider_img_fullnoise = freq2imgnormal( fft_spider_fullnoise  )[:,:,3:253,3:253]

            # spider_img_ = freq2imgnormal(fft_3sample)[:,:,3:253,3:253]
            PSNR_Spider = PSNR( spider_img_fullnoise.cpu().numpy()*255.0, real_img.data.cpu().numpy()*255.0 )
            SSIM_Spider = SSIM( spider_img_fullnoise.cpu()*255.0, real_img.data.cpu()*255.0 ).numpy()
            saveImg(spider_img_fullnoise, "images/%s/"% (savename),num,"%s.png"% ("spider_img_fullnoise"), Gray=True)
            
            
            Sout_img   = Smodel(spider_img)
            Sout_img_  = Sout_img[:,:,3:253,3:253]
            Sout       = img2fft2(Sout_img)


            saveImg( Sout[:,0:1,:,:], "freq/%s/"% (savename), "abs_Smodel","%s.png"% (num), Gray=True)
            saveImg( Sout[:,1:2,:,:], "freq/%s/"% (savename), "angle_Smodel","%s.png"% (num ), Gray=True)

            saveImg(Fout_img, "images/%s/"% (savename),num,"%s.png"% ("Fout_img"), Gray=True)
            saveImg(Sout_img_ , "images/%s/"% (savename),num,"%s.png"% ( "Sout_img"), Gray=True)

            PSNR_Smodel = PSNR( Sout_img_.cpu().detach().numpy()*255.0, real_img.data.cpu().numpy()*255.0 )
            SSIM_Smodel = SSIM( Sout_img_.cpu().detach()*255.0, real_img.data.cpu()*255.0 ).numpy()
            PSNR_Fmodel = PSNR( Fout_img.cpu().detach().numpy()*255.0, real_img.data.cpu().numpy()*255.0 )
            SSIM_Fmodel = SSIM( Fout_img.cpu().detach()*255.0, real_img.data.cpu()*255.0 ).numpy()

            spider_cat = torch.cat((Fout,Sout),1)            
            spider_cat = torch.where(torch.isnan(spider_cat), torch.full_like(spider_cat, 0), spider_cat)
            output_fft = generator(spider_cat)

            saveImg( output_fft[:,0:1,:,:], "freq/%s/"% (savename), "abs_Fuse","%s.png"% (num), Gray=True)
            saveImg( output_fft[:,1:2,:,:], "freq/%s/"% (savename), "angle_Fuse","%s.png"% (num ), Gray=True)

            output_img = freq2imgnormal(output_fft )[:,:,3:253,3:253]
            saveImg(output_img , "images/%s/"% (savename),num,"%s.png"% ( "Zout_img"), Gray=True)

            PSNR_Fuse  = PSNR( output_img.data.cpu().numpy()*255.0, real_img.data.cpu().numpy()*255.0 )
            SSIM_Fuse  = SSIM( output_img.data.cpu()*255.0, real_img.data.cpu()*255.0 ).numpy()
            print(PSNR_Fuse,SSIM_Fuse)
            # PSNR_test = PSNR( real_img.data.cpu().numpy()*200.0, real_img.data.cpu().numpy()*255.0 )
            # SSIM_test = SSIM( real_img.data.cpu()*200.0 , real_img.data.cpu()*255.0 ).numpy()
            # print(PSNR_test,SSIM_test)

            out_put = [savename, num,PSNR_Fuse ,SSIM_Fuse,PSNR_Smodel, SSIM_Smodel ,PSNR_Fmodel, SSIM_Fmodel, PSNR_Spider, SSIM_Spider]
            row = 'A' + str(i)
            try:
                worksheet1.write_row(row, out_put)
                i += 1
            except:
                # print("error!!!!!!!!!!!!!!!!!!!!!!")
                pass
        workbook.close()  

end = time.clock()  #计时结束
#print('pred:{}'.format(pred))
print('infer_time:', end-start)




        # print(dataset)
        # # for i in range(1,1650):
        # #     print(i)
        # #     pic_id = str(i).zfill(6)+ ".npy"
        # #     img_path = "/workspace/zhangzr/project/Spider_Dataset/v10_uvdataset_nofftshift/val/"+ pic_id
        # #     num = img_path.split("/")[-1].split(".")[0]
        # #     transforms_ = [
        # #         transforms.ToTensor(),
        # #     ]
        # #     transform = transforms.Compose(transforms_)
        # #     img_array_ = np.load(img_path)
        # #     img_array = np.nan_to_num(img_array_)
        # #     c, h, w = img_array.shape
        # #     img_A = img_array[:,:,0:int(w/2)].transpose(1,2,0)
        # #     img_B = img_array[0:2,:,int(w/2):w].transpose(1,2,0)
        # #     img_A = transform(img_A).unsqueeze(0) 
        # #     img_B = transform(img_B).unsqueeze(0) 

        # #     spider_img = img_A.to(device)
        # #     real_img = img_B.to(device)
        # #     output_img = (generator(spider_img))
            
        # #     im_lr_fft = torch.fft.fftshift(real_img[:,0:1,:,:]* mask1,dim=(2,3))
        # #     im_lr_fft[0,0,129,129]=1
        # #     im_hr_fft = torch.fft.fftshift(real_img[:,0:1,:,:],dim=(2,3))
        # #     im_out_fft = torch.fft.fftshift(output_img[:,0:1,:,:],dim=(2,3))
        # #     # im_out_fft = (im_out_fft - torch.min(im_out_fft))/(torch.max(im_out_fft) - torch.min(im_out_fft) )
        # #     img_sample_ = torch.cat((im_lr_fft.data, im_out_fft.data, im_hr_fft.data), -2)  
        # #     saveImg(img_sample_, "images/%s/"% (savename),epoch,"fft_%s.png"% ( num), Gray=True)

        # #     def freq2img_crop_normal(img_freq):
        # #         fft_abs = torch.pow(img_freq[:,0:1,:,:], 10)
        # #         fft_angle = img_freq[:,1:2,:,:] * 2 * np.pi
        # #         img_fft = fft_abs * torch.cos(fft_angle) + 1j * fft_abs * torch.sin(fft_angle)
        # #         img_normal = torch.abs(torch.fft.ifftn(img_fft,dim=(2,3))) 
        # #         img_normal_crop = img_normal[:,:,3:253,3:253]
        # #         img = (img_normal_crop - torch.min(img_normal_crop))/(torch.max(img_normal_crop) - torch.min(img_normal_crop) )
        # #         return img
            
        # #     def spider3out(img_freq):
        # #         # fft_img = torch.fft.ifftn(img_freq ,dim=(2,3))
        # #         fft_abs = torch.pow(img_freq[:,0:1,:,:], 10)
        # #         fft_angle = img_freq[:,1:2,:,:] * 2 * np.pi
        # #         img_fft = fft_abs * torch.cos(fft_angle) + 1j * fft_abs * torch.sin(fft_angle)
        # #         fft_img_mask = img_fft * mask1
        # #         img_normal = torch.abs(torch.fft.ifftn(fft_img_mask ,dim=(2,3)))[:,:,3:253,3:253]
        # #         img_normal_3 = (img_normal - torch.min(img_normal))/(torch.max(img_normal) - torch.min(img_normal) )
        # #         return  img_normal_3

        # #     spider_img_out =  spider3out(real_img)
        # #     real_img_out = freq2img_crop_normal(real_img)
        # #     outimg_out = freq2img_crop_normal(output_img)

        # #     from psnr_ssim import *
        # #     PSNR_iteration = PSNR( outimg_out.data.cpu().numpy()*255.0, real_img_out.data.cpu().numpy()*255.0 )
        # #     SSIM_iteration = SSIM( outimg_out.data.cpu()*255.0, real_img_out.data.cpu()*255.0 ).numpy()
            
        # #     PSNR_iteration_before = PSNR( spider_img_out.data.cpu().numpy()*255.0, real_img_out.data.cpu().numpy()*255.0 )
        # #     SSIM_iteration_before = SSIM( spider_img_out.data.cpu()*255.0, real_img_out.data.cpu()*255.0 ).numpy()
            
        # #     img_sample = torch.cat((spider_img_out.data, outimg_out.data, real_img_out.data), -2)
        # #     saveImg(img_sample, "images/%s/"% (savename),type,"%s.png"% ( num), Gray=True)
        # #     # saveImg(spider_img_out.data, "./",type,"%s.png"% ( num), Gray=True)
        # #     # break
        # #     # saveImg(img_sample, "images/%s/"% (savename),type,"%s_%s_%s.png"% ( num,int(PSNR_iteration),int(SSIM_iteration)), Gray=True)
        # #     # print("psnr:",PSNR_iteration)
        # #     # print("ssim:",SSIM_iteration)

        # #     out_put = [type, num,PSNR_iteration,SSIM_iteration,PSNR_iteration_before, SSIM_iteration_before]
        # #     row = 'A' + str(i+1)
        # #     try:
        # #         worksheet1.write_row(row, out_put)
        # #     except:
        # #         print("error!!!!!!!!!!!!!!!!!!!!!!")
        # #         pass
        # # workbook.close() 
