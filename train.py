# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import numpy as np
import time
import datetime
import sys

import torch
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset.datasets import *
from model.UNet_Nested import *
from options import *
from loss.loss import *
from loss.compound_loss import *

from utils.psnr_ssim import *
from utils.img_process_function import *
from utils.weight_init import weights_init_normal
from utils.img_save_function import saveImg
from utils.lr_strategy import *

# Corrupted image file processing
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Choose device 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

opt, parser = get_opt()

# make directorys
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("tensorboard/%s" % opt.dataset_name, exist_ok=True)

# Use tensorborad
writer = SummaryWriter('./tensorboard/%s' % opt.dataset_name)

# difine loss function
criterion_GAN = torch.nn.MSELoss()
# Compound_Loss = CompoundLoss()
criterion_pixelwise = torch.nn.L1Loss()
Perloss = PerceptualLoss(torch.nn.MSELoss())

# Initialize generator and discriminator
generator = UNet_Nested(in_channels=2, out_channels=1)
discriminator = Discriminator(in_channels=1)

# Set up a pretrained model 
Fmodel = UNet_Nested(in_channels=6, out_channels=2)
Smodel = UNet_Nested(in_channels=3, out_channels=1)

# Use Cuda
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    # Compound_Loss.cuda()
    Fmodel = Fmodel.cuda()
    Smodel = Smodel.cuda() 

# Load pretrained models
Fmodel.load_state_dict(torch.load("Fmodel/saved_models/%s/best_Fmodel.pth"% opt.dataset_name))
Smodel.load_state_dict(torch.load("Smodel/saved_models/%s/best_Smodel.pth"% opt.dataset_name))

if opt.load_best == True:
    generator.load_state_dict(torch.load("saved_models/%s/best_generator.pth" % (opt.dataset_name)))
    discriminator.load_state_dict(torch.load("saved_models/%s/best_discriminator.pth" % (opt.dataset_name)))
elif opt.epoch != 0:
    generator.load_state_dict(torch.load("saved_models/%s/generator_last.pth" % (opt.dataset_name)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_last.pth" % (opt.dataset_name)))
else:
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# lr_schedul
scheduler_G = get_scheduler(optimizer_G, opt)
scheduler_D = get_scheduler(optimizer_D, opt)

# Configure dataloaders
transforms_ = [
    transforms.ToTensor(),
    ]

dataloader = DataLoader(
    ImageDataset(r"/data2/zhangzr/SpiderDataset/%s" % opt.dataset_name, transforms_=transforms_, mode="train"),
    batch_size=opt.batch_size,
    num_workers=opt.n_cpu,
    shuffle=True,
    drop_last=True,)

val_dataloader = DataLoader(
    ImageDataset(r"/data2/zhangzr/SpiderDataset/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=1,
    num_workers=1,
    shuffle=True,
    drop_last=True,)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)
# Adversarial ground truths
valid = Variable(Tensor(np.ones((opt.batch_size,*patch))), requires_grad=False)
fake = Variable(Tensor(np.zeros((opt.batch_size,*patch))), requires_grad=False)

# ----------
#  Training
# ----------
if __name__ == '__main__':
    prev_time = time.time()
    evalbest = 0
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            
            # ---------------------
            #    Train  Generator
            # ---------------------
            
            # Model inputs
            spider_fft = batch["A"].to(device)
            real_fft = batch["B"].to(device) 
            
            real_img = freq2img1(real_fft).to(device)
            
            Fout = Fmodel(spider_fft)
            Fout = freq2imgnormal(Fout)

            # spider_img = freq2img3(spider_fft,opt.batch_size).to(device)
            
            spider_img = freq2img3(spider_fft).to(device)
            Sout = Smodel(spider_img)
            # Sout = img2fft2(Sout )
            
            spider_cat = torch.cat((Fout,Sout),1)            
            spider_cat = torch.where(torch.isnan(spider_cat), torch.full_like(spider_cat, 0), spider_cat)
 
            optimizer_G.zero_grad()            
            output_img = generator(spider_cat)

            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(output_img, real_img)
            # loss_pixel = Compound_Loss(output_img, real_img)

            pred_fake = discriminator(output_img)
            loss_GAN = criterion_GAN(pred_fake, valid)

            loss_Perceptual = Perloss.get_loss( output_img, real_img)
            # FFT loss
            if epoch > 200:
                loss_G = loss_pixel + opt.Per_lambda * loss_Perceptual + opt.GAN_lambda * loss_GAN 
                # loss_G = loss_pixel + opt.GAN_lambda * loss_GAN
            else :
                loss_G = loss_pixel 
                
            loss_G.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            pred_real = discriminator(real_img)
            loss_real = criterion_GAN(pred_real, valid)
            
            pred_fake = discriminator(output_img.detach())
            loss_fake = criterion_GAN(pred_fake, fake)
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake) 
            
            if epoch > 200:
                loss_D.backward()
                
            optimizer_D.step()
            
            
            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            lr_G = optimizer_G.param_groups[0]['lr']
            lr_D = optimizer_D.param_groups[0]['lr']
            writer.add_scalar("loss_G",loss_G.item(),batches_done)
            writer.add_scalar("loss_D",loss_D.item(),batches_done)
            writer.add_scalar("loss_GAN",loss_GAN.item(),batches_done)
            writer.add_scalar("loss_pixel",loss_pixel.item(),batches_done) 
            writer.add_scalar("lr_G",lr_G,batches_done)
            writer.add_scalar("lr_D",lr_D,batches_done)
            
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Gen loss: %f] [Dis loss: %f] [GAN loss: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_G.item(),
                    loss_D.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )
            
            if batches_done % opt.sample_interval == 0:
                # Results of training
                train_img_sample = torch.cat((Fout.data, Sout.data, output_img.data,real_img.data), -1)
                saveImg(train_img_sample, "images/%s/"% (opt.dataset_name),epoch,"train_%s.png"% ( batches_done), Gray=True)
                
                val_img_sample = next(iter(val_dataloader))
                val_img_sample_spider_fft = val_img_sample["A"].to(device)
                val_img_sample_real_fft = val_img_sample["B"].to(device)
                
                val_img_sample_real = freq2img1(val_img_sample_real_fft).to(device)
            
                val_img_sample_Fout_before = Fmodel(val_img_sample_spider_fft)
                val_img_sample_Fout = freq2imgnormal(val_img_sample_Fout_before)
                
                val_img_sample_Sout_before = freq2img3(val_img_sample_spider_fft).to(device)
                val_img_sample_Sout = Smodel(val_img_sample_Sout_before)
                
                val_img_sample_cat = torch.cat((val_img_sample_Fout,val_img_sample_Sout),1)
                val_img_sample_output = generator(val_img_sample_cat)
                
                # Results of the test set
                val_img_sample = torch.cat((val_img_sample_Fout.data, val_img_sample_Sout.data, val_img_sample_output.data, val_img_sample_real.data), -1)
                saveImg(val_img_sample, "images/%s/"% (opt.dataset_name),epoch,"val_%s.png"% ( batches_done), Gray=True)
                
                # Fourier spectrum
                # abs
                show_spider_fft_abs = val_img_sample_spider_fft[:,0:1,:,:]
                show_Fout_fft_abs = val_img_sample_Fout_before[:,0:1,:,:]
                show_Sout_fft_abs =  fft2_show_abs_normal(val_img_sample_Sout[:,0:1,:,:])
                show_out_fft_abs = fft2_show_abs_normal(val_img_sample_output[:,0:1,:,:] )
                show_real_fft_abs = val_img_sample_real_fft[:,0:1,:,:] 
                fft_img_sample_abs = torch.cat((show_spider_fft_abs.data, show_Fout_fft_abs.data, show_Sout_fft_abs.data,show_out_fft_abs.data ,show_real_fft_abs.data), -1)
                # angle
                show_spider_fft_angle = val_img_sample_spider_fft[:,1:2,:,:]
                show_Fout_fft_angle = val_img_sample_Fout_before[:,1:2,:,:]
                show_Sout_fft_angle =  fft2_show_angle_normal(val_img_sample_Sout[:,0:1,:,:])
                show_out_fft_angle = fft2_show_angle_normal(val_img_sample_output[:,0:1,:,:] )
                show_real_fft_angle = val_img_sample_real_fft[:,1:2,:,:] 
                fft_img_sample_angle = torch.cat((show_spider_fft_angle.data, show_Fout_fft_angle.data, show_Sout_fft_angle.data,show_out_fft_angle.data ,show_real_fft_angle.data), -1)

                fft_img_sample =  torch.cat((fft_img_sample_abs ,fft_img_sample_angle),-2)
                saveImg(fft_img_sample, "images/%s/"% (opt.dataset_name),epoch,"fft_%s.png"% ( batches_done), Gray=True)
                             
        # Update learning rate
        scheduler_G.step()
        scheduler_D.step()

        # Save the weight
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_last.pth" % (opt.dataset_name))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_last.pth" % (opt.dataset_name))
        
        # Save the weight
        if epoch == 200:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_200.pth" % (opt.dataset_name))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_200.pth" % (opt.dataset_name))
        
        # val all
        PSNR_epoch = []
        SSIM_epoch = []
        for i, val_img in enumerate(val_dataloader):

            spider_img_val_fft = val_img["A"].to(device)
            real_img_val_fft = val_img["B"].to(device)
            real_img_val = freq2img1(real_img_val_fft).to(device)
            
            Fout_val = Fmodel(spider_img_val_fft)
            Fout_val = freq2imgnormal(Fout_val )
            
            Sout_val = freq2img3(spider_img_val_fft).to(device)
            Sout_val = Smodel(Sout_val)
            
            cat_val = torch.cat((Fout_val,Sout_val),1)
            output_val = generator(cat_val)
            
            PSNR_epoch_i = PSNR( output_val.data.cpu().numpy()*255.0, real_img_val.data.cpu().numpy()*255.0 )
            SSIM_epoch_i = SSIM( output_val.data.cpu()*255.0, real_img_val.data.cpu()*255.0 ).numpy()
 
            # Exclude nan cases
            if PSNR_epoch_i > 0 and SSIM_epoch_i > 0:
                PSNR_epoch.append(PSNR_epoch_i) 
                SSIM_epoch.append(SSIM_epoch_i)
                   
        PSNR_epoch = sum(PSNR_epoch)/len(PSNR_epoch)
        SSIM_epoch = sum(SSIM_epoch)/len(SSIM_epoch)
        
        writer.add_scalar("PSNR_epoch",PSNR_epoch,epoch)
        writer.add_scalar("SSIM_epoch",SSIM_epoch,epoch)
        
        if (PSNR_epoch + 50 * SSIM_epoch) > evalbest:
            evalbest = PSNR_epoch + 50 * SSIM_epoch
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/best_generator.pth" % (opt.dataset_name))
            torch.save(discriminator.state_dict(), "saved_models/%s/best_discriminator.pth" % (opt.dataset_name))
            
        
