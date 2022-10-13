import argparse

def get_opt():
# Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=31, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="spiderdata_p5e-3_o0", help="name of the dataset") # v9_uc_f2a_3angle_noise
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches") 
    
    parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")#0.0002
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    
    parser.add_argument("--load_best", type=bool, default=True, help="load the best generator weights")
    parser.add_argument("--lr_policy", type=str, default='step', help="epoch from which to start lr decay")
    parser.add_argument("--step_size", type=int, default=200, help="epoch from which to start lr decay")
    parser.add_argument("--gamma", type=int, default=0.1, help="epoch from which to start lr decay")
    parser.add_argument("--DP", type=bool, default=False, help="Distributed training")
    
    # parser.add_argument("--GAN_lambda", type=float, default=1e-6, help="adam: learning rate")#0.0002
    parser.add_argument("--img_lambda", type=float, default=1e-2, help="adam: learning rate")#0.0002
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height") # 256
    parser.add_argument("--img_width", type=int, default=256, help="size of image width") # 256
    # parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    
    parser.add_argument(
        "--sample_interval", type=int, default=500, help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
    
    opt = parser.parse_args()
    return opt, parser
