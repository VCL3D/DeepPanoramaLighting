import argparse
import os
import cv2
import sys
import numpy as np
import torch
import torchvision
from helpers.sh_functions import *
from loaders.Illum_loader import IlluminationModule, Inference_Data
from loaders.autoenc_ldr2hdr import LDR2HDR    
from torch.utils.data import DataLoader

def parse_arguments(args):
    usage_text = (
        "Inference script for Deep Lighting Environment Map Estimation from Spherical Panoramas"
        "Usage:  python3 inference.py --input_path "
    )
    parser = argparse.ArgumentParser(description=usage_text)    
    parser.add_argument('--input_path', type=str, default='./images/input.jpg', help="Input panorama color image file")
    parser.add_argument('--out_path', type=str, default='./output/', help='Output folder for the predicted environment map panorama')
    parser.add_argument('-g','--gpu', type=str, default='0', help='GPU id of the device to use. Use -1 for CPU.')
    parser.add_argument("--chkpnt_path", default='./models/model.pth', type=str, help='Pre-trained checkpoint file for lighting regression module')    
    parser.add_argument('--ldr2hdr_model', type=str, default='./models/ldr2hdr.pth', help='Pre-trained checkpoint file for ldr2hdr image translation module')
    parser.add_argument("--width", type=float, default=512, help = "Spherical panorama image width.")
    parser.add_argument('--deringing', type=int, default=0, help='Enable low pass deringing filter for the predicted SH coefficients')    
    parser.add_argument('--dr_window', type=float, default='6.0')
    return parser.parse_known_args(args)

def evaluate(
    illumination_module: torch.nn.Module,
    ldr2hdr_module: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device
):
    if (os.path.isdir(args.out_path)!=True):
        os.mkdir(args.out_path)
    
    in_filename, in_file_extention = os.path.splitext(args.input_path) 
    assert in_file_extention in ['.png','.jpg']
    inference_data = Inference_Data(args.input_path)
    out_path = args.out_path + os.path.basename(args.input_path)
    out_filename, out_file_extension = os.path.splitext(out_path)
    out_file_extension = '.exr'
    out_path = out_filename + out_file_extension
    dataloader = DataLoader(inference_data, batch_size=1, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader):
        input_img = data.to(device).float()
        with torch.no_grad():            
            start_time = time.time()
            right_rgb = ldr2hdr_module(input_img)
            p_coeffs = illumination_module(right_rgb).view(1,9,3).to(device).float()
            if args.deringing:
                p_coeffs = deringing(p_coeffs, args.dr_window).to(device).float()
            elapsed_time = time.time() - start_time
            print("Elapsed inference time: %2.4fsec" % elapsed_time)
            pred_env_map = shReconstructSignal(p_coeffs.squeeze(0), width=args.width)            
            cv2.imwrite(out_path, pred_env_map.cpu().detach().numpy())

def main(args):
    device = torch.device("cuda:" + str(args.gpu) if (torch.cuda.is_available() and int(args.gpu) >= 0) else "cpu")    
    # load lighting module
    illumination_module = IlluminationModule(batch_size=1).to(device)
    illumination_module.load_state_dict(torch.load(args.chkpnt_path))
    print("Lighting moduled loaded")
    # load LDR2HDR module     
    ldr2hdr_module = LDR2HDR()
    ldr2hdr_module.load_state_dict(torch.load(args.ldr2hdr_model)['state_dict_G'])
    ldr2hdr_module = ldr2hdr_module.to(device)
    print("LDR2HDR moduled loaded")
    evaluate(illumination_module, ldr2hdr_module, args, device)
    
if __name__ == '__main__':
    args, unknown = parse_arguments(sys.argv)
    main(args)