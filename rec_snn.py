import torch
import os
import numpy as np
from PIL import Image
from os.path import splitext
from utils.util import events_to_voxel_grid, normalize_image, CropParameters, Timer
from model.snn_network import EVSNN_LIF_final, PAEVSNN_LIF_AMPLIF_final
import argparse
import pandas as pd

torch.backends.cudnn.benchmark = True
_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

    #model_name, pretrain_models, event_files, save_path, height, width
def main(model_name:str, pretrain_models:str, event_files:str, save_path:str, height:int, width:int, num_events_per_pixel:float):
    network_kwargs = {'activation_type': 'lif',
                      'mp_activation_type': 'amp_lif',
                      'spike_connection': 'concat',
                      'num_encoders': 3,
                      'num_resblocks': 1,
                      'v_threshold': 1.0,
                      'v_reset': None,
                      'tau': 2.0
                      }
    device = 'cuda:0'

    net = eval(model_name)(kwargs = network_kwargs).to(device)
    net.load(pretrain_models)

    crop = CropParameters(width, height, 3, 0)

    savepath = os.path.join(save_path, model_name)
    if os.path.isdir(savepath): 
        pass
    else:                                                                                       
        os.mkdir(savepath)  
    
    N = int(height*width*num_events_per_pixel)
    event_tensor_iterator = pd.read_csv(event_files, delim_whitespace=True, header=None, names=['t', 'x', 'y', 'pol'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                    engine='c',
                                    skiprows=10, chunksize=N, nrows=None)

    out_pattern_img = 'result-idx{:04d}{:04d}.bmp'
    states = None
    i = 0
    j = 0
    num_bins = 5
    for event in event_tensor_iterator:
        event_tensor = events_to_voxel_grid(event.values,
                                            num_bins=num_bins,
                                            width=width,
                                            height=height)
        event_tensor = torch.from_numpy(event_tensor)
        
        event_tensor = event_tensor[np.newaxis,:,:,:].to(device)
        event_tensor = crop.pad(event_tensor)
        mean, stddev = event_tensor[event_tensor != 0].mean(), event_tensor[event_tensor != 0].std()
        event_tensor[event_tensor != 0] = (event_tensor[event_tensor != 0] - mean) / stddev

        for j in range(num_bins):
            event_input = event_tensor[:,j,:,:].unsqueeze(dim=1)
            with torch.no_grad():
                if model_name == 'EVSNN_LIF_final':
                    membrane_potential = net(event_input, states)
                    states = membrane_potential
                elif model_name == 'PAEVSNN_LIF_AMPLIF_final':
                    membrane_potential, states = net(event_input, states)

            result = (membrane_potential[0, 0, crop.iy0:crop.iy1,crop.ix0:crop.ix1].cpu()).detach().numpy().reshape(height, width)
            result = normalize_image(result)
            img = Image.fromarray(result*255)
            img=img.convert("L")
            #img = img.rotate(180,expand=True)
            img.save(os.path.join(savepath, out_pattern_img.format(i,j)))

        print('\rProcessing: {}.'.format(i*num_bins), end='', flush=True)
        i = i + 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-network', type=str, default = 'EVSNN_LIF_final') #EVSNN_LIF_final or PAEVSNN_LIF_AMPLIF_final
    parser.add_argument('-path_to_pretrain_models', type=str, default = './pretrained_models/EVSNN.pth')  # 
    parser.add_argument('-path_to_event_files', type=str, default = './data/poster_6dof_cut.txt')
    parser.add_argument('-save_path', type=str, default = './results')
    parser.add_argument('-height', type=int, default = 180)
    parser.add_argument('-width', type=int, default = 240)
    parser.add_argument('-num_events_per_pixel', type=float, default = 0.5)
    args = parser.parse_args()
    model_name = args.network
    pretrain_models = args.path_to_pretrain_models
    event_files = args.path_to_event_files
    save_path = args.save_path
    height = args.height
    width = args.width
    num_events_per_pixel = args.num_events_per_pixel
    main(model_name, pretrain_models, event_files, save_path, height, width, num_events_per_pixel)