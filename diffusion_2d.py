import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, einsum, Tensor
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import joblib
import numpy as np
from einops import rearrange
import glob
import os
import re
import hashlib
import json
import wandb
os.environ['KMP_DUPLICATE_LIB_OK']='True'

data = joblib.load('latent_dim_500.joblib')
n, c, width = data['inputs'][:1000].shape
new = data['inputs'][:1000].view(n, c, int(np.sqrt(width)), int(np.sqrt(width)))
del data

def hash_dictionary(dictionary):
    # Convert the dictionary to a JSON string
    json_str = json.dumps(dictionary, sort_keys=True)
    
    # Create a hash object using the SHA256 algorithm
    hash_object = hashlib.sha256(json_str.encode())
    
    # Get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()
    
    return hash_hex

conf = {
    'unet':{
        'dim':4,
        'dim_mults':(1, 2),
        'channels':1,
        'out_dim' : None,
        'self_condition' : False,
        'resnet_block_groups' : 2,
        'learned_variance' : False,
        'learned_sinusoidal_cond' : False,
        'random_fourier_features' : False,
        'learned_sinusoidal_dim' : 16
    },
    'gaussian':{
        'image_size': 32,
        'timesteps' : 200,
        'sampling_timesteps' : None,
        'loss_type' : 'l1', # l1, l2
        'objective' : 'pred_noise', # pred_noise, pred_x0, pred_v
        'beta_schedule' : 'sigmoid', # linear, cosine, sigmoid
        'schedule_fn_kwargs' : dict(),
        'ddim_sampling_eta' : 0.,
        'auto_normalize' : False,
        'min_snr_loss_weight' : False, # https://arxiv.org/abs/2303.09556
        'min_snr_gamma' : 5
    },
    'trainer':{
        'train_batch_size' : 64,
        'gradient_accumulate_every' : 2,
        'train_lr' : 8e-5,
        'train_num_steps' : 5000,
        'ema_update_every' : 10,
        'ema_decay' : 0.995,
        'adam_betas' : (0.9, 0.99),
        'save_and_sample_every' : 500,
        'num_samples' : 1, # must be square
        'amp' : False,
        'fp16' : False,
        'split_batches' : True,
        'calculate_fid' : True,
        'inception_block_idx' : 2048,
        # 'nw':4        
    }
}
wandb.init()
def without_keys(d, keys):
    return {k: d[k] for k in d.keys() - keys}
conf_to_hash_unet = without_keys(conf['unet'], [])
conf_to_hash_gaussian = without_keys(conf['gaussian'], [])
conf_to_hash_trainer = without_keys(conf['trainer'], ['train_batch_size', 'train_num_steps', 'save_and_sample_every', 'num_samples'])

folder_name ='diffusion/{}'.format(hash_dictionary({**conf_to_hash_unet, **conf_to_hash_gaussian,**conf_to_hash_trainer}))
json.dump
print(folder_name)

model = Unet(
    **conf['unet']
).cuda()


diffusion = GaussianDiffusion(
    model,
    **conf['gaussian']
).cuda()


trainer = Trainer(
    diffusion,
    new.detach(),
    results_folder=folder_name,
    **conf['trainer']
)

list_of_files = glob.glob('{}/*'.format(folder_name)) # * means all if need specific format then *.csv
if len(list_of_files)>0:
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    pattern = r"-(.*?)\."
    match = re.search(pattern, latest_file)
    if match:
        result = match.group(1)
    print('Loading checkpoint: ', result)
    trainer.load(result)

trainer.train()
wandb.finish()
