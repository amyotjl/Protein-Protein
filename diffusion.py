import torch
from torch.utils.data import Dataset
from torch import Tensor
from denoising_diffusion_pytorch import Trainer1D, Unet1D, GaussianDiffusion1D
from load_dataset import get_input_seqs_dataloader
import joblib
import glob
import os
import re
import hashlib
import json
import wandb

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
        'seq_length': 1024,
        'timesteps' : 200,
        'sampling_timesteps' : None,
        'loss_type' : 'l1',
        'objective' : 'pred_noise',
        'beta_schedule' : 'cosine',
        'ddim_sampling_eta' : 0.,
        'auto_normalize' : True
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
        # 'nw':4        
    }
}

def without_keys(d, keys):
    return {k: d[k] for k in d.keys() - keys}
conf_to_hash_unet = without_keys(conf['unet'], [])
conf_to_hash_gaussian = without_keys(conf['gaussian'], [])
conf_to_hash_trainer = without_keys(conf['trainer'], ['train_batch_size', 'train_num_steps', 'save_and_sample_every', 'num_samples'])

folder_name ='diffusion/{}'.format(hash_dictionary({**conf_to_hash_unet, **conf_to_hash_gaussian,**conf_to_hash_trainer}))
os.makedirs(folder_name, exist_ok=True)

with open('{}/config.json'.format(folder_name), 'w') as file:
    json.dump(conf, file)
print(folder_name)

class Dataset1D(Dataset):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()

wandb.init()
model = Unet1D(
    **conf['unet']
).cuda()

diffusion = GaussianDiffusion1D(
    model,
    **conf['gaussian']
).cuda()

dataset = joblib.load('data/latent_dim_500.joblib')

dataset_train = Dataset1D(dataset['inputs'].detach())


trainer = Trainer1D(
    diffusion,
    dataset = dataset_train,
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