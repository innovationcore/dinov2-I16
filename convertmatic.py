import torch
from transformers.models import vits

from dinov2.models.vision_transformer import DinoVisionTransformer, vit_small
from transformers import Dinov2Model
from dinov2.layers import MemEffAttention, NestedTensorBlock as Block
from functools import partial


pretrained_weights = 'teacher_checkpoint.pth'
state_dict = torch.load(pretrained_weights, map_location="cpu")
state_dict = state_dict['teacher']

model = vit_small(
img_size=518,
patch_size=16,
num_register_tokens=0)
model.load_state_dict(state_dict)

model = vits.dict['small'](
img_size=518,
patch_size=patch_size,
init_values=1.0,
block_chunks=0,
num_register_tokens=self.registers)


'''
# only the backbone keys
state_dict = {key: value for key, value in state_dict.items() if 'backbone' in key}
# remove 'backbone.' prefix
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

model = vit_small(patch_size=16)

# load state_dict into the model
model.load_state_dict(state_dict)

model.save_pretrained(pretrained_weights)

model.load_state_dict(state_dict)
'''