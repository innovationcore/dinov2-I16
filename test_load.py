#curl -L https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth -o dinov2_vitl14_pretrain.pth

import torch
from dinov2.models.vision_transformer import vit_large, vit_small

model = vit_small(
    patch_size=14,
    img_size=518,
    init_values=1.0,
    block_chunks=0
 )
#model.load_state_dict(torch.load('models/dinov2_vits14_pretrain.pth'))
restored_model = torch.load('teacher_checkpoint.pth', map_location="cpu")

print(restored_model)
#model.save_pretrained('pop')


#print(model)
#model = torch.hub.load('dinov2', 'dinov2_vits14', source='local', pretrained=False)
#model = torch.hub.load("facebookresearch/dinov2", 'dinov2_vitl14', pretrained=True)
#model.load_state_dict(torch.load('dinov2_vits14.pth'))