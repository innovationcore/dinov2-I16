#https://colab.research.google.com/drive/1gHDOi8RL8hHmfAJvF7IqyBF7tmosG0ko#scrollTo=NDH3kjLASrXs
#https://github.com/huggingface/transformers/pull/32905

from transformers import AutoImageProcessor, AutoModel, AutoConfig
from PIL import Image
import requests

#url = 'https://pic.imgdb.cn/item/666aadd5d9c307b7e956dccb.png'
#image = Image.open(requests.get(url, stream=True).raw)
file_path = 'example_dataset/ct_1.png'
image = Image.open(file_path)
#model_name = 'facebook/dinov2-large'
model_name = 'models/vits_70e_hf'
# facebook/dinov2-giant dinov2-large dinov2-base
config = AutoConfig.from_pretrained(model_name)
config.output_attentions = True
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, config=config)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

#2
from einops import reduce
import torch
att_mat = torch.stack(outputs.attentions).squeeze(1)
att_mat = att_mat.cpu().detach()

att_mat = reduce(att_mat, 'b h len1 len2 -> b len1 len2', 'mean')

#3
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

residual_att = torch.eye(att_mat.size(1))
aug_att_mat = att_mat + residual_att
aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

# Recursively multiply the weight matrices
joint_attentions = torch.zeros(aug_att_mat.size())
joint_attentions[0] = aug_att_mat[0]

for n in range(1, aug_att_mat.size(0)):
    joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

# Attention from the output token to the input space.
v = joint_attentions[-1]
grid_size = int(np.sqrt(aug_att_mat.size(-1)))
mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
mask = cv2.resize(mask / mask.max(), image.size)[..., np.newaxis]
result = (mask * image).astype("uint8")

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 12))

ax1.set_title('Original')
ax2.set_title('Attention Mask')
ax3.set_title('Attention Map')
_ = ax1.matshow(image)
_ = ax2.matshow(mask.squeeze())
_ = ax3.matshow(result)

fig.show()

#4
attentions = outputs.attentions[-1]
# last layer attention weights
nh = attentions.shape[1]
attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
# query: cls, key: all other patches, except the cls token
print(attentions.shape)
fix, ax = plt.subplots(4, 4, figsize=(8, 6))
ax = ax.flatten()

attentions = attentions.reshape(nh, 16, 16)
attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=config.patch_size, mode="nearest")[0].detach().numpy()

for i in range(nh):
    ax[i].set_title(f"Head {i}")
    sat = ax[i].matshow(attentions[i], cmap=plt.colormaps["plasma"])
    ax[i].set_xticks(())
    ax[i].set_yticks(())

plt.colorbar(sat, ax=ax.ravel().tolist())
plt.show()

#5
fix, ax = plt.subplots(1,4, figsize=(16,4))
ax = ax.flatten()

ax[0].set_title(f"Original Image")
ax[0].imshow(image)
ax[0].set_xticks(())
ax[0].set_yticks(())

ax[1].set_title(f"Attention Map")
attmap = ax[1].matshow(reduce(attentions, 'he w h -> w h', 'mean'), cmap=plt.colormaps['viridis'])
ax[1].set_xticks(())
ax[1].set_yticks(())

norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
ax[2].set_title(f"$L_2$-Norms")
outlier = ax[2].matshow(torch.norm(outputs[0], 2, dim=-1).squeeze()[1:].reshape((16, 16)).detach().numpy(), cmap=plt.colormaps['viridis'], norm=norm)
ax[2].set_xticks(())
ax[2].set_yticks(())
plt.colorbar(attmap, ax=ax[1])
plt.colorbar(outlier, ax=ax[2])

ax[3].set_title(f"$L_2$-Norm distribution")
ax[3].hist(torch.norm(outputs[0], 2, dim=-1).squeeze()[1:].reshape((16, 16)).detach().numpy().flatten(), bins=20)
plt.show()


