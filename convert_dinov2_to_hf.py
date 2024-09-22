import os
from pathlib import Path
import numpy as np
import requests
import torch
import yaml
from PIL import Image
from numpy import asarray
from torchvision import transforms
from transformers import BitImageProcessor, Dinov2Config, Dinov2Model
from transformers.image_utils import PILImageResampling
from functools import partial
from dinov2.models.vision_transformer import DinoVisionTransformer
from dinov2.layers import MemEffAttention, NestedTensorBlock as Block

def get_model(model_config):

    patch_size = model_config['student']['patch_size']
    in_chans = model_config['train']['in_chans']
    embed_dim = model_config['dino']['head_bottleneck_dim']
    img_size = model_config['crops']['global_crops_size']
    ffn_bias = model_config['student']['ffn_bias']
    ffn_layer = model_config['student']['ffn_layer']
    num_register_tokens = model_config['student']['num_register_tokens']
    block_chunks = model_config['teacher']['block_chunks']
    arch = model_config['student']['arch']
    mlp_ratio = 4
    init_values = 1.0

    depth = None
    num_heads = None

    if arch == 'vit_small':
        depth = 12
        num_heads = 6
    elif arch == 'vit_base':
        depth = 12
        num_heads = 12
    elif arch == 'vit_large':
        depth = 24
        num_heads = 16
    elif arch == 'vit_giant2':
        depth = 40
        num_heads = 24

    model = DinoVisionTransformer(
        depth=depth,
        mlp_ratio=mlp_ratio,
        init_values=init_values,
        num_heads=num_heads,
        ffn_layer=ffn_layer,
        ffn_bias=ffn_bias,
        img_size=img_size,
        embed_dim=embed_dim,
        patch_size=patch_size,
        in_chans=in_chans,
        num_register_tokens=num_register_tokens,
        block_chunks=block_chunks,
        block_fn=partial(Block, attn_class=MemEffAttention),
    )

    mlp_index = int(depth/mlp_ratio)

    return model, mlp_index

def prepare_img(config):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    if config.num_channels == 1:
        image = Image.open(requests.get(url, stream=True).raw).convert("I;16")
    else:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    return image

def modify_header(teacher_dict):
    renamed_teacher_dict = {}

    for k in teacher_dict.keys():

        if 'backbone' in k:
            new_key = k.replace('backbone.', '')

            #print(k)
            #if "vits" == model_type:
            #match = re.search(r'blocks\.(\d+)\.(\d+)\.', new_key)
            #if match:
            #    new_key = new_key.replace(f'blocks.{match.group(1)}.{match.group(2)}.', f'blocks.{match.group(2)}.')

            renamed_teacher_dict[new_key] = teacher_dict[k]
    return renamed_teacher_dict

def get_config():

    model_config = None

    if os.path.isfile(args.model_config):
        with open(args.model_config, 'r') as file:
            model_config = yaml.safe_load(file)

    return model_config

def get_dinov2_config(model_config):

    patch_size = model_config['student']['patch_size']
    in_chans = model_config['train']['in_chans']
    embed_dim = model_config['dino']['head_bottleneck_dim']
    img_size = model_config['crops']['global_crops_size']
    arch = model_config['student']['arch']

    config = Dinov2Config(
        hidden_size=embed_dim,
        image_size=img_size,
        patch_size=patch_size,
        num_channels=in_chans
    )

    if arch == 'vit_small':
        config.num_attention_heads = 6
    elif arch == 'vit_base':
        pass
    elif arch == 'vit_large':
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    elif arch == 'vit_giant2':
        #config.use_swiglu_ffn = True
        config.num_hidden_layers = 40
        config.num_attention_heads = 24
    else:
        raise ValueError('Model', arch, 'not supported')

    return config

def create_rename_keys(config, model_config, mlp_index):

    #head_nlayers = model_config['dino']['head_nlayers']

    rename_keys = []
    # fmt: off

    # patch embedding layer
    rename_keys.append(("cls_token", "embeddings.cls_token"))
    rename_keys.append(("mask_token", "embeddings.mask_token"))
    rename_keys.append(("pos_embed", "embeddings.position_embeddings"))
    rename_keys.append(("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"))

    i = 0
    ii_count = 0
    for ii in range(config.num_hidden_layers):
        # layernorms
        rename_keys.append((f"blocks.{i}.{ii}.norm1.weight", f"encoder.layer.{ii}.norm1.weight"))
        rename_keys.append((f"blocks.{i}.{ii}.norm1.bias", f"encoder.layer.{ii}.norm1.bias"))
        rename_keys.append((f"blocks.{i}.{ii}.norm2.weight", f"encoder.layer.{ii}.norm2.weight"))
        rename_keys.append((f"blocks.{i}.{ii}.norm2.bias", f"encoder.layer.{ii}.norm2.bias"))
        # MLP
        if config.use_swiglu_ffn:
            rename_keys.append((f"blocks.{i}.{ii}.mlp.w12.weight", f"encoder.layer.{ii}.mlp.w12.weight"))
            rename_keys.append((f"blocks.{i}.{ii}.mlp.w12.bias", f"encoder.layer.{ii}.mlp.w12.bias"))
            rename_keys.append((f"blocks.{i}.{ii}.mlp.w3.weight", f"encoder.layer.{ii}.mlp.w3.weight"))
            rename_keys.append((f"blocks.{i}.{ii}.mlp.w3.bias", f"encoder.layer.{ii}.mlp.w3.bias"))
        else:
            rename_keys.append((f"blocks.{i}.{ii}.mlp.fc1.weight", f"encoder.layer.{ii}.mlp.fc1.weight"))
            rename_keys.append((f"blocks.{i}.{ii}.mlp.fc1.bias", f"encoder.layer.{ii}.mlp.fc1.bias"))
            rename_keys.append((f"blocks.{i}.{ii}.mlp.fc2.weight", f"encoder.layer.{ii}.mlp.fc2.weight"))
            rename_keys.append((f"blocks.{i}.{ii}.mlp.fc2.bias", f"encoder.layer.{ii}.mlp.fc2.bias"))
        # layerscale
        rename_keys.append((f"blocks.{i}.{ii}.ls1.gamma", f"encoder.layer.{ii}.layer_scale1.lambda1"))
        rename_keys.append((f"blocks.{i}.{ii}.ls2.gamma", f"encoder.layer.{ii}.layer_scale2.lambda1"))
        # attention projection layer
        rename_keys.append((f"blocks.{i}.{ii}.attn.proj.weight", f"encoder.layer.{ii}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.{ii}.attn.proj.bias", f"encoder.layer.{ii}.attention.output.dense.bias"))

        #print('ii_count:', ii_count)
        #print('i:', i, 'ii', ii)
        if ii_count == (mlp_index -1):
            ii_count = 0
            i += 1
        else:
            ii_count += 1


    # final layernorm
    rename_keys.append(("norm.weight", "layernorm.weight"))
    rename_keys.append(("norm.bias", "layernorm.bias"))

    # fmt: on
    return rename_keys

def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val

def read_in_q_k_v(state_dict, config, model_config, mlp_index):

    #head_nlayers = model_config['dino']['head_nlayers']

    i = 0
    ii_count = 0

    for ii in range(config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.{ii}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.{ii}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"encoder.layer.{ii}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"encoder.layer.{ii}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"encoder.layer.{ii}.attention.attention.key.weight"] = in_proj_weight[
                                                                               config.hidden_size : config.hidden_size * 2, :
                                                                               ]
        state_dict[f"encoder.layer.{ii}.attention.attention.key.bias"] = in_proj_bias[
                                                                             config.hidden_size : config.hidden_size * 2
                                                                             ]
        state_dict[f"encoder.layer.{ii}.attention.attention.value.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"encoder.layer.{ii}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]

        #print('ii_count:', ii_count)
        #print('i:', i, 'ii', ii)
        if ii_count == (mlp_index - 1):
            ii_count = 0
            i += 1
        else:
            ii_count += 1

def convert_teacher_to_pytorch_dinov2(args, model_config):

    pytorch_model, mlp_index = get_model(model_config)

    pytorch_model.eval()

    teacher_dict = torch.load(args.teacher_checkpoint_path, map_location=torch.device('cpu'))['teacher']
    reshaped_teacher_dict = modify_header(teacher_dict=teacher_dict)

    pytorch_model.load_state_dict(reshaped_teacher_dict)
    #pytorch_model_path = str(uuid.uuid4()) + '.pth'
    #torch.save(reshaped_teacher_dict, pytorch_model_path)

    hf_model_dict = pytorch_model.state_dict()
    hf_model_dict_keys = set(hf_model_dict.keys())
    renamed_keys = set(reshaped_teacher_dict.keys())
    non_matching_keys = hf_model_dict_keys.symmetric_difference(renamed_keys)

    for key in non_matching_keys:
        print('Non-matching keys between small model and renamed teacher model:', key)
        exit(0)

    print('Checking shape compatibility between pytorch model and teacher model...')
    shape_mismatch_keys = []
    for key in hf_model_dict_keys.intersection(renamed_keys):
        if hf_model_dict[key].shape != reshaped_teacher_dict[key].shape:
            shape_mismatch_keys.append((key, hf_model_dict[key].shape, reshaped_teacher_dict[key].shape))

    if shape_mismatch_keys:
        print('Shape mismatch found in the following keys:')
        for key, small_shape, renamed_shape in shape_mismatch_keys:
            print(f"Key: {key} | Small model shape: {small_shape} | Renamed teacher shape: {renamed_shape}")
        exit()
    else:
        print("All matching keys have compatible shapes.")

    print('Teacher model converted to pytorch model')

    return pytorch_model, mlp_index

def convert_pytorch_to_hf_dinov2(args, model_config, pytorch_model, mlp_index):

    config = get_dinov2_config(model_config)

    print('Converting PyTorch Dinov2 keys to HF...')

    state_dict = pytorch_model.state_dict()
    rename_keys = create_rename_keys(config, model_config, mlp_index)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, model_config, mlp_index)

    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if "w12" in key:
            key = key.replace("w12", "weights_in")
        if "w3" in key:
            key = key.replace("w3", "weights_out")
        state_dict[key] = val

    model = Dinov2Model(config).eval()
    model.load_state_dict(state_dict)

    # load image
    image = prepare_img(config)

    transformations = transforms.Compose(
        [
            transforms.Resize(config.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomCrop(config.image_size),
            transforms.ToTensor(),
        ]
    )

    original_pixel_values = transformations(image).unsqueeze(0)  # insert batch dimension

    processor = BitImageProcessor(
        size={'height': config.image_size, 'width': config.image_size},
        do_center_crop=False,
        crop_size={'height': config.image_size, 'width': config.image_size},
        resample=PILImageResampling.NEAREST,
        rescale_factor=0.00001525902,
        image_mean=[],
        image_std=[],
        do_convert_rgb=False,
        do_normalize=False,
    )

    image = asarray(image)
    image = np.expand_dims(image, axis=-1)

    pixel_values = processor(image, return_tensors="pt").pixel_values

    try:
        assert torch.allclose(original_pixel_values, pixel_values)
    except Exception as e:
        print(e)

    with torch.no_grad():
        outputs = model(pixel_values, output_hidden_states=True)
        original_outputs = pytorch_model(pixel_values)

    assert outputs.last_hidden_state[:, 0].shape == original_outputs.shape
    assert torch.allclose(outputs.last_hidden_state[:, 0], original_outputs, atol=1e-3)

    if args.output_path is not None:
        Path(args.output_path).mkdir(exist_ok=True)
        print(f"Saving model to {args.output_path}")
        model.save_pretrained(args.output_path)
        print(f"Saving image processor to {args.output_path}")
        processor.save_pretrained(args.output_path)

def main(args):

    model_config = get_config()

    if (model_config is not None) and (os.path.isfile(args.teacher_checkpoint_path)):
        pytorch_model, mlp_index = convert_teacher_to_pytorch_dinov2(args, model_config)
        convert_pytorch_to_hf_dinov2(args, model_config, pytorch_model, mlp_index)

    else:
        if model_config is None:
            print('Model config not provided.')
        if os.path.isfile(args.teacher_checkpoint_path):
            print('No teacher checkpoint found...')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple feature extraction job')
    parser.add_argument('--model_config', type=str, help='Save file name for csv output', default="vits.yaml")
    parser.add_argument('--teacher_checkpoint_path', type=str, help='teacher_checkpoint_path', default="teacher_checkpoint.pth")
    parser.add_argument('--output_path', type=str, help='Save file name for csv output', default="converted_model")

    args = parser.parse_args()

    main(args)


