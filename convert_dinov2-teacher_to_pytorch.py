import os
import torch
from functools import partial
import yaml

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

    return model


def modify_header(teacher_dict):
    renamed_teacher_dict = {}

    for k in teacher_dict.keys():

        if 'backbone' in k:
            new_key = k.replace('backbone.', '')

            print(k)
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
def main(args):

    model_config = get_config()

    if (model_config is not None) and (os.path.isfile(args.teacher_checkpoint_path)):

        hf_model = get_model(model_config)

        hf_model.eval()

        teacher_dict = torch.load(args.teacher_checkpoint_path, map_location=torch.device('cpu'))['teacher']
        reshaped_teacher_dict = modify_header(teacher_dict=teacher_dict)

        hf_model.load_state_dict(reshaped_teacher_dict)
        print('Model loaded successfully!')
        torch.save(reshaped_teacher_dict, args.dump_path_vit_model)
        hf_model_dict = hf_model.state_dict()

        print(hf_model)

        hf_model_dict_keys = set(hf_model_dict.keys())
        renamed_keys = set(reshaped_teacher_dict.keys())

        non_matching_keys = hf_model_dict_keys.symmetric_difference(renamed_keys)

        print('Non-matching keys between small model and renamed teacher model:')
        for key in non_matching_keys:
            print(key)

        print('Checking shape compatibility between small model and renamed teacher model...')

        shape_mismatch_keys = []
        for key in hf_model_dict_keys.intersection(renamed_keys):
            if hf_model_dict[key].shape != reshaped_teacher_dict[key].shape:
                shape_mismatch_keys.append((key, hf_model_dict[key].shape, reshaped_teacher_dict[key].shape))
            #else:
            #    print('key:', key, 'shape:', str(hf_model_dict[key].shape))

        if shape_mismatch_keys:
            print('Shape mismatch found in the following keys:')
            for key, small_shape, renamed_shape in shape_mismatch_keys:
                print(f"Key: {key} | Small model shape: {small_shape} | Renamed teacher shape: {renamed_shape}")
        else:
            print("All matching keys have compatible shapes.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple feature extraction job')
    parser.add_argument('--model_config', type=str, help='Save file name for csv output', default="vits.yaml")
    parser.add_argument('--teacher_checkpoint_path', type=str, dest='teacher_checkpoint_path', help='teacher_checkpoint_path', default="teacher_checkpoint.pth")
    parser.add_argument('--dump_path_vit_model', type=str, dest='dump_path_vit_model', help='Save file name for csv output', default="dino_vit.pth")

    args = parser.parse_args()

    main(args)
