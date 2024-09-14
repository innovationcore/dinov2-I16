import torch
import copy
import json
from functools import partial
import re
from huggingface_hub import hf_hub_download
from transformers import Dinov2Config
from dinov2.models.vision_transformer import DinoVisionTransformer

from transformers import Dinov2Model
from dinov2.layers import MemEffAttention, NestedTensorBlock as Block


def get_model(model_type):
    model = None
    if "vits" in model_type:
        model = DinoVisionTransformer(
            patch_size=16,
            in_chans=1,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            img_size=512,
            ffn_bias=True,
            ffn_layer='mlp',
            block_fn=partial(Block, attn_class=MemEffAttention),
            num_register_tokens=0,
            init_values=1.0,
            block_chunks=0,
        )

    return model


def remove_header(teacher_dict):
    renamed_teacher_dict = {}

    for k in teacher_dict.keys():
        if 'backbone' in k:
            new_key = k.replace('backbone.', '')

            match = re.search(r'blocks\.(\d+)\.(\d+)\.', new_key)
            if match:
                new_key = new_key.replace(f'blocks.{match.group(1)}.{match.group(2)}.', f'blocks.{match.group(2)}.')

            # if 'w12' in new_key:
            #    new_key = new_key.replace('w12', 'fc1')

            # if 'w3' in new_key:
            #    new_key = new_key.replace('w3', 'fc2')

            renamed_teacher_dict[new_key] = teacher_dict[k]
    return renamed_teacher_dict


def main(teacher_checkpoint_path, dump_path_vit_model):
    hf_model = get_model('vits')

    hf_model.eval()

    teacher_dict = torch.load(teacher_checkpoint_path, map_location=torch.device('cpu'))['teacher']
    reshaped_teacher_dict = remove_header(teacher_dict=teacher_dict)

    hf_model.load_state_dict(reshaped_teacher_dict)
    print('Model loaded successfully!')
    torch.save(reshaped_teacher_dict, dump_path_vit_model)
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
    parser.add_argument('--teacher_checkpoint_path', type=str, dest='teacher_checkpoint_path',
                        help='teacher_checkpoint_path', default="teacher_checkpoint.pth")
    parser.add_argument('--dump_path_vit_model', type=str, dest='dump_path_vit_model',
                        help='Save file name for csv output', default="dino_vit_small.pth")

    args = parser.parse_args()
    teacher_checkpoint_path = args.teacher_checkpoint_path
    dump_path_vit_model = args.dump_path_vit_model

    main(teacher_checkpoint_path=teacher_checkpoint_path, dump_path_vit_model=dump_path_vit_model)
