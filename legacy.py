# Copyright (c) SenseTime Research. All rights reserved.

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 
import pickle
import dnnlib
import re
from typing import List, Optional
import torch
import copy
import numpy as np
from torch_utils import misc


#----------------------------------------------------------------------------
## loading torch pkl
def load_network_pkl(f, force_fp16=False):
    data = _LegacyUnpickler(f).load()

    ## We comment out this part, if you want to convert TF pickle, you can use the original script from StyleGAN2-ada-pytorch
    # # Legacy TensorFlow pickle => convert.
    # if isinstance(data, tuple) and len(data) == 3 and all(isinstance(net, _TFNetworkStub) for net in data):
    #     tf_G, tf_D, tf_Gs = data
    #     G = convert_tf_generator(tf_G)
    #     D = convert_tf_discriminator(tf_D)
    #     G_ema = convert_tf_generator(tf_Gs)
    #     data = dict(G=G, D=D, G_ema=G_ema)

    # Add missing fields.
    if 'training_set_kwargs' not in data:
        data['training_set_kwargs'] = None
    if 'augment_pipe' not in data:
        data['augment_pipe'] = None

    # Validate contents.
    assert isinstance(data['G'], torch.nn.Module)
    assert isinstance(data['D'], torch.nn.Module)
    assert isinstance(data['G_ema'], torch.nn.Module)
    assert isinstance(data['training_set_kwargs'], (dict, type(None)))
    assert isinstance(data['augment_pipe'], (torch.nn.Module, type(None)))

    # Force FP16.
    if force_fp16:
        for key in ['G', 'D', 'G_ema']:
            old = data[key]
            kwargs = copy.deepcopy(old.init_kwargs)
            if key.startswith('G'):
                kwargs.synthesis_kwargs = dnnlib.EasyDict(kwargs.get('synthesis_kwargs', {}))
                kwargs.synthesis_kwargs.num_fp16_res = 4
                kwargs.synthesis_kwargs.conv_clamp = 256
            if key.startswith('D'):
                kwargs.num_fp16_res = 4
                kwargs.conv_clamp = 256
            if kwargs != old.init_kwargs:
                new = type(old)(**kwargs).eval().requires_grad_(False)
                misc.copy_params_and_buffers(old, new, require_all=True)
                data[key] = new
    return data 

class _TFNetworkStub(dnnlib.EasyDict):
    pass

class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'dnnlib.tflib.network' and name == 'Network':
            return _TFNetworkStub
        return super().find_class(module, name)

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

# # My extended version of this helper function:
# def _parse_num_range(s):
#     '''
#     Input:
#         s (str): Comma separated string of numbers 'a,b,c', a range 'a-c',
#         or even a combination of both 'a,b-c', 'a-b,c', 'a,b-c,d,e-f,...'
#     Output:
#         nums (list): Ordered list of ascending ints in s, with repeating values deleted
#     '''
#     # Sanity check 0:
#     # In case there's a space between the numbers (impossible due to argparse,
#     # but hey, I am that paranoid):
#     s = s.replace(' ', '')
#     # Split w.r.t comma
#     str_list = s.split(',')
#     nums = []
#     for el in str_list:
#         if '-' in el:
#             # The range will be 'a-b', so we wish to find both a and b using re:
#             range_re = re.compile(r'^(\d+)-(\d+)$')
#             match = range_re.match(el)
#             # We get the two numbers:
#             a = int(match.group(1))
#             b = int(match.group(2))
#             # Sanity check 1: accept 'a-b' or 'b-a', with a<=b:
#             if a <= b: r = [n for n in range(a, b + 1)]
#             else: r = [n for n in range(b, a + 1)]
#             # Use extend since r will also be an array:
#             nums.extend(r)
#         else:
#             nums.append(int(el))
#     # Sanity check 2: delete repeating numbers:
#     nums = list(set(nums))
#     return nums #sorted(nums)

#----------------------------------------------------------------------------
#### loading tf pkl
def load_pkl(file_or_url):
    with open(file_or_url, 'rb') as file:
        return pickle.load(file, encoding='latin1')

#----------------------------------------------------------------------------

### For editing
def visual(output, out_path):
    import torch
    import cv2
    import numpy as np
    output = (output + 1)/2
    output = torch.clamp(output, 0, 1)
    if output.shape[1] == 1:
        output = torch.cat([output, output, output], 1)
    output = output[0].detach().cpu().permute(1,2,0).numpy()
    output = (output*255).astype(np.uint8)
    output = output[:,:,::-1]
    cv2.imwrite(out_path, output)

def save_obj(obj, path):
    with open(path, 'wb+') as f:
        pickle.dump(obj, f, protocol=4)

#----------------------------------------------------------------------------

## Converting pkl to pth, change dict info inside pickle

def convert_to_rgb(state_ros, state_nv, ros_name, nv_name):
    state_ros[f"{ros_name}.conv.weight"] = state_nv[f"{nv_name}.torgb.weight"].unsqueeze(0)
    state_ros[f"{ros_name}.bias"] = state_nv[f"{nv_name}.torgb.bias"].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    state_ros[f"{ros_name}.conv.modulation.weight"] = state_nv[f"{nv_name}.torgb.affine.weight"]
    state_ros[f"{ros_name}.conv.modulation.bias"] = state_nv[f"{nv_name}.torgb.affine.bias"]


def convert_conv(state_ros, state_nv, ros_name, nv_name):
    state_ros[f"{ros_name}.conv.weight"] = state_nv[f"{nv_name}.weight"].unsqueeze(0)
    state_ros[f"{ros_name}.activate.bias"] = state_nv[f"{nv_name}.bias"]
    state_ros[f"{ros_name}.conv.modulation.weight"] = state_nv[f"{nv_name}.affine.weight"]
    state_ros[f"{ros_name}.conv.modulation.bias"] = state_nv[f"{nv_name}.affine.bias"]
    state_ros[f"{ros_name}.noise.weight"] = state_nv[f"{nv_name}.noise_strength"].unsqueeze(0)


def convert_blur_kernel(state_ros, state_nv, level):
    """Not quite sure why there is a factor of 4 here"""
    # They are all the same
    state_ros[f"convs.{2*level}.conv.blur.kernel"] = 4*state_nv["synthesis.b4.resample_filter"]
    state_ros[f"to_rgbs.{level}.upsample.kernel"] = 4*state_nv["synthesis.b4.resample_filter"]


def determine_config(state_nv):
    mapping_names = [name for name in state_nv.keys() if "mapping.fc" in name]
    sythesis_names = [name for name in state_nv.keys() if "synthesis.b" in name]

    n_mapping =  max([int(re.findall("(\d+)", n)[0]) for n in mapping_names]) + 1
    resolution =  max([int(re.findall("(\d+)", n)[0]) for n in sythesis_names])
    n_layers = np.log(resolution/2)/np.log(2)

    return n_mapping, n_layers


def convert(network_pkl, output_file):
    with dnnlib.util.open_url(network_pkl) as f:
        G_nvidia = load_network_pkl(f)['G_ema']

    state_nv = G_nvidia.state_dict()
    n_mapping, n_layers = determine_config(state_nv)

    state_ros = {}

    for i in range(n_mapping):
        state_ros[f"style.{i+1}.weight"] = state_nv[f"mapping.fc{i}.weight"]
        state_ros[f"style.{i+1}.bias"] = state_nv[f"mapping.fc{i}.bias"]

    for i in range(int(n_layers)):
        if i > 0:
            for conv_level in range(2):
                convert_conv(state_ros, state_nv, f"convs.{2*i-2+conv_level}", f"synthesis.b{4*(2**i)}.conv{conv_level}")
                state_ros[f"noises.noise_{2*i-1+conv_level}"] = state_nv[f"synthesis.b{4*(2**i)}.conv{conv_level}.noise_const"].unsqueeze(0).unsqueeze(0)

            convert_to_rgb(state_ros, state_nv, f"to_rgbs.{i-1}", f"synthesis.b{4*(2**i)}")
            convert_blur_kernel(state_ros, state_nv, i-1)
        
        else:
            state_ros[f"input.input"] = state_nv[f"synthesis.b{4*(2**i)}.const"].unsqueeze(0)
            convert_conv(state_ros, state_nv, "conv1", f"synthesis.b{4*(2**i)}.conv1")
            state_ros[f"noises.noise_{2*i}"] = state_nv[f"synthesis.b{4*(2**i)}.conv1.noise_const"].unsqueeze(0).unsqueeze(0)
            convert_to_rgb(state_ros, state_nv, "to_rgb1", f"synthesis.b{4*(2**i)}")

    # https://github.com/yuval-alaluf/restyle-encoder/issues/1#issuecomment-828354736
    latent_avg = state_nv['mapping.w_avg']
    state_dict = {"g_ema": state_ros, "latent_avg": latent_avg}
    torch.save(state_dict, output_file)

