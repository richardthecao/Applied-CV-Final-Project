import cv2
import glob
import numpy as np
import torch
from PIL import Image
from random import shuffle
from torchvision import transforms

dinov2_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
dinov2_model = dinov2_model

file_list_unlabel = glob.glob('data/pokemon/**/*.jpg',recursive=True)
shuffle(file_list_unlabel)


all_features_unlabel = {}
for index, file in enumerate(file_list_unlabel):
    pokemon_index = file.split('/')[-1]

    if pokemon_index in all_features_unlabel:
        if len(all_features_unlabel[pokemon_index])>5:
            continue

    input_path_unlabel = file_list_unlabel[index]
    input_unlabel = cv2.imread(input_path_unlabel)

    ## numpy to tensor and pre-processing
    input_object_pil_unlabel = Image.fromarray(np.uint8(input_unlabel))
    input_object_unlabel = np.array(input_object_pil_unlabel)
    input_object_tensor_unlabel = dinov2_preprocess(input_object_pil_unlabel)
    input_object_tensor_unlabel = input_object_tensor_unlabel.type(torch.float32)


    ## dinov2 feature exatarction
    input_feat_unlabel = dinov2_model.forward_features(input_object_tensor_unlabel.unsqueeze(0))
    patch_tokens_input_unlabel = input_feat_unlabel['x_norm_patchtokens']  # 1x256x1536
    class_token_input_unlabel = input_feat_unlabel['x_norm_clstoken']

    ## dinov2 feature
    input_feat_unlabel = class_token_input_unlabel.detach().cpu().numpy()


    if pokemon_index not in all_features_unlabel:
        all_features_unlabel[pokemon_index] = []
        all_features_unlabel[pokemon_index].append(input_feat_unlabel)
    else:
        all_features_unlabel[pokemon_index].append(input_feat_unlabel)
# write to disk
print(all_features_unlabel)
np.save('all_features_unlabel5.npy', all_features_unlabel)
