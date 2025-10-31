# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random, os

import torch
from torch.utils.data._utils.collate import default_collate

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
    #     import pdb; pdb.set_trace()

    # cam_seq_len = 2
    # bs = len(samples_list)

    # collated_ori_teacher_lst = []
    # for i in range(cam_seq_len):
    #     collated_ori_teacher = torch.stack([s[0]["ori_teacher"][i] for s in samples_list])
    #     collated_ori_teacher_lst.append(collated_ori_teacher)
    
    # collated_ori_lst = []
    # for i in range(cam_seq_len):
    #     collated_ori = torch.stack([s[0]["ori"][i] for s in samples_list])
    #     collated_ori_lst.append(collated_ori)

    # num_stu_ensemble = len(samples_list[0][0]["rotated"])
    # collated_rotated_lst = [[] for _ in range(num_stu_ensemble)]
    # for k in range(num_stu_ensemble):
    #     for i in range(cam_seq_len):
    #         collated_rotated = torch.stack([s[0]["rotated"][k][i] for s in samples_list])
    #         collated_rotated_lst[k].append(collated_rotated)

    features, targets, tokens = default_collate(samples_list)

    B = len(features['ori'][0])
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)  # [n_samples_masked+1,]
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))  # [np, np]
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)  # len(masks_list) == n_samples_masked (nsm)

    # upsampling, because vit use AdaptivePooling...
    collated_masks = torch.stack(masks_list)  # [nsm, np, np]
    collated_masks_up = torch.nn.functional.interpolate(collated_masks.unsqueeze(1).float(), scale_factor=2, mode='nearest').squeeze(1).bool()
    
    if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        # import pdb; pdb.set_trace()
        # Visualize and save masks
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(collated_masks[0].cpu().numpy())
        plt.title('Original Mask')
        plt.subplot(122)
        plt.imshow(collated_masks_up[0].cpu().numpy())
        plt.title('Upsampled Mask')
        plt.savefig('masks_visualization.png')
        plt.close()  # [nsm, 2*np, 2*np]

    collated_masks = collated_masks.flatten(1)  # [nsm, np*np]
    mask_indices_list = collated_masks.flatten().nonzero().flatten()  # [\sum_{i=0}^{bs*n_global_crops-1}{num_nonzero},]
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    collated_masks_up = collated_masks_up.flatten(1)

    features.update({
        "collated_masks": collated_masks_up,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    })

    return features, targets, tokens

    