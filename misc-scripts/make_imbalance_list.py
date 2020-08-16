import os
import sys

def get_img_num_per_cls(cls_num, imb_type, imb_factor):
    img_max = total_examples / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls






dataset = "multi"
domain = "real"
imbalance_type = "step"
imbalance_factor = 0.1

balanced_lists_path = "../data/txt/%s/"%(dataset)
