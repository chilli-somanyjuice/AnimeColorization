# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import cv2
import numpy as np
import os
import os.path as osp
import random
import sys
import torch
import torch.utils.data as data
from collections import Counter
from glob import glob
from skimage import io

from basicsr.utils.registry import DATASET_REGISTRY
from paint.utils import load_json, read_img_2_np, read_seg_2_np, recolorize_seg


class LineSegAugmentor:
    def __init__(self, rotate, resize, crop, rotate_prob):
        if resize < crop:
            raise ValueError("Crop size must be smaller than resize size.")
        self.rotate_degree = rotate
        self.resize_range = [crop, resize]
        self.crop_size = crop
        self.rotate_prob = rotate_prob

    def rotate_and_scale(self, image, angle):
        rows, cols = image.shape[:2]
        center = (cols / 2, rows / 2)
        angle_rad = np.deg2rad(angle)
        scale_factor = abs(np.sin(angle_rad)) + abs(np.cos(angle_rad))

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)

        transformed_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            flags=cv2.INTER_NEAREST,
        )
        return transformed_image

    def resize(self, image, resize_size):
        resized_image = cv2.resize(image, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
        return resized_image

    def crop(self, image, start_x, start_y):
        return image[start_y : start_y + self.crop_size, start_x : start_x + self.crop_size]

    def __call__(self, line, seg):

        apply_rotation = np.random.rand() <= self.rotate_prob
        angle = np.random.uniform(-self.rotate_degree, self.rotate_degree) if apply_rotation else 0
        resize_size = np.random.randint(self.resize_range[0], self.resize_range[1] + 1)
        start_x = np.random.randint(0, resize_size - self.crop_size + 1)
        start_y = np.random.randint(0, resize_size - self.crop_size + 1)

        line_transformed = self.rotate_and_scale(line, angle)
        seg_transformed = self.rotate_and_scale(seg, angle)

        line_resized = self.resize(line_transformed, resize_size)
        seg_resized = self.resize(seg_transformed, resize_size)

        line_cropped = self.crop(line_resized, start_x, start_y)
        seg_cropped = self.crop(seg_resized, start_x, start_y)

        # in case cropped frame excludes prospect character
        while len(np.unique(seg_cropped)) < 3:
            start_x = np.random.randint(0, resize_size - self.crop_size + 1)
            start_y = np.random.randint(0, resize_size - self.crop_size + 1)
            line_cropped = self.crop(line_resized, start_x, start_y)
            seg_cropped = self.crop(seg_resized, start_x, start_y)

        return line_cropped, seg_cropped


class AnimeLabelSegDataset(data.Dataset):
    def __init__(self, aug_params=None):
        if aug_params is not None:
            print("Data augmentation loaded!")
            self.augmentor = LineSegAugmentor(**aug_params)
        else:
            self.augmentor = None

        self.shuffle_label = False  # Shuffle the index of the labels
        self.color_redistribution_type = False  # Recolorize the label based on randomly selected colors, mainly used for the optical flow module
        self.merge_label_prob = 0.0  # Merge the labels as the data augmentation
        self.num_target_frames = 12

        # image, seg, matching index
        self.line_list = []
        self.label_list = []
        self.seg_list = []
        self.idx_list = []

    def __getitem__(self, index):
        
        index = index % len(self.line_list)

        file_name = self.line_list[index][0][:-4]

        # Load multiple target frames
        line_list = []
        target_seg_list = []
        keypoints_list = []
        centerpoints_list = []
        numpixels_list = []
        mat_index_list = []
        
        # Process reference frame first (outside the loop)
        line_ref = read_img_2_np(self.line_list[index][0])
        label_ref = read_seg_2_np(self.label_list[index][0])

        # Augment reference frame if needed
        if self.augmentor is not None:
            line_ref, label_ref = self.augmentor(line_ref, label_ref)

        label_ref_list = np.unique(label_ref[label_ref != 0])
        if self.shuffle_label:
            np.random.shuffle(label_ref_list)
            # -1 means background in label, move it to top
            label_ref_list = np.concatenate(([-1], label_ref_list[label_ref_list != -1])) if -1 in label_ref_list else label_ref_list

        # Process reference frame keypoints
        h, w = label_ref.shape
        hh = np.arange(h)
        ww = np.arange(w)
        xx, yy = np.meshgrid(ww, hh)

        label_ref_index = 1
        keypoints_ref = []
        centerpoints_ref = []
        numpixels_ref = []
        label_ref_relabeled = np.zeros_like(label_ref)

        for i, label_idx in enumerate(label_ref_list):
            mask = label_ref == label_idx
            merge = i > 5 and np.random.rand() <= self.merge_label_prob
            if merge:
                selected_label = np.random.randint(2, label_ref_index)
            else:
                selected_label = label_ref_index
                label_ref_index += 1
            label_ref_relabeled[mask] = selected_label
            mask = label_ref_relabeled == selected_label

            xs = xx[mask]
            ys = yy[mask]

            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            xmean = xs.mean()
            ymean = ys.mean()

            if merge:
                centerpoints_ref[selected_label - 1] = [xmean, ymean]
                numpixels_ref[selected_label - 1] = mask.sum()
                keypoints_ref[selected_label - 1] = [xmin, xmax, ymin, ymax]
            else:
                centerpoints_ref.append([xmean, ymean])
                numpixels_ref.append(mask.sum())
                keypoints_ref.append([xmin, xmax, ymin, ymax])

        keypoints_ref = np.stack(keypoints_ref)
        centerpoints_ref = np.stack(centerpoints_ref)
        numpixels_ref = np.stack(numpixels_ref)

        # Process target frames in loop
        for i in range(self.num_target_frames):
            # read target image and seg
            line = read_img_2_np(self.line_list[index][i+1])
            seg = read_seg_2_np(self.seg_list[index][i+1])

            # seg idx to label idx mapping
            seg_label_idx_map = load_json(self.idx_list[index][i+1])
            seg_label_idx_map = {int(k): v[1] for k, v in seg_label_idx_map.items()}

            # augment target frame if needed
            if self.augmentor is not None:
                line, seg = self.augmentor(line, seg)

            seg_list = np.unique(seg[seg != 0])
            if self.shuffle_label:
                np.random.shuffle(seg_list)

            # match seg to label_ref
            seg_list = list(seg_list)
            mat_index = [(label_ref_list.index(seg_label_idx_map[x]) if seg_label_idx_map[x] in label_ref_list else -1) for x in seg_list]
            mat_index = np.array(mat_index).astype(np.int64)

            keypoints = []
            centerpoints = []
            numpixels = []
            seg_relabeled = np.zeros_like(seg)

            for j, seg_idx in enumerate(seg_list):
                mask = seg == seg_idx
                xs = xx[mask]
                ys = yy[mask]

                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                xmean = xs.mean()
                ymean = ys.mean()

                centerpoints.append([xmean, ymean])
                numpixels.append(mask.sum())
                keypoints.append([xmin, xmax, ymin, ymax])
                seg_relabeled[mask] = j + 1

            keypoints = np.stack(keypoints)
            centerpoints = np.stack(centerpoints)
            numpixels = np.stack(numpixels)

            # Convert to torch tensors
            line = torch.from_numpy(line).permute(2, 0, 1).float() / 255.0
            seg = torch.from_numpy(seg_relabeled)[None]
            numpixels = torch.from_numpy(numpixels)[None]
            mat_index = torch.from_numpy(mat_index).float()

            line_list.append(line)
            target_seg_list.append(seg)
            keypoints_list.append(keypoints)
            centerpoints_list.append(centerpoints)
            numpixels_list.append(numpixels)
            mat_index_list.append(mat_index)

        # Convert reference frame to torch tensors
        line_ref = torch.from_numpy(line_ref).permute(2, 0, 1).float() / 255.0
        label_ref = torch.from_numpy(label_ref_relabeled)[None]
        numpixels_ref = torch.from_numpy(numpixels_ref)[None]

        if self.color_redistribution_type == "seg":
            recolorized_img = recolorize_seg(label_ref)
        else:
            recolorized_img = torch.Tensor(0)

        return {
            "keypoints_list": keypoints_list,
            "keypoints_ref": keypoints_ref,
            "centerpoints_list": centerpoints_list,
            "centerpoints_ref": centerpoints_ref,
            "line_list": line_list,
            "line_ref": line_ref,
            "numpixels_list": numpixels_list,
            "numpixels_ref": numpixels_ref,
            "segment_list": target_seg_list,
            "segment_ref": label_ref,
            "recolorized_img": recolorized_img,
            "all_matches_list": mat_index_list,
            "file_name": file_name,
        }

    def __rmul__(self, v):
        self.line_list = v * self.line_list
        self.label_list = v * self.label_list
        self.seg_list = v * self.seg_list
        self.idx_list = v * self.idx_list
        return self

    def __len__(self):
        return len(self.line_list)


@DATASET_REGISTRY.register()
class PaintBucketLabelSegDataset(AnimeLabelSegDataset):
    def __init__(self, opt):
        # This class is mainly for training.
        aug_params = opt["aug_params"] if "aug_params" in opt else None
        super(PaintBucketLabelSegDataset, self).__init__(aug_params)

        self.opt = opt
        self.root = opt["root"]
        self.shuffle_label = opt["shuffle_label"] if "shuffle_label" in opt else False
        self.color_redistribution_type = opt["color_redistribution_type"] if "color_redistribution_type" in opt else None
        assert self.color_redistribution_type in [None, "seg"]
        self.merge_label_prob = opt["merge_label_prob"] if "merge_label_prob" in opt else 0.0
        self.frame_distance_list = opt["frame_distance_list"] if "frame_distance_list" in opt else [1]
        self.num_target_frames = 12

        for character in os.listdir(self.root):

            idx_root = osp.join(self.root, character, "json_index")
            label_root = osp.join(self.root, character, "label")
            line_root = osp.join(self.root, character, "line")
            seg_root = osp.join(self.root, character, "seg")

            idx_list = sorted(glob(osp.join(idx_root, "*.json")))
            label_list = sorted(glob(osp.join(label_root, "*.png")))
            line_list = sorted(glob(osp.join(line_root, "*.png")))
            seg_list = sorted(glob(osp.join(seg_root, "*.png")))

            L = len(line_list)

            for i in range(L - self.num_target_frames):
                target_indices = range(i + 1, i + 1 + self.num_target_frames)
                
                self.idx_list.append([idx_list[i]] + [idx_list[j] for j in target_indices])
                self.label_list.append([label_list[i]] + [label_list[j] for j in target_indices])
                self.line_list.append([line_list[i]] + [line_list[j] for j in target_indices])
                self.seg_list.append([seg_list[i]] + [seg_list[j] for j in target_indices])

        print("Length of Training Sequence is", len(self.line_list))
