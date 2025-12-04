from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import cv2
import torch
from torchvision import transforms
import random
import scipy.spatial.transform as R
from .mono_dataset import MonoDataset

class SyntheticColonDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SyntheticColonDataset, self).__init__(*args, **kwargs)

        # Define camera intrinsics matrix
        self.K = np.array([[227.60416/475.0, 0, 237.5/475.0, 0],
                          [0, 227.60416/475.0, 237.5/475.0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        
        # self.K = np.array([[107.898/ 512.0, 0, 256/ 512.0,0],
        #                           [0, 107.898/ 512.0, 256/ 512.0,0],
        #                           [0, 0, 1,0],
        #                           [0, 0, 0,1]], dtype=np.float32)
        # Cache for position and rotation data
        self.position_cache = {}
        self.rotation_cache = {}

    def check_depth(self):
        """Check if the dataset has ground truth depth"""
        return True

    def get_color(self, folder, frame_index, side, do_flip):
        """Load color image with bounds checking"""
        try:
            color_path = self.get_image_path(folder, frame_index)
            color = self.loader(color_path)
            
            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
                
            return color
        except:
            raise FileNotFoundError(f"Could not find image for frame {frame_index} in {folder}")

    def get_image_path(self, folder, frame_index):
        """Get path to color image"""
        frame_str = "{:04d}".format(frame_index)
        image_path = os.path.join(
            self.data_path,
            folder,
            f"FrameBuffer_{frame_str}.png")
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        """Load depth image"""
        frame_str = "{:04d}".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            f"Depth_{frame_str}.png")

        # Read depth image
        depth_gt = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        if depth_gt is None:
            raise FileNotFoundError(f"Could not find depth file: {depth_path}")
            
        # Convert and normalize depth values
        if depth_gt.dtype == np.uint16:
            depth_gt = depth_gt.astype(np.float32) / 65000.0
        elif depth_gt.dtype == np.uint8:
            depth_gt = depth_gt.astype(np.float32) / 255.0
            
        # Convert to meters (based on your dataset's scale)
        depth_gt = depth_gt * 20 * 10.0
        
        # Handle invalid values
        depth_gt[depth_gt <= 0] = 0.001
        
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_pose(self, folder, frame_index):
        """Get pose information from position and rotation files"""
        sequence = folder.split('_')[-1]  # Gets 'S1' from 'Frames_S1'
        
        # Load pose data if not cached
        self.load_pose_data(sequence)
        
        # Get position and rotation for the current frame
        position = self.position_cache[sequence][frame_index]
        quaternion = self.rotation_cache[sequence][frame_index]
        
        # Convert to transformation matrix
        rotation_matrix = self.quaternion_to_matrix(quaternion)
        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = position
        
        return pose.astype(np.float32)

    def load_pose_data(self, sequence):
        """Load position and rotation data for a sequence if not already cached"""
        if sequence not in self.position_cache:
            position_file = os.path.join(self.data_path, f"SavedPosition_{sequence}.txt")
            rotation_file = os.path.join(self.data_path, f"SavedRotationQuaternion_{sequence}.txt")
            
            try:
                positions = np.loadtxt(position_file)
                positions = positions * 10.0
                rotations = np.loadtxt(rotation_file)
                self.position_cache[sequence] = positions
                self.rotation_cache[sequence] = rotations
            except:
                print(f"Failed to load pose data for sequence {sequence}")
                raise

    def quaternion_to_matrix(self, quaternion):
        """Convert quaternion to rotation matrix"""
        
        # Convert quaternion to rotation matrix using scipy
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        
        return rotation_matrix

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary."""
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]
        
        # Extract sequence number without 'S' prefix and convert to int
        sequence = folder.split('_')[-1]  # Gets 'S1' from 'Frames_S1'
        sequence_num = int(sequence[1:])  # Gets '1' from 'S1'
        inputs["sequence"] = torch.tensor(sequence_num)
        
        frame_index = int(line[1])
        inputs["frame_id"] = torch.tensor(frame_index)
        
        # Load target frame first
        # inputs[("color", 0, -1)] = self.get_color(folder, frame_index, None, do_flip)
        
        # Load adjacent frames
        for i in self.frame_idxs:
            
            next_frame_index = frame_index + i
            # Check if next frame exists (to handle sequence boundaries)
            try:
                inputs[("color", i, -1)] = self.get_color(folder, next_frame_index, None, do_flip)
                # resize the image to the same size as the target frame
                inputs[("color", i, -1)] = inputs[("color", i, -1)].resize((self.width, self.height))
            except:
                # report error:
                print(f"Error: Could not find image for frame {next_frame_index} in {folder}")

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            # print the value of intrinsic matrix

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        do_color_aug = False
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            if ("color", i, -1) in inputs:
                del inputs[("color", i, -1)]
            if ("color_aug", i, -1) in inputs:
                del inputs[("color_aug", i, -1)]
            

        # if self.load_depth:
        #     depth_gt = self.get_depth(folder, frame_index, None, do_flip)
        #     # resize the depth to the same size as the target frame
        #     depth_gt = cv2.resize(depth_gt, (self.width, self.height))
        #     inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        #     inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        # TODO: fix the adding pose step
        # # Add pose information if available
        # try:
        #     pose_gt = self.get_pose(folder, frame_index)
        #     inputs["pose_gt"] = torch.from_numpy(pose_gt)
        # except:
        #     print(f"Error: Could not find pose for frame {frame_index} in {folder}")
        #     pass  # Skip if pose data is not available

        return inputs