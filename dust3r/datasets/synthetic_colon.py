import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import cv2
import glob
from scipy.spatial.transform import Rotation as R

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.datasets.base.mast3r_base_stereo_view_dataset import MASt3RBaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.misc import get_stride_distribution

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

class SyntheticColonDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='data/synthetic_colon',
                 dset='train',
                 use_augs=False,
                 S=2,
                 strides=[1,2,3],
                 clip_step=2,
                 quick=False,
                 verbose=False,
                 dist_type=None,
                 *args, 
                 **kwargs
                 ):

        self.dataset_label = 'synthetic_colon'
        self.split = dset
        self.S = S  # number of frames
        self.verbose = verbose
        self.use_augs = use_augs
        self.dset = dset

        super().__init__(*args, **kwargs)

        print('loading synthetic colon dataset...')

        self.rgb_paths = []
        self.depth_paths = []
        self.pose_paths = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = strides

        # Find all sequence folders (Frames_S*)
        sequences = glob.glob(os.path.join(dataset_location, "Frames_S*"))
        self.sequences = []
        
        # Filter sequences based on train/test split
        for seq in sequences:
            sequence_num = int(seq.split('_S')[-1])
            # Only use S5 and S15 for test set, all others for train
            if dset == 'test' and sequence_num in [5, 15]:
                self.sequences.append(seq)
            elif dset == 'train' and sequence_num not in [5, 15]:
                self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        if self.verbose:
            print(self.sequences)
        print('found %d unique sequences in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))

        if quick:
           self.sequences = self.sequences[1:2] 
        
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)

            sequence = f"S{seq.split('_S')[-1]}"  # Gets 'S1' from 'Frames_S1'
            rgb_path = seq
            depth_path = seq  # Depth files are in the same folder
            
            # Load pose data
            position_file = os.path.join(dataset_location, f"SavedPosition_{sequence}.txt")
            rotation_file = os.path.join(dataset_location, f"SavedRotationQuaternion_{sequence}.txt")
            
            if os.path.isfile(position_file) and os.path.isfile(rotation_file):
                for stride in strides:
                    frame_count = len(glob.glob(os.path.join(rgb_path, "FrameBuffer_*.png")))
                    for ii in range(0, frame_count-self.S*stride+1, clip_step):
                        full_idx = ii + np.arange(self.S)*stride
                        self.rgb_paths.append([os.path.join(rgb_path, f"FrameBuffer_{idx:04d}.png") for idx in full_idx])
                        self.depth_paths.append([os.path.join(depth_path, f"Depth_{idx:04d}.png") for idx in full_idx])
                        self.pose_paths.append((position_file, rotation_file))
                        self.full_idxs.append(full_idx)
                        self.sample_stride.append(stride)
                    if self.verbose:
                        sys.stdout.write('.')
                        sys.stdout.flush()

        self.stride_counts = {}
        self.stride_idxs = {}
        for stride in strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)
        print('stride counts:', self.stride_counts)
        
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))

        if len(self.rgb_paths) == 0:
            raise ValueError(f"No data found in {dataset_location} for split {dset}")

        # Define camera intrinsics matrix from cam.txt values
        self.intrinsics = np.array([[227.60416, 0, 227.60416],
                                  [0, 237.5, 237.5],
                                  [0, 0, 1]], dtype=np.float32)

    def _resample_clips(self, strides, dist_type):
        # Get distribution of strides, and sample based on that
        dist = get_stride_distribution(strides, dist_type=dist_type)
        dist = dist / np.max(dist)
        max_num_clips = self.stride_counts[strides[np.argmax(dist)]]
        num_clips_each_stride = [min(self.stride_counts[stride], int(dist[i]*max_num_clips)) for i, stride in enumerate(strides)]
        print('resampled_num_clips_each_stride:', num_clips_each_stride)
        resampled_idxs = []
        for i, stride in enumerate(strides):
            resampled_idxs += np.random.choice(self.stride_idxs[stride], num_clips_each_stride[i], replace=False).tolist()

        self.rgb_paths = [self.rgb_paths[i] for i in resampled_idxs]
        self.depth_paths = [self.depth_paths[i] for i in resampled_idxs]
        self.pose_paths = [self.pose_paths[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths)

    def quaternion_to_matrix(self, quaternion):
        """Convert quaternion to rotation matrix"""
        
        # Convert quaternion to rotation matrix using scipy
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        
        return rotation_matrix
    
    def _get_views(self, index, resolution, rng):
        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        position_path, rotation_path = self.pose_paths[index]
        full_idx = self.full_idxs[index]

        # Load pose data
        positions = np.loadtxt(position_path)
        rotations = np.loadtxt(rotation_path)

        views = []
        for i in range(2):
            impath = rgb_paths[i]
            depthpath = depth_paths[i]

            # Load camera params
            position = positions[full_idx[i]]
            quaternion = rotations[full_idx[i]]
            rotation_matrix = self.quaternion_to_matrix(quaternion)
            
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = rotation_matrix
            # position is in mm
            camera_pose[:3, 3] = position*10.0


            # Load image and depth
            rgb_image = imread_cv2(impath)
            depth16 = cv2.imread(depthpath, cv2.IMREAD_ANYDEPTH)
            depthmap = depth16.astype(np.float32) / 65000.0 * 20.0 * 10.0  # Convert to mm
            # depthmap = depthmap * 1000.0 # Convert to mm
            depthmap[depthmap <= 0] = 0.001  # Handle invalid values

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, self.intrinsics, resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=rgb_paths[i].split('/')[-3],
                instance=osp.split(rgb_paths[i])[1],
            ))
        return views


class SyntheticColonDUSt3RMetric(SyntheticColonDUSt3R, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_metric_scale = True


if __name__ == "__main__":
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = SyntheticColonDUSt3R(
        use_augs=False,
        S=2,
        strides=[1,2,3],
        clip_step=2,
        quick=False,
        verbose=False,
        resolution=(256,256),
        dist_type='linear_9_1',
        aug_crop=16)

    def visualize_scene(idx):
        views = dataset[idx]
        assert len(views) == 2
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 1)
        label = views[0]['label']
        instance = views[0]['instance']
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                        focal=views[view_idx]['camera_intrinsics'][0, 0],
                        color=(255, 0, 0),
                        image=colors,
                        cam_size=cam_size)
        path = f"./tmp/synthetic_colon/colon_scene_{label}_{instance}.glb"
        return viz.save_glb(path)

    idxs = np.arange(0, len(dataset)-1, (len(dataset)-1)//10)
    for idx in idxs:
        print(f"Visualizing scene {idx}...")
        visualize_scene(idx)