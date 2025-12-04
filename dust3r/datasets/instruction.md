You are a senior deep learning engineer. You are given a dataset and you are asked to build a dataloader for it.

I have a dataset called SyntheticColon. It is a dataset of synthetic colonoscopy images. I previously write a dataloader file called synthetic_colon.py. Now, I want to write a new dataloader file for new model training. The existing dataloader files are files such as tartanair.py and pointodyssey.py for reference.

Please write a new dataloader file for the SyntheticColon dataset based on my existing synthetic_colon.py. You can refer tartanair.py and pointodyssey.py for the structure of the dataloader file.

The SyntheticColon dataset is a dataset of synthetic colonoscopy images. It has the following structure:

-Frames_S1:
    -Depth_0000.png
    -Depth_0001.png
    -Depth_0002.png
    -...
    -Depth_0009.png
    -...
    -Depth_0099.png
    -...
    -Depth_0100.png
    -...
    -FrameBuffer_0000.png
    -FrameBuffer_0001.png
    -FrameBuffer_0002.png
    -...
    -FrameBuffer_0099.png
    -...
    -FrameBuffer_0100.png

-Frames_S2:
    -...

-...

-Frames_S15:
    -...

-SavedPosition_S1.txt
-SavedPosition_S2.txt
-...
-SavedPosition_S15.txt

-SavedRotationQuaternion_S1.txt
-SavedRotationQuaternion_S2.txt
-...
-SavedRotationQuaternion_S15.txt

-SavedIntrinsicMatrix_S1.txt
-SavedIntrinsicMatrix_S2.txt
-...
-SavedIntrinsicMatrix_S15.txt

-cam.txt


Among these files, Frames_S5 and Frames_S15 are the test set. The rest are the training set. The cam.txt file contains the camera intrinsic matrix: 227.60416 0 227.60416 0 237.5 237.5 0 0 1.