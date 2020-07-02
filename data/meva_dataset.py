import torch
from torch.utils.data import Dataset as Dataset
from  config.config import cfg as cfg
import os
import cv2
import numpy as np
from math import ceil as ceil
import random
from diva_io.video import VideoReader

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    if len(pic.shape) == 4:
        return torch.from_numpy(pic.transpose([3, 0, 1, 2]))
    else:
        return torch.from_numpy(pic)

def load_frames_from_video(video_path, start, num, stride=1):
    frames = []

    cap = VideoReader(video_path)
    start_frame_id = start * stride
    video_len = cap.length
    
    length = num * stride 
    if length > video_len - start_frame_id:
        start_frame_id = video_len - length

    cap.seek(start_frame_id)

    count = 0
    for frame in cap.get_iter(length):
        if count % stride:
            count += 1
            continue

        img = frame.numpy()

        assert(len(img.shape) > 1)
        img = img[:, :, [2, 1, 0]]
        h, w, c = img.shape
        #print('shape: w: %d, h: %d, c: %d' % (w, h, c))

        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        img = (img / 255.) * 2 - 1
        frames.append(img)
        count += 1

    return np.asarray(frames, dtype=np.float32), start_frame_id

def make_dataset(video_names, video_root):
    dataset = []
    num_frames_collect = []

    i = 0
    for vid in video_names:
        video_path = os.path.join(video_root, vid)
        try:
            cap = VideoReader(video_path)
        except:
            print('Error in reading %s' % video_path)
            continue


        if not os.path.exists(video_path):
            print('Warning: %s not exist!' % video_path)
            continue

        num_frames = cap.length
        if num_frames < cfg.DATASET.CLIP_LEN:
            print('Skipping %s due to the short length %d.' % (video_path, num_frames))
            continue

        dataset.append(video_path)
        num_frames_collect.append(num_frames)

        #print(vid, num_frames)
        i += 1
    
    return dataset, num_frames_collect

class MEVATest(Dataset):
    def __init__(self, video_names, video_root, transforms=None):
        self.video_names = video_names
        self.video_root = video_root

        self.clip_len = cfg.DATASET.CLIP_LEN
        self.clip_stride = cfg.DATASET.CLIP_STRIDE

        self.data, self.num_frames = make_dataset(video_names, video_root)
        self.efinds = []
        self.get_efinds()
        self.transforms = transforms

    def get_efinds(self):
        self.efinds = []
        stride = self.clip_len * self.clip_stride
        for nf in self.num_frames:
            last = self.efinds[-1] if len(self.efinds) > 0 else -1
            num_ind = ceil(1.0 * nf / stride)
            self.efinds.append(last + num_ind)
        return

    def get_vid_and_fid(self, index):
        vind = 0
        for ind in self.efinds:
            if ind >= index:
                break
            vind += 1

        start_find = self.efinds[vind - 1] + 1 if vind > 0 else 0
        start = index - start_find
        return vind, start

    def __getitem__(self, index):
        vind, start = self.get_vid_and_fid(index)
        vid = self.data[vind]

        clip_len = self.clip_len
        stride = self.clip_stride

        cur_video = vid
        start_f = clip_len * start
        try:
            imgs, real_start = load_frames_from_video(cur_video, start_f, clip_len, stride=stride)
        except:
            assert(False), 'video %s, start_f %d, clip_len %d, stride %d' % (cur_video, start_f, clip_len, stride)

        if self.transforms is not None:
            imgs = self.transforms(imgs)
        
        imgs = video_to_tensor(imgs)
        return vid, real_start, imgs

    def __len__(self):
        return self.efinds[-1] + 1
