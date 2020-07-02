import torch
import argparse
from math import ceil as ceil
import os
import numpy as np
import torch.nn.functional as F
import cv2
import torch.multiprocessing as mp
import random

from utils.utils import to_cuda
from torch.backends import cudnn
#from model import model as model
from model import model_i3d as model
from config.config import cfg, cfg_from_file, cfg_from_list
import sys
import pprint
import data.video_transforms as video_transforms
from torchvision import transforms
from data.meva_dataset import MEVATest as Dataset
from diva_io.video import VideoReader

video_formats = ['avi', 'mp4']
sys.setrecursionlimit(80000)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with specified model parameters',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name', 
                        default='exp', type=str)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def crop(img, bbox, ratio=0.13):
    x1, y1, x2, y2 = bbox
    img_h, img_w = img.shape[0], img.shape[1]

    cx = (x1 + x2) / 2.
    cy = (y1 + y2) / 2.
    w = x2 - x1 + 1
    h = y2 - y1 + 1

    edge = max(w, h)
    new_edge_half = edge * (1 + 2 * ratio) / 2.
    new_x1 = int(max(0, cx - new_edge_half))
    new_y1 = int(max(0, cy - new_edge_half))
    new_x2 = int(min(img_w - 1, cx + new_edge_half))
    new_y2 = int(min(img_h - 1, cy + new_edge_half))
    return img[new_y1:new_y2+1, new_x1:new_x2+1, :]

def prepare_data(videos):
    test_dataset = Dataset(videos, cfg.DATASET.DATAROOT)
    video_paths = test_dataset.data
    return video_paths

def find_seeds(image, unvisited, pixel_val_thres):
    seeds = set()
    H, W = image.shape
    for h in range(H):
        for w in range(W):
            if image[h, w] >= pixel_val_thres and unvisited[h, w] > 0:
                pos = h * W + w
                seeds.add(pos)
    return seeds

def growing(pos, image, seeds, unvisited, region_id, labels, pixel_val_thres):
    H, W = pos
    max_H, max_W = image.shape
    if image[H, W] < pixel_val_thres or unvisited[H, W] == 0:
        unvisited[H, W] = 0
        return
    else:
        labels[H, W] = region_id
        unvisited[H, W] = 0
        pos = H * max_W + W
        if pos in seeds:
            seeds.remove(pos)

    for h in range(max(H-1, 0), min(H+2, max_H)):
        for w in range(max(W-1, 0), min(W+2, max_W)):
            if h == H and w == W:
                continue
            else:
                growing((h, w), image, seeds, unvisited, region_id, labels, pixel_val_thres)
    return

def regional_growing(image, pixel_val_thres=0.7):
    H, W = image.shape
    unvisited = np.ones([H, W])
    labels = np.zeros([H, W]) - 1.0
    count = 0
    seeds = find_seeds(image, unvisited, pixel_val_thres)
    while np.sum(unvisited) > 0:
        if len(seeds) == 0:
            break
        seed_pos = seeds.pop()
        h = seed_pos // W
        w = seed_pos - h * W
        #print('pos: %d, %d; prob: %f.' % (h, w, image[h, w]))
        # growing
        growing((h, w), image, seeds, unvisited, count, labels, pixel_val_thres)
        #print('unvisited: %d' % np.sum(unvisited))
        count += 1
    return labels, count

def IoU(A, B):
    xA1, yA1, xA2, yA2 = A
    xB1, yB1, xB2, yB2 = B
    
    intersection = max(min(xA2, xB2) - max(xA1, xB1), 0.0) * max(min(yA2, yB2) - max(yA1, yB1), 0.0)
    union = (max(xA2, xB2) - min(xA1, xB1)) * (max(yA2, yB2) - min(yA1, yB1))
    return 1.0 * intersection / union

def associate_bboxes(frame_id, cur, proposal, thres=0.3):
    used = [False] * len(cur)
    for prop in proposal:
        _, last_bbox = prop[-1]
        if last_bbox is None:
            continue

        max_iou = -1.0
        max_cb = []
        max_cid = 0
        cid = 0
        for cb in cur:
            iou = IoU(last_bbox, cb)
            if iou > max_iou:
                max_iou = iou
                max_cb = cb
                max_cid = cid

            cid += 1

        if max_iou > thres:
            prop.append((frame_id, max_cb))
            used[max_cid] = True
        else:
            prop.append((frame_id, None))

    for i in range(len(used)):
        if not used[i]:
            #print(frame_id)
            proposal.append([(frame_id, cur[i])])

    return proposal

def smoothing(mask, threshold=0.8, len_thres=5):
    h, w = mask.shape
    for i in range(h):
        start = -1
        end = -1
        for j in range(w):
            if mask[i, j] >= threshold and start == -1:
                start = j
            if mask[i, j] < threshold and start > 0:
                end = j

            if start > 0 and end > 0:
                length = end - start + 1
                if length < len_thres:
                    #print('Start %d, end %d' % (start, end))
                    mask[i, start:end+1] = 0
                start = end = -1

    for i in range(w):
        start = -1
        end = -1
        for j in range(h):
            if mask[j, i] >= threshold and start == -1:
                start = j
            if mask[j, i] < threshold and start > 0:
                end = j

            if start > 0 and end > 0:
                length = end - start + 1
                if length < len_thres:
                    #print('Start %d, end %d' % (start, end))
                    mask[start:end+1, i] = 0
                start = end = -1

def filtering(image, labels, num_regions, threshold=10):
    H, W = image.shape
    label_count = [0] * num_regions
    region_min_x = [W] * num_regions
    region_max_x = [-1] * num_regions
    region_min_y = [H] * num_regions 
    region_max_y = [-1] * num_regions
    #print('number of regions: %d' % num_regions)

    mask = np.zeros([H, W])
    for h in range(H):
        for w in range(W):
            if labels[h, w] == -1:
                continue
            #print(labels[h, w])
            region_id = int(labels[h, w])
            label_count[region_id] += 1
            region_min_x[region_id] = min(w, region_min_x[region_id])
            region_min_y[region_id] = min(h, region_min_y[region_id])
            region_max_x[region_id] = max(w, region_max_x[region_id])
            region_max_y[region_id] = max(h, region_max_y[region_id])

    eps = 1e-5
    region_to_remove = []
    for r in range(num_regions):
        # 1) occupation should be larger
        metric1 = label_count[r]
        if metric1 < threshold:
            region_to_remove += [r]
            continue

        # 2) more like a rectangle
        width = region_max_x[r] - region_min_x[r] + 1
        height = region_max_y[r] - region_min_y[r] + 1
        if width < 3 or height < 3:
            region_to_remove += [r]
            continue

        metric2 = 1.0 * label_count[r] / (width * height)
        if metric2 < 0.3:
            region_to_remove += [r]
            continue

        #metric3 = 1.0 * width / height
        #if metric3 > 2 or metric3 < 0.5:
        #    region_to_remove += [r]
        #    continue

    bboxes = []
    for r in range(num_regions):
        if r not in region_to_remove:
            x1, y1, x2, y2 = region_min_x[r], region_min_y[r], region_max_x[r], region_max_y[r]
            bboxes += [(x1, y1, x2, y2)]
            #print(bboxes[-1])
   
    #print('Remaining %d regions' % (num_regions - len(region_to_remove)))
    for h in range(H):
        for w in range(W):
            if labels[h, w] >= 0 and labels[h, w] not in region_to_remove:
                mask[h, w] = 1.0
    return mask, bboxes

def save_proposals(cfg, imgs, proposals, name, vid, start_f, prop_id):
    save_path = os.path.join(cfg.SAVE_DIR, name, vid)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, 'props.txt'), 'a') as f:
        end_f = start_f + cfg.DATASET.CLIP_LEN * cfg.DATASET.CLIP_STRIDE - 1
        for prop in proposals:
            _, bbox = prop[0]
            x1, y1, x2, y2 = bbox
            f.write('%d, %d, %d: %d, %d, %d, %d\n' % (prop_id, start_f, end_f, x1, y1, x2, y2))

            crop_save_path = os.path.join(save_path, str(prop_id))
            if not os.path.exists(crop_save_path):
                os.makedirs(crop_save_path)

            count = 0
            assert(len(imgs) == cfg.DATASET.CLIP_LEN * cfg.DATASET.CLIP_STRIDE), len(imgs)
            for n in range(cfg.DATASET.CLIP_LEN * cfg.DATASET.CLIP_STRIDE):
                img = imgs[n]
                # TODO:
                crop_img = crop(img, bbox, ratio=0.13)
                #crop_img = img[y1:y2+1, x1:x2+1, :]
                save_name = os.path.join(crop_save_path, 'image_%05d.jpg'% count)
                #save_name = os.path.join(crop_save_path, '%d.jpg' % count)
                cv2.imwrite(save_name, crop_img)
                count += 1

            prop_id += 1

    return prop_id

def processing_frames(imgs, test_transforms):
    frames = []
    for img in imgs:
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

    frames = np.asarray(frames, dtype=np.float32)
    if test_transforms is not None:
        frames = test_transforms(frames)
    return frames

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

def test_and_save_mask(cfg, net, vids, test_transforms):
    clip_len = cfg.DATASET.CLIP_LEN
    clip_stride = cfg.DATASET.CLIP_STRIDE

    net.eval()
    for vid in vids:
        cur_video_path = vid
        cur_vid = os.path.split(cur_video_path)[-1]

        prop_id = 0
        cur_video = VideoReader(cur_video_path)

        video_len = cur_video.length
        last_start_f = (video_len // (clip_len * clip_stride)) * (clip_len * clip_stride)
        if last_start_f == video_len:
            last_start_f = -1
            last_clip_len = -1
        else:
            last_clip_len = video_len - last_start_f
        print('video_len: %d, last_start_f: %d, last_clip_len: %d' % (video_len, last_start_f, last_clip_len))
       
        f_count = 0
        clip_imgs = []
        start_f = 0

        for frame in cur_video:
            f_count += 1
            clip_imgs.append(frame.numpy())
            if len(clip_imgs) < clip_len * clip_stride:
                continue

            start_f = f_count - (clip_len * clip_stride)
            print('video: %s, start_f: %d' % (cur_video_path, start_f))

            clips = processing_frames(clip_imgs[0::clip_stride], test_transforms)
            clips = video_to_tensor(clips).unsqueeze(0)
            assert(len(clips.size()) == 5), clips.size()
            assert(clips.size(1) == 3), clips.size(1)
            assert(clips.size(2) == clip_len), clips.size(2)

            # forward and get the prediction result
            vpred = net(to_cuda(clips))
            probs = F.softmax(vpred, dim=1)
            pos_probs = probs[:, 1, :, :, :]

            # generate and save proposals
            proposals = []
            for count in range(clip_len):
                cur_pos_probs = pos_probs[0, count, :, :].cpu().numpy()

                # to speedup, first downsample the probability map
                resized_cur_pos_probs = cv2.resize(cur_pos_probs, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                #smoothing(resized_cur_pos_probs, len_thres=2)
                labels, num_regions = regional_growing(resized_cur_pos_probs, pixel_val_thres=0.3)
                # upsample the label map
                ori_h, ori_w = cur_pos_probs.shape
                labels = cv2.resize(labels, dsize=(ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

                cur_pos_probs, bboxes = filtering(cur_pos_probs, labels, num_regions, 5)
                if len(proposals) == 0:
                    proposals = [[(count, bbox)] for bbox in bboxes]
                else:
                    associate_bboxes(count, bboxes, proposals)

            h, w = cur_video.height, cur_video.width
            new_proposals = [prop for prop in proposals if len(prop) >= 7]
            print('Number of proposals before and after filtering: %d, %d' % (len(proposals), len(new_proposals)))

            if len(new_proposals) == 0:
                save_path = os.path.join(cfg.SAVE_DIR, 'proposals', cur_vid)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                with open(os.path.join(save_path, 'props.txt'), 'a') as f:
                    f.close()
            else:
                stride_x = 1.0 * w / cfg.DATA_TRANSFORM.FINESIZE
                stride_y = 1.0 * h / cfg.DATA_TRANSFORM.FINESIZE
                new_proposals = resize_proposals(new_proposals, stride_x, stride_y, w, h)
                prop_id = save_proposals(cfg, clip_imgs, new_proposals, 'proposals', cur_vid, start_f, prop_id)

            if start_f == (last_start_f - clip_len * clip_stride):
                clip_imgs = clip_imgs[last_clip_len:]
            else:
                clip_imgs = []

def resize_proposals(proposals, stride_x, stride_y, W, H):
    new_proposals = []
    for prop in proposals:
        frame_ids = []
        min_x1 = 1e5
        max_x2 = -1
        min_y1 = 1e5
        max_y2 = -1
        for p in prop:
            frame_id, bbox = p
            if bbox is None:
                continue
            frame_ids += [frame_id]

            x1, y1, x2, y2 = bbox
            #new_bbox = (int(x1 * stride_x), int(y1 * stride_y), int(x2 * stride_x), int(y2 * stride_y))
            min_x1 = min(int(x1 * stride_x), min_x1)
            max_x2 = max(int(x2 * stride_x), max_x2)
            min_y1 = min(int(y1 * stride_y), min_y1)
            max_y2 = max(int(y2 * stride_y), max_y2)

        new_bbox = (max(min_x1, 0), max(min_y1, 0), min(max_x2, W-1), min(max_y2, H-1))
        new_proposals += [[(frame_id, new_bbox) for frame_id in frame_ids]]
    return new_proposals

def test(cfg, vids, test_transforms, gpu_id):
    with torch.cuda.device(gpu_id):
        # initialize model
        model_state_dict = None
        if cfg.WEIGHTS != '':
            param_dict = torch.load(cfg.WEIGHTS, map_location='cuda:%d'%(gpu_id))
            model_state_dict = param_dict['parameters']

        net = model.get_MaskGenNet(cfg.MODEL.PRETRAINED_ENCODER_WEIGHTS, state_dict=model_state_dict)
        net.cuda()
        with torch.no_grad():
            test_and_save_mask(cfg, net, vids, test_transforms)

    print('Finished!')

if __name__ == '__main__':
    #cudnn.benchmark = True 
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.weights is not None:
        cfg.MODEL = args.weights
    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name 

    print('Using config:')
    pprint.pprint(cfg)

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)
    print('Output will be saved to %s.' % cfg.SAVE_DIR)

    mp = mp.get_context('spawn')

    test_filename = cfg.DATASET.TEST_SPLIT_NAME
    video_names = []
    with open(test_filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vid = line.strip()
            if vid == "":
                continue
            suffix = vid.split('.')[-1]
            if suffix not in video_formats:
                vid += '.%s' % cfg.DATASET.VIDEO_FORMAT
            video_names += [vid]
    print('Processing %d videos in total.' % len(video_names))

    threads_per_gpu = cfg.THREADS_PER_GPU
    num_threads = threads_per_gpu * cfg.NUM_GPUS
    videos_per_thread = len(video_names) // num_threads
    remain_videos = len(video_names) - num_threads * videos_per_thread
    num_videos = [videos_per_thread] * num_threads
    for i in range(remain_videos):
        num_videos[i] += 1

    test_video_paths = []
    start = 0
    random.shuffle(video_names)
    for n in range(num_threads):
        end = start + num_videos[n]
        videos_to_process = video_names[start:end]
        # prepare the data
        test_video_paths += [prepare_data(videos_to_process)]
        start = end
        if start >= len(video_names):
            break

    test_transforms = transforms.Compose([
        video_transforms.Resize(cfg.DATA_TRANSFORM.FINESIZE)])

    ps = []
    for n in range(len(test_video_paths)):
        gpu_id = int(n // threads_per_gpu)
        vids = test_video_paths[n]
        p1 = mp.Process(target=test, args=[cfg, vids, test_transforms, gpu_id])
        p1.start()
        ps += [p1]

    for p in ps:
        p.join()

