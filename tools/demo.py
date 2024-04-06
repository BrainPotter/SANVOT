from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.model_load import load_pretrain

import pyrealsense2 as rs
from ctypes import *
import numpy.ctypeslib as npct

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', default='/home/unitree/siamban/experiments/siamban_r50_l234/config.yaml', type=str, help='config file')
parser.add_argument('--snapshot', type=str, default='/home/unitree/siamban/experiments/siamban_r50_l234/model.pth', help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--save', action='store_true', default=True, 
        help='whether visualzie result')
args = parser.parse_args()


# Configure depth and color streams
pipeline = rs.pipeline()
# 创建 config 对象：
config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

score_threshold = 0.1

class Detector():
    def __init__(self,model_path,dll_path):
        self.yolov5 = CDLL(dll_path)
        self.yolov5.Detect.argtypes = [c_void_p,c_int,c_int,POINTER(c_ubyte),npct.ndpointer(dtype = np.float32, ndim = 2, shape = (50, 6), flags="C_CONTIGUOUS")]
        self.yolov5.Init.restype = c_void_p
        self.yolov5.Init.argtypes = [c_void_p]
        self.yolov5.cuda_free.argtypes = [c_void_p]
        self.c_point = self.yolov5.Init(model_path)

    def predict(self,img):
        rows, cols = img.shape[0], img.shape[1]
        res_arr = np.zeros((50,6),dtype=np.float32)
        self.yolov5.Detect(self.c_point,c_int(rows), c_int(cols), img.ctypes.data_as(POINTER(c_ubyte)),res_arr)
        self.bbox_array = res_arr[~(res_arr==0).all(1)]
        return self.bbox_array

    def free(self):
        self.yolov5.cuda_free(self.c_point)


def get_frames(video_name):
    # if not video_name:
    #     cap = cv2.VideoCapture(0)
    #     # warmup
    #     for i in range(5):
    #         cap.read()
    #     while True:
    #         ret, frame = cap.read()
    #         if ret:
    #             yield frame
    #         else:
    #             break
    if not video_name:
        frames = pipeline.wait_for_frames()
        # warmup
        for i in range(5):
            color_frame = frames.get_color_frame()
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                color_frame = np.asanyarray(color_frame.get_data())
                yield color_frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4') or \
        video_name.endswith('mov'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()
    det = Detector(model_path=b"./last1.engine",dll_path="./libyolov5.so")  # b'' is needed

    # load model
    # model.load_state_dict(torch.load(args.snapshot,
    #     map_location=lambda storage, loc: storage.cpu()))
    # model.eval().to(device)

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'camera'
    #cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    print(video_name)

    save_video_path = 'camera_tracking2.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # frame_size = (frame.shape[1], frame.shape[0]) # (w, h)
    frame_size = (640,480) # (w, h)
    video_writer = cv2.VideoWriter(save_video_path, fourcc, 3, frame_size,True)
    try:
        for frame in get_frames(args.video_name):
            if first_frame:
                # build video writer
                if args.save:
                    if args.video_name.endswith('avi') or \
                        args.video_name.endswith('mp4') or \
                        args.video_name.endswith('mov'):
                        # cap = cv2.VideoCapture(args.video_name)
                        # fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
                        fps = 30
                    else:
                        fps = 30

                    # save_video_path = args.video_name.split(video_name)[0] + video_name + '_tracking.mp4'

                    print('predicting')
                try:
                    #cv2.imshow(video_name, frame)
                    result = det.predict(frame)
                    for temp in result:
                        bbox = [max(0,temp[0]),max(0,temp[1]),max(0,temp[2]),max(0,temp[3])]  #xywh
                        clas = int(temp[4]) #clas  0 have_mask 1 no_mask 2 incorrect_mask
                        if (clas == 1):
                            print('no_mask')
                            print(bbox)
                            init_frame = frame[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]
                            init_rect = bbox
                            tracker.init(frame, tuple(init_rect))
                            first_frame = False
                            cv2.imshow('init_frame', init_frame)
                            break
                except IndexError as e:
                    print(e)
                    exit()
            else:
                outputs = tracker.track(frame)
                if 'polygon' in outputs:
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                    frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                else:
                    bbox = list(map(int, outputs['bbox']))
                    print(outputs['best_score'],type(outputs['best_score']))
                    score = outputs['best_score']
                    #print(score)
                    if score <= score_threshold:
                        first_frame = True
                        cv2.destroyWindow('init_frame')
                    else:
                        cv2.rectangle(frame, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                      (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            video_writer.write(frame)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                print('exit')
                cv2.destroyAllWindows()
                if args.save:
                    video_writer.release()
                    print('save video')
                    exit()
    except KeyboardInterrupt as e:
        print("pipeline stop")
        pipeline.stop()
        video_writer.release()




        # if args.save:
        #     video_writer.write(frame)
    
    # if args.save:
    #     video_writer.release()


if __name__ == '__main__':
    main()
