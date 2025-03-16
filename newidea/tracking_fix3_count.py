# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import logging
from logging.handlers import RotatingFileHandler
import os
import argparse

# 配置日志
def setup_logger():
    logger = logging.getLogger("TrackingLogger")
    logger.setLevel(logging.INFO)

    # 控制台日志（带颜色）
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('\033[92m%(asctime)s - %(levelname)s - %(message)s\033[0m') 
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件日志
    file_handler = RotatingFileHandler('tracking.log', maxBytes=1024 * 1024, backupCount=5)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

def tracking(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {input_video}")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_x1, roi_y1, roi_x2, roi_y2 = 60,20,550,475
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (roi_x2-roi_x1, roi_y2-roi_y1), isColor=True)
    mapped_state = {
        0: 'state1',  # 无遮挡
        1: 'state2',  # 一分多
        2: 'state3'   # 多合一
    }

    tracklist = []  
    trackcolor = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),(255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
        (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (0, 64, 64), (64, 0, 64),
        (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (0, 192, 192), (192, 0, 192),
        (255, 128, 0), (128, 255, 0), (255, 0, 128), (128, 0, 255), (0, 255, 128), (0, 128, 255)
    ]
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    
    output_image_folder = os.path.join("tracking_images", video_name)
    effect_image = os.path.join(output_image_folder, 'effect_image')
    result_dir = os.path.join(output_image_folder,'result')
    bg_dir = os.path.join(output_image_folder, 'bg')
    inflate_dir = os.path.join(output_image_folder, 'inflate')
    
    for d in [output_image_folder, effect_image, bg_dir, inflate_dir,result_dir]:
        os.makedirs(d, exist_ok=True)

    saved_tracks = set()
    frame_count = 0
    from dataclasses import dataclass
    @dataclass
    class TrackBlock:
        track_indexes: list
        counter: int = 0  # 可扩展其他属性

    OCCLUSION_AREA_RATIO = 1.8
    MIN_CONTOUR_AREA = 60
    same_blocks_list=[[]]
    last_area_num=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if frame_count == 47:
            cv2.waitKey(1)
        fg_mask = fgbg.apply(frame)
        cv2.imwrite(os.path.join(bg_dir, f'{frame_count}.jpg'), fg_mask)
        
        fg_mask = cv2.erode(fg_mask, np.ones((3,3), np.uint8), iterations=1)
        fg_mask = cv2.dilate(fg_mask, np.ones((3,3), np.uint8), iterations=1)
        cv2.imwrite(os.path.join(inflate_dir, f'{frame_count}.jpg'), fg_mask)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        contour_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_CONTOUR_AREA < area < 490*450:
                valid_contours.append(contour)
                contour_areas.append(area)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        xy_areas = []
        for i, cnt in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w/2, y + h/2)
            area = contour_areas[i]
            xy_areas.append((center, area))

        avg_area = np.mean(contour_areas) if contour_areas else 0
        if avg_area == 0:
            continue
        max_area = np.max(contour_areas) if contour_areas else 0

        tracklist = [ (t[0], t[1], t[2], t[3]+1, t[4],t[5],t[6]) for t in tracklist ]

        if not tracklist:
            for (xy, area) in xy_areas:
                tracklist.append((xy, (0, 0), 1, 0, area,0,mapped_state[0]))
                same_blocks_list=[TrackBlock([0],0)]
        else:
            dis_list = []
            for (xy, current_area) in xy_areas:     
                
                dis = 1000
                min_dis = 1000
                closest = 0

                for i, trackitem in enumerate(tracklist):
                    predict = (trackitem[0][0] + trackitem[1][0], 
                             trackitem[0][1] + trackitem[1][1])
                    current_dis = math.dist(xy, predict)
                    if current_dis<min_dis and trackitem[3]<2:
                        min_dis=current_dis
                        min_index = trackitem[5]
                    if current_dis < min(45, dis)and trackitem[3]<2 and xy[0]<484:
                        dis = current_dis
                        color_index=trackitem[5]
                        closest = i

                # cv2.waitKey(0)
                #判断这个距离超过阈值的积木是否是新积木
                if dis == 1000:
                    if xy[0]<460:#不是新下落的积木，可能前一帧有遮挡  此为1分2情况
                        color_index=0
                        closest=min_index
                        for i, linked_list in enumerate(same_blocks_list):
                            if min_index in linked_list.track_indexes:
                                motion = (xy[0] - tracklist[closest][0][0], 
                                    xy[1] - tracklist[closest][0][1])
                                tracklist.append((xy, motion, 1, 0, current_area,min_index, mapped_state[1]))                

                                color_index=i
                                linked_list.track_indexes.append(len(tracklist)-1)  
                                linked_list.obstructed_situation=1
                        cv2.circle(frame, (round(xy[0]), round(xy[1])), 3, 
                            trackcolor[same_blocks_list[color_index].track_indexes[0] % len(trackcolor)], 2,1)
                    
                    else:#新下落的积木
                        
                        tracklist.append((xy, (0, 0), 1, 0, current_area,len(tracklist), mapped_state[0]))  
                        same_blocks_list.append(TrackBlock([len(tracklist)-1],0))
                        cv2.circle(frame, (round(xy[0]), round(xy[1])), 3, 
                                trackcolor[(len(tracklist)-1) % len(trackcolor)], 2)
                    
                        closest=len(tracklist)-1
                        dis_list.append(closest)
                
                else:
                    
                    dis_list.append(closest)
                    motion = (xy[0] - tracklist[closest][0][0], 
                            xy[1] - tracklist[closest][0][1])
                    tracklist[closest] = (xy, motion, tracklist[closest][2]+1, tracklist[closest][3]-1, current_area,tracklist[closest][5],tracklist[closest][6])
                    cv2.circle(frame, (round(xy[0]), round(xy[1])), 3, 
                             trackcolor[color_index % len(trackcolor)], 2)
                for d in [result_dir+f"/{tracklist[closest][5]}"]:
                    os.makedirs(d, exist_ok=True)
                if xy[0]>460 or xy[0]<10 or xy[1]>450 or xy[1]<5:
                    continue
                element_positions = []
                for index, element in enumerate(dis_list):
                    if element in element_positions:
                        if element == closest: #1分2情况，在中间多出来个积木，标定这个积木是1分2
                            new_tuple = (
                                tracklist[closest][0],  # xy
                                tracklist[closest][1],  # motion
                                tracklist[closest][2],  # 连续帧
                                tracklist[closest][3],  # 失帧数
                                tracklist[closest][4],  # current_area
                                tracklist[closest][5],  # 积木类别
                                mapped_state[1]         # 遮挡类别 
                            )
                            tracklist[closest]=new_tuple
                        
                    else:
                        element_positions.append(element)      
                x, y = int(round(xy[0])), int(round(xy[1]))
                x1, y1 = max(0,x-30), max(0,y-30)
                x2, y2 = min(frame.shape[1],x+30), min(frame.shape[0],y+30)
                cv2.imwrite(os.path.join(result_dir, 
                                    f"{tracklist[closest][5]}/{frame_count}_{xy}_area({current_area})_obstruct({tracklist[closest][6]}).jpg"), 
                            frame[y1:y2, x1:x2])
            

        save_dir = effect_image
        cv2.imwrite(os.path.join(save_dir, f'frame_{frame_count:04d}.jpg'), frame)
        cv2.imshow('Frame', frame)
        out.write(frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    block_num=0        
    for i,block in enumerate(same_blocks_list):
        for j in block.track_indexes:
            block.counter += tracklist[j][2]
        if block.counter>=8:
            block_num+=1
    for i, trackitem in enumerate(tracklist):
        if trackitem[3]!=0 and trackitem[0][0]>50 and trackitem[6]!=mapped_state[1]:#多合一

            new_tuple = (
                tracklist[i][0],  # xy
                tracklist[i][1],  # motion
                tracklist[i][2],  # 连续帧
                tracklist[i][3],  # 失帧数
                tracklist[i][4],  # current_area
                tracklist[i][5],  # 积木类别
                mapped_state[2]         # 遮挡类别 
            )
            tracklist[i]=new_tuple
            
            print(result_dir, f"{trackitem[5]}/{frame_count}_{xy}_area({4})_obstruct({tracklist[i][6]}).jpg")
            x, y = int(round(xy[0])), int(round(xy[1]))
            x1, y1 = max(0,x-30), max(0,y-30)
            x2, y2 = min(640,x+30), min(480,y+30)
            width, height = 640, 480
            color = (255, 255, 255)  # 白色 (B, G, R)
            blank_image = np.full((height, width, 3), color, dtype=np.uint8)
            cv2.imwrite(os.path.join(result_dir, 
                                f"{trackitem[5]}/{frame_count}_{xy}_area({4})_obstruct({tracklist[i][6]}).jpg",),
                        blank_image[y1:y2, x1:x2])
            
    logger.info(f"block_num={block_num}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_videos(folder_path):
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mkv'))]
    
    if not video_files:
        logger.error(f"文件夹中没有视频文件: {folder_path}")
        return
    
    output_folder = "tracking"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in video_files:
        input_video = os.path.join(folder_path, video_file)
        logger.info(f"正在处理视频: {input_video}")
        video_name = os.path.splitext(video_file)[0]
        output_video = os.path.join(output_folder, f"{video_name}_tracking.mp4")
        tracking(input_video, output_video)

if __name__ == "__main__":
    video_folder = "fix/"
    process_videos(video_folder)