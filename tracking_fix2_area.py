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
    file_handler = RotatingFileHandler('tracking.log', maxBytes=1024 * 1024, backupCount=5)  # 日志文件最大1MB，保留5个备份
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

def tracking(input_video, output_video):
    # 打开视频文件
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {input_video}")
        return

    # 创建背景减除器
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=42, detectShadows=False)

    # 获取视频的帧率和尺寸
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义ROI坐标 [x_start:x_end, y_start:y_end]
    roix1, roiy1, roix2, roiy2 = 60,20,550,475
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (roix2-roix1, roiy2-roiy1), isColor=True)
    
    tracklist = []  
    trackcolor = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),(255, 255, 0), (0, 255, 255), (255, 0, 255),
        # ... [保持原有颜色列表不变] ...
    ]
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    
    # 创建保存图片的文件夹
    output_image_folder = os.path.join("tracking_images", video_name)
    zhedang_dir = os.path.join(output_image_folder, 'zhedang')
    buzhedang_dir = os.path.join(output_image_folder, 'buzhedang')
    result_dir = os.path.join(output_image_folder,'result')
    bg_dir = os.path.join(output_image_folder, 'bg')
    inflate_dir = os.path.join(output_image_folder, 'inflate')
    
    # 创建所有需要的目录
    for d in [output_image_folder, zhedang_dir, buzhedang_dir, bg_dir, inflate_dir,result_dir]:
        os.makedirs(d, exist_ok=True)

    saved_tracks = set()
    frame_count = 0

    # 遮挡检测参数
    OCCLUSION_AREA_RATIO = 1.8  # 最大面积/平均面积阈值
    MIN_CONTOUR_AREA = 60        # 最小有效轮廓面积

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[roiy1:roiy2, roix1:roix2]
        
        # 背景减除处理 
        fgmask = fgbg.apply(frame)
        
        # 保存背景处理中间结果
        cv2.imwrite(os.path.join(bg_dir, f'{frame_count}.jpg'), fgmask)
        
        # 形态学处理 
        fgmask = cv2.erode(fgmask, np.ones((3,3), np.uint8), iterations=1)
        fgmask = cv2.dilate(fgmask, np.ones((3,3), np.uint8), iterations=1)
        cv2.imwrite(os.path.join(inflate_dir, f'{frame_count}.jpg'), fgmask)

        # 轮廓检测（修改为记录面积信息）
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        contour_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_CONTOUR_AREA < area < 490*450:
                valid_contours.append(contour)
                contour_areas.append(area)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # 计算面积特征 
        avg_area = np.mean(contour_areas) if contour_areas else 0
        if(avg_area==0):
            continue
        max_area = np.max(contour_areas) if contour_areas else 0
        # print(f"avg_area ={avg_area}")
        # 更新所有目标的丢失计数器 
        tracklist = [ (t[0], t[1], t[2], t[3]+1) for t in tracklist ]
        # 提取当前帧的积木中心点（修改为包含面积信息）
        xylist = [(x + w/2, y + h/2) for cnt in valid_contours 
                 for x,y,w,h in [cv2.boundingRect(cnt)]]

        # 目标匹配逻辑
        if not tracklist:
            for xy in xylist:
                tracklist.append((xy, (0, 0), 1, 0))  # 新增目标
        else:
            for xy in xylist:
                dis = 1000
                closest = 0
                for i, trackitem in enumerate(tracklist):
                    predict = (trackitem[0][0] + trackitem[1][0], 
                             trackitem[0][1] + trackitem[1][1])
                    current_dis = math.dist(xy, predict)
                    if current_dis < min(45, dis):
                        dis = current_dis
                        closest = i
                if dis == 1000:
                    tracklist.append((xy, (0, 0), 1, 0))  # 新增目标
                else:
                    motion = (xy[0] - tracklist[closest][0][0], 
                            xy[1] - tracklist[closest][0][1])
                    # 丢失计数-1 
                    tracklist[closest] = (xy, motion, tracklist[closest][2]+1, tracklist[closest][3]-1)
                    cv2.circle(frame, (round(xy[0]), round(xy[1])), 3, 
                             trackcolor[closest % len(trackcolor)], 2)

        occlusion_reasons = []
        # 存在异常大面积区域
        if avg_area > 50 and max_area > avg_area * OCCLUSION_AREA_RATIO:
            occlusion_reasons.append("LargeArea")

        # 分类保存帧
        save_dir = zhedang_dir if occlusion_reasons else buzhedang_dir
        cv2.imwrite(os.path.join(save_dir, f'frame_{frame_count:04d}.jpg'), frame)
        
        # 在画面显示遮挡状态 
        if occlusion_reasons:
            text = f"Occlusion: {','.join(occlusion_reasons)}"
            cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 
                     0.7, (0,0,255), 2)
            logger.info(f"帧 {frame_count} 遮挡原因: {text}")
            continue
        

        # 保存积木图像
        for i,trackitem in enumerate(tracklist):
            if trackitem[3]>0:
                tracklist[i] = (tracklist[i][0], tracklist[i][1], 0, tracklist[i][3])
        for i, trackitem in enumerate(tracklist):
                if trackitem[2] > 5 and i not in saved_tracks :
                    x, y = int(round(trackitem[0][0])), int(round(trackitem[0][1]))
                    x1, y1 = max(0,x-30), max(0,y-30)
                    x2, y2 = min(frame.shape[1],x+30), min(frame.shape[0],y+30)
                    cv2.imwrite(os.path.join(result_dir, 
                                        f"track_{i}_frame_{frame_count}.jpg"), 
                                frame[y1:y2, x1:x2])
                    saved_tracks.add(i)
        # print(len(saved_tracks))


        # 显示和写入视频 
        cv2.imshow('Frame', frame)
        out.write(frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # num=0
    # for i, trackitem in enumerate(tracklist):
    #     if trackitem[2] > 5  :# trackitem[0][0]<400:#最开始两个遮挡的积木进入采集区 无法判断是否遮挡。
    #         num+=1
    #     else :
    #         pattern = "*.jpg"
    #         import glob
    #         if i in saved_tracks:    
    #             search_pattern = os.path.join(result_dir, f"track_{i}_frame_{pattern}")
    #             matching_files = glob.glob(search_pattern)
    #             for file_path in matching_files:
    #                 try:
    #                     os.remove(file_path)
    #                     print(f"成功删除图片文件: {file_path}")
    #                 except Exception as e:
                        # print(f"删除文件时出错: {file_path} - {e}")
    print(f"保存积木数量:{len(saved_tracks)} ")
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
# 遍历文件夹并处理所有视频文件
def process_videos(folder_path):        # 显示前景掩码
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