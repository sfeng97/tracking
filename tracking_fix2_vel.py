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
    console_formatter = logging.Formatter('\033[92m%(asctime)s - %(levelname)s - %(message)s\033[0m')  # 绿色日志
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
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)

    # 获取视频的帧率和尺寸
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义ROI坐标 [x_start:x_end, y_start:y_end]
    roix1, roiy1, roix2, roiy2 = 60,20,550,475
    # roix1, roiy1, roix2, roiy2 = 60, 10, 580, 470
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (roix2-roix1, roiy2-roiy1), isColor=True)
    
    # tracking用数据结构
    tracklist = []  # [x,y,dirx,diry,len]
    SMOOTHING_FACTOR = 0.1
    trackcolor = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),(255, 255, 0), (0, 255, 255), (255, 0, 255),
        (255, 128, 0), (128, 255, 0), (128, 0, 255), (255, 255, 128), (128, 255, 255), (255, 128, 255),
        (255, 0, 128), (0, 255, 128), (0, 128, 255), (255, 128, 128), (128, 255, 128), (128, 128, 255),
        (128, 0, 128), (0, 128, 128), (0, 128, 128), (0, 0, 128), (128, 0, 0), (0, 128, 0),
        (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (0, 192, 192), (192, 0, 192),
        (192, 64, 0), (64, 192, 0), (64, 0, 192), (192, 192, 64), (64, 192, 192), (192, 64, 192),
        (192, 0, 64), (0, 192, 64), (0, 64, 192), (192, 64, 64), (64, 192, 64), (64, 64, 192),
        (64, 0, 64), (0, 64, 64), (0, 64, 64), (0, 0, 64), (64, 0, 0), (0, 64, 0)
    ]
    video_name = os.path.splitext(os.path.basename(input_video))[0]  # 新增代码
    
    # 创建保存图片的文件夹
    output_image_folder = os.path.join("tracking_images", video_name)  # 修改代码
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder, exist_ok=True)  # 自动创建嵌套目录

    saved_tracks = set()
    frame_count = 0
    bg_dir=output_image_folder+'/bg'
    inflate_dir=output_image_folder+'/inflate'
    while True:
        ret, frame = cap.read()
        if not ret:#读取下一帧失败
            break
        frame = frame[roiy1:roiy2, roix1:roix2]
                # 显示原始帧

        # 应用背景减除器
        fgmask = fgbg.apply(frame)
        # 显示前景掩码             
        # cv2.imshow('bg', fgmask)
        filename = os.path.join(bg_dir, f'{frame_count}.jpg')
        cv2.imwrite(filename, fgmask)
        if fgmask is None:
            logger.error("背景减除器未能生成前景掩码")
            continue
        index = np.nonzero(fgmask)

        # print(fgmask[index])

        # 去除噪声
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel=np.ones((3, 3), np.uint8))   
        cv2.imshow('inflate', fgmask)
        filename = os.path.join(inflate_dir, f'{frame_count}.jpg')
        cv2.imwrite(filename, fgmask)
        # 找到前景物体的轮廓
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 取前景物体的中心点
        xylist = []
        # 可视化每个前景物体
        for contour in contours:

            if cv2.contourArea(contour) > 100:  # 过滤掉太小的轮廓

                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                xylist.append((x + w/2, y + h/2))
                # print(f"帧号{frame_count},contours数量{len(contours)}")
            # print(f"轮廓面积{cv2.contourArea(contour)}")
        cv2.imshow('Foreground Mask', frame)
        if (frame_count==210):
            cv2.waitKey(0)
        if not tracklist:
            for xy in xylist:
                print(f"frame_count{frame_count},xy{xy}")
                tracklist.append((xy, (0, 0), 1))
        else:
            for xy in xylist:
                min_distance = 50  # 匹配阈值
                best_match = None
                best_dot = -float('inf')  # 方向一致性得分

                for i, trackitem in enumerate(tracklist):
                    # 使用平滑后的速度进行预测
                    predicted_x = trackitem[0][0] + trackitem[1][0]
                    predicted_y = trackitem[0][1] + trackitem[1][1]
                    predicted_pos = (predicted_x, predicted_y)
                    
                    # 计算实际距离
                    distance = math.dist(xy, predicted_pos)
                    
                    # 计算方向一致性（向量点积）
                    dx = xy[0] - trackitem[0][0]
                    dy = xy[1] - trackitem[0][1]
                    dot_product = dx * trackitem[1][0] + dy * trackitem[1][1]

                    # 双重匹配条件：距离接近且方向一致
                    if distance < min_distance and dot_product > 0:
                        # 优先选择距离更近且方向更一致的
                        if distance < min_distance or (distance == min_distance and dot_product > best_dot):
                            min_distance = distance
                            best_dot = dot_product
                            best_match = i

                if best_match is not None:
                    # 使用平滑更新速度
                    old_x, old_y = tracklist[best_match][0]
                    old_vx, old_vy = tracklist[best_match][1]
                    
                    # 计算当前帧速度
                    current_vx = xy[0] - old_x
                    current_vy = xy[1] - old_y
                    
                    # 应用速度平滑
                    new_vx = SMOOTHING_FACTOR * old_vx + (1-SMOOTHING_FACTOR) * current_vx
                    new_vy = SMOOTHING_FACTOR * old_vy + (1-SMOOTHING_FACTOR) * current_vy
                    
                    # 更新轨迹
                    tracklist[best_match] = (xy, (new_vx, new_vy), tracklist[best_match][2]+1)
                    
                    # 绘制跟踪点
                    cv2.circle(frame, (round(xy[0]), round(xy[1])), 3, trackcolor[best_match % len(trackcolor)], 2)
                    # 绘制速度矢量
                    cv2.arrowedLine(frame, 
                                  (int(round(xy[0])), int(round(xy[1]))),
                                  (int(round(xy[0] + new_vx*5)), int(round(xy[1] + new_vy*5))),  # 放大5倍显示
                                  trackcolor[best_match % len(trackcolor)], 
                                  1)
                else:
                    # 新增轨迹的逻辑保持不变
                    tracklist.append((xy, (0, 0), 1))
        # 检查是否需要保存积木图片
        for i, trackitem in enumerate(tracklist):
            if trackitem[2] > 5 and i not in saved_tracks:
                x, y = int(round(trackitem[0][0])), int(round(trackitem[0][1]))
                # 截取积木区域（50x50）
                x1 = max(0, x - 25)
                y1 = max(0, y - 25)
                x2 = min(frame.shape[1], x + 25)
                y2 = min(frame.shape[0], y + 25)
                block_image = frame[y1:y2, x1:x2]
                image_path = os.path.join(output_image_folder, f"track_{i}_frame_{frame_count}.jpg")
                cv2.imwrite(image_path, block_image)
                # logger.info(f"保存积木 {i} 的图片: {image_path}")
                saved_tracks.add(i)
        cv2.imshow('Frame', frame)
        frame_count += 1  # 帧计数器递增
        cv2.imshow('Foreground Mask', fgmask)
        
        # 写入输出视频
        out.write(frame)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #    break
    # 统计积木数量
    cnt = 0
    for trackitem in tracklist:
        if trackitem[2] > 5:
            cnt += 1

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    logger.info(f"跟踪到的积木数量: {cnt}个。视频已保存到 {output_video}")

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
