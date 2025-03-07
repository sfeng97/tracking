import cv2
import os
import sys
import logging
import colorlog

# 确保提供了足够的命令行参数
if len(sys.argv) != 3:
    print("Usage: python video_to_image.py <input_directory> <output_directory>")
    exit(1)

# 从命令行参数获取输入和输出目录
video_dir = sys.argv[1]  # 视频文件夹路径
base_output_dir = sys.argv[2]  # 基础输出目录

# 配置带颜色的logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    }
))

file_handler = logging.FileHandler('video_processing.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger()
logger.addHandler(handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# 确保输入目录存在
if not os.path.exists(video_dir):
    logger.error(f"Error: Video directory {video_dir} does not exist")
    exit(1)

# 获取所有视频文件
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
logger.info(f"Found {len(video_files)} video files to process")

for video_file in video_files:
    # 创建对应的输出目录
    video_name = os.path.splitext(video_file)[0]
    output_dir = os.path.join(base_output_dir, video_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理每个视频文件
    video_path = os.path.join(video_dir, video_file)
    logger.info(f"Attempting to open video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_file}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video info - Frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_name = f"frame_{frame_count:04d}.jpg"
        img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(img_path, frame)
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {frame_count} frames from {video_file} to {output_dir}")

logger.info("All videos processed successfully!")
