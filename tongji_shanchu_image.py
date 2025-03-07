import os
import re
import shutil
import numpy as np
from pathlib import Path

def process_images(root_dir="result"):
    """
    处理主函数
    """
    # 遍历根目录下的所有子文件夹
    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue

        process_folder(entry.path)

def process_folder(folder_path):
    """
    处理单个文件夹
    """
    # 初始化数据存储
    area_data = []
    file_info = []
    abnormal_files=[]
    # 收集文件信息
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 跳过子目录和非图片文件
        if os.path.isdir(file_path) or not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 提取面积值
        area = extract_area(filename)
        if area is not None:
            area_data.append(area)
            file_info.append((file_path, area))

    # 跳过无有效数据的文件夹
    if len(area_data) < 2:
        print(f"跳过 {folder_path}（数据不足）")
        return

    # 计算统计指标
    mean = np.mean(area_data)
    std = np.std(area_data)
    threshold = 1.5  # 可调整的sigma值

    # 创建异常目录
    abnormal_dir = os.path.join(folder_path, "不正常")
    os.makedirs(abnormal_dir, exist_ok=True)

    # 检测并移动异常文件
    abnormal_count = 0
    for path, area in file_info:
        if is_abnormal(area, mean, std, threshold):
            move_abnormal_file(path, abnormal_dir,abnormal_files)
            abnormal_count += 1
   # 删除原始文件夹中的异常文件

    for file_path in abnormal_files:
        try:
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
        except OSError as e:
            print(f"无法删除文件 {file_path}: {e.strerror}")

    print(f"处理完成：{folder_path}，发现异常 {abnormal_count} 个")

def extract_area(filename):
    """
    使用正则表达式提取面积值
    """
    match = re.search(r"area\(([\d.]+)\)", filename)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def is_abnormal(value, mean, std, sigma=2.5):
    """
    使用标准差检测异常值
    """
    if std == 0:  # 避免除零错误
        return False
    z_score = abs((value - mean) / std)
    return z_score > sigma

def move_abnormal_file(src_path, dest_dir,abnormal_files):
    """
    移动异常文件并保留原始文件名
    """
    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, filename)
    
    # 避免覆盖已有文件
    counter = 1
    while os.path.exists(dest_path):
        name, ext = os.path.splitext(filename)
        dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
        counter += 1

    shutil.copy2(src_path, dest_path)  # 保留元数据
    abnormal_files.append(src_path)

if __name__ == "__main__":
    # 使用pathlib处理路径更安全
    root = Path("./tracking_images/20250307_12/result")
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")

    process_images(root)