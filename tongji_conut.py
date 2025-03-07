import os
import re
import shutil
import numpy as np
from pathlib import Path

class FolderProcessor:
    def __init__(self):
        self.folder_markers = {}  # 存储文件夹标记结果
        self.block_num = 0

    def process_images(self, root_dir="result"):
        """处理主函数"""
        # 遍历根目录下的所有子文件夹
        for entry in os.scandir(root_dir):
            if not entry.is_dir():
                continue
            self.process_folder(entry.path)
        
        # 输出所有文件夹标记
        print("\n文件夹标记汇总：")
        for folder, marker in self.folder_markers.items():
            print(f"{folder} : {marker}")
        print(f"积木总数：{self.block_num}")
    def process_folder(self, folder_path):
        """处理单个文件夹"""
        # 初始化数据存储
        
        area_data = []
        file_info = []
        abnormal_files = []
        valid_image_files = []

        # 第一阶段：收集文件信息
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isdir(file_path) or not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            valid_image_files.append(file_path)
            area = self.extract_area(filename)
            if area is not None:
                area_data.append(area)
                file_info.append((file_path, area))

        # 跳过无效文件夹
        if len(area_data) < 2:
            print(f"跳过 {folder_path}（数据不足）")
            return

        # 处理异常文件
        mean = np.mean(area_data)
        std = np.std(area_data)
        abnormal_dir = os.path.join(folder_path, "不正常")
        os.makedirs(abnormal_dir, exist_ok=True)

        # 移动异常文件
        for path, area in file_info:
            if self.is_abnormal(area, mean, std):
                self.move_abnormal_file(path, abnormal_dir, abnormal_files)

        # 删除原始异常文件
        for file_path in abnormal_files:
            try:
                os.remove(file_path)
                print(f"已删除异常文件: {file_path}")
            except OSError as e:
                print(f"无法删除文件 {file_path}: {e.strerror}")

        # 第二阶段：统计剩余文件
        remaining_files = [f for f in valid_image_files if f not in abnormal_files]
        number_stats = self.collect_number_stats(remaining_files)
        
        # 统计数量
        multiple_entries = {k:v for k,v in number_stats.items() if v >= 2}
        multiple_count = len(multiple_entries)
        single_entries = {k:v for k,v in number_stats.items() if v == 1}
        single_count = len(single_entries)

        # 判断并处理文件
        marker = ""
        if single_count > multiple_count:
            marker = "单个积木"
            self.block_num+=1
            self.handle_files(folder_path, remaining_files, multiple_entries.keys())
        elif multiple_count > single_count:
            self.block_num+=2
            marker = "多个积木"
            self.handle_files(folder_path, remaining_files, single_entries.keys())
        else:
            self.block_num+=0
            marker = "未确定"

        # 记录标记结果
        self.folder_markers[folder_path] = marker
        print(f"\n{folder_path} 标记为：{marker}")

    def handle_files(self, folder_path, files, target_numbers):
        """处理目标文件"""
        abnormal_dir = os.path.join(folder_path, "不正常")
        os.makedirs(abnormal_dir, exist_ok=True)

        # 收集需要处理的文件
        to_process = []
        for file_path in files:
            filename = os.path.basename(file_path)
            prefix = filename.split('_', 1)[0]
            try:
                if int(prefix) in target_numbers:
                    to_process.append(file_path)
            except ValueError:
                continue

        # 移动并删除文件
        moved_files = []
        for src_path in to_process:
            dest_path = os.path.join(abnormal_dir, os.path.basename(src_path))
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(os.path.basename(src_path))
                dest_path = os.path.join(abnormal_dir, f"{name}_{counter}{ext}")
                counter += 1
            shutil.copy2(src_path, dest_path)
            moved_files.append(src_path)

        # 删除原始文件
        for file_path in moved_files:
            try:
                os.remove(file_path)
                print(f"已清理文件: {file_path}")
            except OSError as e:
                print(f"文件清理失败: {file_path} - {e.strerror}")

    def collect_number_stats(self, files):
        """收集数字统计信息"""
        number_stats = {}
        for file_path in files:
            filename = os.path.basename(file_path)
            parts = filename.split('_', 1)
            if len(parts) < 1:
                continue
            try:
                num = int(parts[0])
                number_stats[num] = number_stats.get(num, 0) + 1
            except ValueError:
                continue
        return number_stats

    @staticmethod
    def extract_area(filename):
        """提取面积值"""
        match = re.search(r"area\(([\d.]+)\)", filename)
        return float(match.group(1)) if match else None

    @staticmethod
    def is_abnormal(value, mean, std, sigma=1.5):
        """异常检测"""
        if std == 0:
            return False
        return abs((value - mean) / std) > sigma

    @staticmethod
    def move_abnormal_file(src_path, dest_dir, record_list):
        """移动异常文件"""
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, filename)
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
            counter += 1
        shutil.copy2(src_path, dest_path)
        record_list.append(src_path)

if __name__ == "__main__":
    root = Path("./tracking_images/20250307_15/result")
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")
    
    processor = FolderProcessor()
    processor.process_images(root)