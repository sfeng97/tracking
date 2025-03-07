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
        if len(area_data) < 6:
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
                # print(f"已删除异常文件: {file_path}")
            except OSError as e:
                print(f"无法删除文件 {file_path}: {e.strerror}")
        # 跳过无效文件夹
        if len(file_path) < 6:
            print(f"跳过 {file_path}（剩余数据不足）")
            return

        # 第二阶段：统计剩余文件
        remaining_files = [f for f in valid_image_files if f not in abnormal_files]
        number_stats = self.collect_number_stats(remaining_files)
        
        # 统计数量
        multiple_entries2 = {k:v for k,v in number_stats.items() if v == 2}
        # multiple_count2 = len(multiple_entries2)
        single_entries = {k:v for k,v in number_stats.items() if v == 1}
        # single_count = len(single_entries)
        multiple_entries3 = {k:v for k,v in number_stats.items() if v == 3}
        # multiple_count3 = len(multiple_entries3)
        multiple_entries4 = {k:v for k,v in number_stats.items() if v == 4}
        # multiple_count4 = len(multiple_entries4)
        
        # 计算各个计数

        counts = {
            'multiple_count2':len(multiple_entries2),
            'single_count': len(single_entries),
            'multiple_count3': len(multiple_entries3),
            'multiple_count4': len(multiple_entries4),
        }
        
        # 找到最大的计数和对应的变量名
        max_count_key, max_count_value = max(counts.items(), key=lambda item: item[1])
        # print(max_count_value)
        if max_count_key == 'multiple_count2':
            marker = "两个积木"
            self.handle_files(folder_path, remaining_files, multiple_entries3.keys())
            self.handle_files(folder_path, remaining_files, multiple_entries4.keys())
            self.handle_files(folder_path, remaining_files, single_entries.keys())
            self.block_num+=2
        elif max_count_key == 'single_count':
            marker = "单个积木"
            self.handle_files(folder_path, remaining_files, multiple_entries3.keys())
            self.handle_files(folder_path, remaining_files, multiple_entries4.keys())
            self.handle_files(folder_path, remaining_files, multiple_entries2.keys())
            self.block_num+=1
        elif max_count_key == 'multiple_count3':
            marker = "三个积木"
            self.handle_files(folder_path, remaining_files, multiple_entries2.keys())
            self.handle_files(folder_path, remaining_files, multiple_entries4.keys())
            self.handle_files(folder_path, remaining_files, single_entries.keys())
            self.block_num+=3
        elif max_count_key == 'multiple_count4':
            marker = "四个积木"
            self.handle_files(folder_path, remaining_files, multiple_entries2.keys())
            self.handle_files(folder_path, remaining_files, multiple_entries4.keys())
            self.handle_files(folder_path, remaining_files, single_entries.keys())
            self.block_num+=4
        else:
            marker = "未确定"
        # 记录标记结果
        self.folder_markers[folder_path] = marker
        # print(f"\n{folder_path} 标记为：{marker}")

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
                # print(f"已清理文件: {file_path}")
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

def get_latest_result_dir(base_path="tracking_images"):
    base = Path(base_path)
    
    # 获取所有包含result的子目录
    candidates = []
    for parent_dir in base.iterdir():
        if parent_dir.is_dir():
            result_dir = parent_dir / "result"
            if result_dir.exists() and result_dir.is_dir():
                candidates.append(parent_dir)
    
    if not candidates:
        raise FileNotFoundError("未找到任何有效的result目录")
    
    # 按目录名降序排序（假设目录名包含时间信息）
    sorted_dirs = sorted(candidates, key=lambda x: x.name, reverse=True)
    
    # 或者按修改时间排序（如果需要根据实际创建时间）
    # sorted_dirs = sorted(candidates, key=lambda x: os.path.getmtime(x), reverse=True)
    
    latest_parent = sorted_dirs[0]
    return latest_parent / "result"
if __name__ == "__main__":
    try:
        # 获取最新result目录
        latest_result = get_latest_result_dir()
        print(f"最新result目录: {latest_result}")
        # 初始化处理器
        processor = FolderProcessor()
        processor.process_images(latest_result)
    except Exception as e:
        print(f"发生错误: {str(e)}")