import os
import re
import shutil
import numpy as np
from pathlib import Path

class FolderProcessor:
    def __init__(self):
        self.folder_markers = {}  # 存储文件夹标记结果
        self.block_num = 0
        self.obstruction = ''

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
        mapped_state = {
            0: 'state1',  # 将 'state1' 映射到列表中的第一个元素（索引0）
            1: 'state2',  # 将 'state2' 映射到列表中的第二个元素（索引1）
            2: 'state3'   # 将 'state3' 映射到列表中的第三个元素（索引2）
        }
        max_state_num = 0  # 用于记录最大的状态数字
        # 第一阶段：收集文件信息
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isdir(file_path) or not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            valid_image_files.append(file_path)
            area = self.extract_area(filename)
            state_str = self.extract_obstruct_value(filename)
            if state_str and state_str.startswith('state'):
                try:
                    current_num = int(state_str[5:])  # 提取state后的数字
                    if current_num > max_state_num:
                        max_state_num = current_num
                except ValueError:
                    continue  # 忽略无效的状态格式
            if area is not None:
                area_data.append(area)
                file_info.append((file_path, area))
        if max_state_num >= 1:
            key = max_state_num - 1  # 转换为mapped_state的键
            self.obstruction = mapped_state.get(key, f'state{max_state_num}')
        else:
            self.obstruction = None  # 或设置为默认值
        # 跳过无效文件夹
        if len(area_data) < 6 :
            print(f"跳过 {folder_path}（数据不足）")
            return
        # 处理异常文件
        mean = np.mean(area_data)
        std = np.std(area_data)
        abnormal_dir = os.path.join(folder_path, "不正常")
        os.makedirs(abnormal_dir, exist_ok=True)

        # 移动异常文件
        # for path, area in file_info:
        #     if self.is_abnormal(area, mean, std):
        #         self.move_abnormal_file(path, abnormal_dir, abnormal_files)

        # # 删除原始异常文件
        # for file_path in abnormal_files:
        #     try:
        #         os.remove(file_path)
        #         # print(f"已删除异常文件: {file_path}")
        #     except OSError as e:
        #         print(f"无法删除文件 {file_path}: {e.strerror}")
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
        multiple_entries= {k:v for k,v in number_stats.items() if v > 4}
        self.handle_files(folder_path, remaining_files, multiple_entries.keys())
        
        # 计算各个计数

        counts = {
            'multiple_count2':len(multiple_entries2),
            'single_count': len(single_entries),
            'multiple_count3': len(multiple_entries3),
            'multiple_count4': len(multiple_entries4),
            'multiple_count': len(multiple_entries)
            
        }
        marker="无效积木"
        if self.obstruction==mapped_state[0]:#正常
        # 找到最大的计数和对应的变量名
            # max_count_key, max_count_value = max(counts.items(), key=lambda item: item[1])
            self.block_num+=1
            marker = "一个积木"
        elif self.obstruction==mapped_state[1]:#一分多
            # max_count_key = 'multiple_count2'
        
            if counts['multiple_count4'] !=0:
                self.block_num+=4
                self.handle_files(folder_path, remaining_files, multiple_entries2.keys())
                self.handle_files(folder_path, remaining_files, multiple_entries4.keys())
                self.handle_files(folder_path, remaining_files, single_entries.keys())            
                marker = "四个积木"
            elif  counts['multiple_count3'] !=0:
                self.block_num+=3
                marker = "三个积木"
                self.handle_files(folder_path, remaining_files, multiple_entries2.keys())
                self.handle_files(folder_path, remaining_files, multiple_entries4.keys())
                self.handle_files(folder_path, remaining_files, single_entries.keys())
            elif counts['multiple_count2'] !=0:
                self.block_num+=2  
                self.handle_files(folder_path, remaining_files, single_entries.keys())
                marker = "两个积木" 
                
        elif self.obstruction==mapped_state[2]:#duo一 
            self.block_num+=1
            # if counts['multiple_count4'] !=0:
            #     self.block_num+=4        
            #     marker = "四个积木"
            # elif  counts['multiple_count3'] !=0:
            #     self.block_num+=3
            #     marker = "三个积木"
            # elif counts['multiple_count2'] !=0:
            #     self.block_num+=2  
            #     marker = "两个积木" 
                
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

    # @staticmethod
    # def extract_area(filename):
    #     """提取面积值"""
    #     match = re.search(r"area\(([\d.]+)\)", filename)
    #     print(match)
    #     return float(match.group(1)) if match else None
    @staticmethod
    def extract_area(filename):
        """提取面积值"""
        # 修改正则表达式以匹配新的格式，只提取 area 括号内的第一个数字部分
        match = re.search(r"area\(([\d.]+)(?:_[^)]*)?\)", filename)
        # 打印匹配对象用于调试（可选）
        # print(match)
        # 如果找到匹配，则返回第一个捕获组（面积值），否则返回 None
        return float(match.group(1)) if match else None
    @staticmethod
    def extract_obstruct_value(filename):
        match = re.search(r"obstruct\(([^)]+)\)", filename)
        return match.group(1) if match else None
 
    @staticmethod
    def is_abnormal(value, mean, std, sigma=1):
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
    # 获取所有符合要求的result目录
    base_path = Path("./tracking_images")
    result_dirs = []
    
    # 查找所有tracking_images/xxx/result目录
    for parent_dir in base_path.glob("*"):
        if parent_dir.is_dir():
            result_dir = parent_dir / "result"
            if result_dir.exists() and result_dir.is_dir():
                result_dirs.append(result_dir)
    
    if not result_dirs:
        raise FileNotFoundError("未找到任何有效的result目录")
    
    processor = FolderProcessor()
    
    # 处理每个找到的result目录
    for result_dir in result_dirs:
        print(f"\n{'='*30}")
        print(f"开始处理目录: {result_dir}")
        processor.process_images(result_dir)
        print(f"总积木数量: {processor.block_num}")
        processor.block_num=0
        
    
    # 输出最终统计结果
    # print("\n\n处理完成，汇总信息：")
    # print(f"总积木数量: {processor.block_num}")
    # print("各文件夹标记：")
    # for folder, marker in processor.folder_markers.items():
    #     print(f"- {folder} : {marker}")
