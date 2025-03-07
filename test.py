from dataclasses import dataclass

@dataclass
class TrackBlock:
    track_indexes: list
    min_index: int
    counter: int = 0

# 初始化列表
same_blocks_list = [
    TrackBlock([1,2], 2), 
    TrackBlock([4], 3)
]

# 场景1：给第一个区块追加跟踪点
same_blocks_list[0].track_indexes.append(3)

# 场景2：创建新区块并追加
new_block = TrackBlock(track_indexes=[5], min_index=4)
same_blocks_list.append(new_block)

# 场景3：通过min_index查找并追加
for block in same_blocks_list:
    if block.min_index == 3:  # 查找min_index=3的区块
        block.track_indexes.append(6)
        block.counter += 1    # 修改其他属性

# 最终结果
print(same_blocks_list)
# 输出：
# [
#   TrackBlock(track_indexes=[1, 2, 3], min_index=2, counter=0),
#   TrackBlock(track_indexes=[4, 6], min_index=3, counter=1),
#   TrackBlock(track_indexes=[5], min_index=4, counter=0)
# ]