import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 基于交易量的自定义时间桶生成器
# 交易量是最基本且常用的自定义时间桶维度，相比传统时间维度，能更好地反映市场活动强度

def generate_volume_bars(file_path, initial_volume_threshold, volume_increment, num_iterations=5):
    """
    生成基于交易量累计阈值的自定义时间桶
    
    参数:
    - file_path: 原始数据文件路径
    - initial_volume_threshold: 初始交易量阈值
    - volume_increment: 每次迭代的阈值增量
    - num_iterations: 迭代次数
    
    返回:
    - 生成的文件列表
    """
    # 读取原始数据
    df = pd.read_csv(file_path)
    
    # 保存生成的文件路径
    generated_files = []
    
    current_threshold = initial_volume_threshold
    
    for i in range(num_iterations):
        print(f"\n处理第 {i+1} 次迭代，交易量阈值: {current_threshold}")
        
        # 初始化缓存和结果列表
        buf_o = []  # 开盘价缓存
        buf_h = []  # 最高价缓存
        buf_l = []  # 最低价缓存
        buf_c = []  # 收盘价缓存
        buf_v = []  # 成交量缓存
        res = []    # 结果列表
        
        volume_sum = 0  # 累计成交量
        
        # 遍历原始数据生成新的时间桶
        for j in range(len(df)):
            # 获取当前行数据
            timestamp = df['datetime'].iloc[j]
            open_price = df['open'].iloc[j]
            high_price = df['high'].iloc[j]
            low_price = df['low'].iloc[j]
            close_price = df['close'].iloc[j]
            volume = df['volume'].iloc[j]
            
            # 将当前数据添加到缓存
            buf_o.append(open_price)
            buf_h.append(high_price)
            buf_l.append(low_price)
            buf_c.append(close_price)
            buf_v.append(volume)
            
            # 累计成交量
            volume_sum += volume
            
            # 当累计成交量达到阈值时，生成一个新的时间桶
            if volume_sum >= current_threshold:
                # 计算新K线的OHLCV
                o = buf_o[0]      # 第一个开盘价
                h = max(buf_h)    # 最高价
                l = min(buf_l)    # 最低价
                c = buf_c[-1]     # 最后一个收盘价
                v = sum(buf_v)    # 总成交量
                
                # 添加到结果列表
                res.append({
                    'eob': timestamp,    # 结束时间
                    'open': o,          # 开盘价
                    'high': h,          # 最高价
                    'low': l,           # 最低价
                    'close': c,         # 收盘价
                    'volume': v,        # 成交量
                    'volume_sum': volume_sum,  # 实际累计的成交量
                    'index': j          # 原始索引
                })
                
                # 重置缓存和累计值
                buf_o = []
                buf_h = []
                buf_l = []
                buf_c = []
                buf_v = []
                volume_sum = 0
        
        # 转换结果为DataFrame
        result_df = pd.DataFrame(res)
        
        # 分割训练集和测试集（以2023-01-01为界）
        test_mask = result_df['eob'] > '2023-01-01'
        test_count = len(result_df[test_mask])
        total_count = len(result_df)
        
        print(f"生成的时间桶总数: {total_count}, 测试集数量: {test_count}")
        
        # 保存结果到CSV文件
        output_file = f'./temp-Volume_Bar_Generator/RB99_1m_Volume_Bar_{current_threshold}_{total_count}_{test_count}.csv'
        result_df.to_csv(output_file, index=False)
        generated_files.append(output_file)
        print(f"已保存文件: {output_file}")
        
        # 绘制收益率分布图表
        plot_return_distribution(result_df, output_file, i+1, current_threshold, total_count, test_count)
        
        # 更新阈值用于下次迭代
        current_threshold += volume_increment
    
    return generated_files

def plot_return_distribution(bars_df, file_path, iteration_num, threshold, total_count, test_count):
    """
    绘制收益率分布的KDE图
    """
    # 计算不同周期的对数收益率
    bars_df.set_index('eob', inplace=True)
    returns_1 = np.log(bars_df['close']).diff().dropna()
    returns_2 = np.log(bars_df['close']).diff(periods=2).dropna()
    returns_3 = np.log(bars_df['close']).diff(periods=3).dropna()
    returns_4 = np.log(bars_df['close']).diff(periods=4).dropna()
    returns_5 = np.log(bars_df['close']).diff(periods=5).dropna()
    
    # 标准化收益率
    standard_1 = (returns_1 - returns_1.mean()) / returns_1.std() if returns_1.std() > 0 else returns_1
    standard_2 = (returns_2 - returns_2.mean()) / returns_2.std() if returns_2.std() > 0 else returns_2
    standard_3 = (returns_3 - returns_3.mean()) / returns_3.std() if returns_3.std() > 0 else returns_3
    standard_4 = (returns_4 - returns_4.mean()) / returns_4.std() if returns_4.std() > 0 else returns_4
    standard_5 = (returns_5 - returns_5.mean()) / returns_5.std() if returns_5.std() > 0 else returns_5
    
    # 绘制收益率分布的KDE图
    plt.figure(figsize=(16, 12))
    
    # 绘制各周期收益率的KDE曲线
    sns.kdeplot(standard_1, label="1", color='darkred')
    sns.kdeplot(standard_2, label="2", color='green')
    sns.kdeplot(standard_3, label="3", color='blue')
    sns.kdeplot(standard_4, label="4", color='orange')
    sns.kdeplot(standard_5, label="5", color='magenta')
    
    # 绘制标准正态分布作为参考
    sns.kdeplot(np.random.normal(size=1000000), label="Normal", color='black', linestyle="--")
    
    # 设置图表属性
    plt.xticks(range(-5, 6))
    plt.legend(loc=8, ncol=5)
    plt.title(f"Volume_Bar_{threshold}_{total_count}_{test_count}_No{iteration_num}", 
              loc='center', fontsize=20, fontweight="bold", fontname="Times New Roman")
    plt.xlim(-5, 5)
    plt.grid(True)
    
    # 保存图表
    plt.savefig(f"{file_path}_No{iteration_num}.jpg")
    plt.close()
    print(f"已保存图表: {file_path}_No{iteration_num}.jpg")

# 主函数
if __name__ == "__main__":
    # 参数设置
    file_path = "RB99_1m_2010-01-05_2025-10-27.csv"  # 原始数据文件
    initial_volume = 10000    # 初始交易量阈值
    volume_increment = 5000   # 每次迭代的阈值增量
    num_iterations = 5        # 迭代次数
    
    print("开始生成基于交易量的自定义时间桶...")
    print(f"原始数据文件: {file_path}")
    print(f"初始交易量阈值: {initial_volume}")
    print(f"每次增量: {volume_increment}")
    print(f"迭代次数: {num_iterations}")
    
    # 执行生成
    files = generate_volume_bars(file_path, initial_volume, volume_increment, num_iterations)
    
    print(f"\n生成完成!共生成 {len(files)} 个文件。")
    print("交易量时间桶生成器的特点：")
    print("1. 相比时间维度，交易量维度能更好地反映市场活动强度")
    print("2. 在市场波动剧烈时，时间桶会更密集；市场平静时，时间桶会更稀疏")
    print("3. 适合需要捕捉市场活跃度变化的策略")
    print("4. 可以避免传统时间维度中交易量不均匀导致的信息密度不一致问题")