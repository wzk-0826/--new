import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 基于市场微观结构的自定义时间桶生成器
# 市场微观结构维度能够更好地捕捉市场的流动性和价格形成机制

def calculate_spread_proxy(high, low, close):
    """
    计算买卖价差的代理指标
    使用(high-low)/close作为流动性的代理指标
    
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    
    返回:
    - 买卖价差代理指标序列
    """
    spread_proxy = (high - low) / close
    return spread_proxy

def detect_price_jumps(returns, jump_threshold=3.0):
    """
    检测价格跳跃
    使用标准化收益率超过阈值的情况作为跳跃
    
    参数:
    - returns: 收益率序列
    - jump_threshold: 跳跃检测阈值
    
    返回:
    - 跳跃强度指标序列
    """
    # 标准化收益率
    returns_mean = returns.mean()
    returns_std = returns.std()
    
    # 避免除以零
    if returns_std == 0:
        returns_std = 1e-10
    
    normalized_returns = (returns - returns_mean) / returns_std
    
    # 计算跳跃强度（超出阈值的部分）
    jump_strength = np.maximum(0, abs(normalized_returns) - jump_threshold)
    
    return jump_strength

def calculate_volume_imbalance(volume, returns, window=5):
    """
    计算交易量不平衡指标
    衡量交易量与价格变动的相关性，反映买卖压力
    
    参数:
    - volume: 成交量序列
    - returns: 收益率序列
    - window: 计算窗口
    
    返回:
    - 交易量不平衡指标序列
    """
    # 创建DataFrame进行计算
    df = pd.DataFrame({
        'volume': volume,
        'returns': returns
    })
    
    # 计算方向性交易量
    df['signed_volume'] = df['volume'] * np.sign(df['returns'])
    
    # 计算移动平均不平衡
    df['imbalance'] = df['signed_volume'].rolling(window=window).mean() / df['volume'].rolling(window=window).mean()
    
    # 取绝对值作为不平衡强度
    df['imbalance_strength'] = abs(df['imbalance'])
    
    return df['imbalance_strength']

def generate_microstructure_bars(file_path, initial_micro_threshold, micro_increment, 
                                num_iterations=5, method='spread_jump', window=5):
    """
    生成基于市场微观结构特征累计阈值的自定义时间桶
    
    参数:
    - file_path: 原始数据文件路径
    - initial_micro_threshold: 初始微观结构阈值
    - micro_increment: 每次迭代的阈值增量
    - num_iterations: 迭代次数
    - method: 微观结构特征计算方法 ('spread_jump', 'volume_imbalance')
    - window: 计算窗口
    
    返回:
    - 生成的文件列表
    """
    # 读取原始数据
    df = pd.read_csv(file_path)
    
    # 计算收益率
    returns = np.log(df['close']).diff()
    
    # 计算微观结构指标
    if method == 'volume_imbalance':
        # 使用交易量不平衡作为微观结构指标
        micro_series = calculate_volume_imbalance(df['volume'], returns, window)
        micro_name = 'VolumeImbalance'
    else:
        # 综合使用买卖价差代理和价格跳跃检测
        spread_proxy = calculate_spread_proxy(df['high'], df['low'], df['close'])
        jump_strength = detect_price_jumps(returns)
        
        # 标准化并合并两种指标
        spread_normalized = (spread_proxy - spread_proxy.mean()) / (spread_proxy.std() + 1e-10)
        jump_normalized = (jump_strength - jump_strength.mean()) / (jump_strength.std() + 1e-10)
        
        # 综合指标
        micro_series = (spread_normalized + jump_normalized) / 2
        micro_name = 'SpreadJump'
    
    # 保存生成的文件路径
    generated_files = []
    
    current_threshold = initial_micro_threshold
    
    for i in range(num_iterations):
        print(f"\n处理第 {i+1} 次迭代，{micro_name}阈值: {current_threshold}")
        
        # 初始化缓存和结果列表
        buf_o = []  # 开盘价缓存
        buf_h = []  # 最高价缓存
        buf_l = []  # 最低价缓存
        buf_c = []  # 收盘价缓存
        buf_v = []  # 成交量缓存
        res = []    # 结果列表
        
        micro_sum = 0  # 累计微观结构指标
        
        # 确定有效的起始索引
        start_idx = window
        
        # 遍历原始数据生成新的时间桶
        for j in range(start_idx, len(df)):
            # 获取当前行数据
            timestamp = df['datetime'].iloc[j]
            open_price = df['open'].iloc[j]
            high_price = df['high'].iloc[j]
            low_price = df['low'].iloc[j]
            close_price = df['close'].iloc[j]
            volume = df['volume'].iloc[j]
            
            # 获取当前微观结构指标值
            micro_value = micro_series.iloc[j]
            if np.isnan(micro_value):
                micro_value = 0
            
            # 将当前数据添加到缓存
            buf_o.append(open_price)
            buf_h.append(high_price)
            buf_l.append(low_price)
            buf_c.append(close_price)
            buf_v.append(volume)
            
            # 累计微观结构指标
            micro_sum += abs(micro_value)
            
            # 当累计微观结构指标达到阈值时，生成一个新的时间桶
            if micro_sum >= current_threshold and len(buf_o) > 0:
                # 计算新K线的OHLCV
                o = buf_o[0]      # 第一个开盘价
                h = max(buf_h)    # 最高价
                l = min(buf_l)    # 最低价
                c = buf_c[-1]     # 最后一个收盘价
                v = sum(buf_v)    # 总成交量
                
                # 添加到结果列表
                res.append({
                    'eob': timestamp,           # 结束时间
                    'open': o,                 # 开盘价
                    'high': h,                 # 最高价
                    'low': l,                  # 最低价
                    'close': c,                # 收盘价
                    'volume': v,               # 成交量
                    'micro_sum': micro_sum,    # 累计微观结构指标
                    'last_micro': micro_value, # 最后一个微观结构指标值
                    'index': j                 # 原始索引
                })
                
                # 重置缓存和累计值
                buf_o = []
                buf_h = []
                buf_l = []
                buf_c = []
                buf_v = []
                micro_sum = 0
        
        # 转换结果为DataFrame
        result_df = pd.DataFrame(res)
        
        # 分割训练集和测试集（以2023-01-01为界）
        test_mask = result_df['eob'] > '2023-01-01'
        test_count = len(result_df[test_mask])
        total_count = len(result_df)
        
        print(f"生成的时间桶总数: {total_count}, 测试集数量: {test_count}")
        
        # 保存结果到CSV文件
        output_file = f'./temp/RB99_1m_Microstructure_{micro_name}_{current_threshold}_{total_count}_{test_count}.csv'
        result_df.to_csv(output_file, index=False)
        generated_files.append(output_file)
        print(f"已保存文件: {output_file}")
        
        # 绘制收益率分布图表
        plot_return_distribution(result_df, output_file, i+1, current_threshold, total_count, test_count, micro_name)
        
        # 更新阈值用于下次迭代
        current_threshold += micro_increment
    
    return generated_files

def plot_return_distribution(bars_df, file_path, iteration_num, threshold, total_count, test_count, micro_name):
    """
    绘制收益率分布的KDE图
    """
    if len(bars_df) < 6:  # 至少需要6个数据点来计算5阶差分
        print("数据点不足，无法绘制收益率分布图")
        return
    
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
    plt.title(f"Microstructure_{micro_name}_{threshold}_{total_count}_{test_count}_No{iteration_num}", 
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
    initial_micro = 100      # 初始微观结构阈值
    micro_increment = 50     # 每次迭代的阈值增量
    num_iterations = 5       # 迭代次数
    method = 'spread_jump'   # 微观结构特征计算方法 ('spread_jump', 'volume_imbalance')
    window = 5               # 计算窗口
    
    print("开始生成基于市场微观结构的自定义时间桶...")
    print(f"原始数据文件: {file_path}")
    print(f"初始微观结构阈值: {initial_micro}")
    print(f"每次增量: {micro_increment}")
    print(f"迭代次数: {num_iterations}")
    print(f"微观结构特征计算方法: {method.upper()}")
    print(f"计算窗口: {window}")
    
    # 执行生成
    files = generate_microstructure_bars(file_path, initial_micro, micro_increment, 
                                        num_iterations, method, window)
    
    print(f"\n生成完成!共生成 {len(files)} 个文件。")
