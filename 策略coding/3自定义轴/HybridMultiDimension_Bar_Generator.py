import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 混合多维度组合的自定义时间桶生成器
# 综合多种市场维度，提供更全面的市场信息捕捉能力

def calculate_volume_intensity(volume, window=5):
    """
    计算交易量强度指标
    使用标准化交易量作为强度指标
    
    参数:
    - volume: 成交量序列
    - window: 滚动窗口大小
    
    返回:
    - 交易量强度序列
    """
    # 计算滚动均值和标准差
    volume_mean = volume.rolling(window=window).mean()
    volume_std = volume.rolling(window=window).std()
    
    # 标准化交易量，避免除以零
    volume_std = volume_std.replace(0, 1e-10)
    volume_intensity = (volume - volume_mean) / volume_std
    
    return volume_intensity

def calculate_volatility(high, low, close, window=14):
    """
    计算真实波动幅度均值(ATR)作为波动率指标
    
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - window: 窗口大小
    
    返回:
    - 标准化ATR序列
    """
    # 计算真实波动幅度
    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    })
    
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 计算ATR（移动平均）
    df['atr'] = df['tr'].rolling(window=window).mean()
    
    # 标准化ATR
    atr_mean = df['atr'].rolling(window=window).mean()
    atr_std = df['atr'].rolling(window=window).std()
    atr_std = atr_std.replace(0, 1e-10)
    df['normalized_atr'] = (df['atr'] - atr_mean) / atr_std
    
    return df['normalized_atr']

def calculate_momentum(prices, window=14):
    """
    计算价格动量指标
    
    参数:
    - prices: 价格序列
    - window: 计算窗口大小
    
    返回:
    - 标准化动量序列
    """
    # 计算动量
    momentum = prices.diff(periods=window)
    
    # 标准化动量
    momentum_mean = momentum.rolling(window=window).mean()
    momentum_std = momentum.rolling(window=window).std()
    momentum_std = momentum_std.replace(0, 1e-10)
    normalized_momentum = (momentum - momentum_mean) / momentum_std
    
    return normalized_momentum

def calculate_microstructure(high, low, close, volume, window=5):
    """
    计算市场微观结构特征
    使用(high-low)/close作为流动性的代理指标
    
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - volume: 成交量序列
    - window: 滚动窗口大小
    
    返回:
    - 标准化微观结构特征序列
    """
    # 计算价差代理
    spread_proxy = (high - low) / close
    
    # 标准化价差代理
    spread_mean = spread_proxy.rolling(window=window).mean()
    spread_std = spread_proxy.rolling(window=window).std()
    spread_std = spread_std.replace(0, 1e-10)
    normalized_spread = (spread_proxy - spread_mean) / spread_std
    
    return normalized_spread

def generate_hybrid_bars(file_path, initial_threshold, threshold_increment, num_iterations=5,
                        volume_weight=0.25, volatility_weight=0.25, momentum_weight=0.25, 
                        microstructure_weight=0.25, combination_type='any'):
    """
    生成基于混合多维度组合的自定义时间桶
    
    参数:
    - file_path: 原始数据文件路径
    - initial_threshold: 初始阈值
    - threshold_increment: 每次迭代的阈值增量
    - num_iterations: 迭代次数
    - volume_weight: 交易量维度权重
    - volatility_weight: 波动率维度权重
    - momentum_weight: 动量维度权重
    - microstructure_weight: 微观结构维度权重
    - combination_type: 组合类型 ('any' 任一维度达标, 'all' 所有维度达标, 'weighted' 加权平均达标)
    
    返回:
    - 生成的文件列表
    """
    # 读取原始数据
    df = pd.read_csv(file_path)
    
    # 计算各维度指标
    volume_intensity = calculate_volume_intensity(df['volume'])
    volatility = calculate_volatility(df['high'], df['low'], df['close'])
    momentum = calculate_momentum(df['close'])
    microstructure = calculate_microstructure(df['high'], df['low'], df['close'], df['volume'])
    
    # 保存生成的文件路径
    generated_files = []
    
    current_threshold = initial_threshold
    
    for i in range(num_iterations):
        print(f"\n处理第 {i+1} 次迭代，阈值: {current_threshold}")
        print(f"维度权重 - 交易量: {volume_weight}, 波动率: {volatility_weight}, 动量: {momentum_weight}, 微观结构: {microstructure_weight}")
        print(f"组合类型: {combination_type}")
        
        # 初始化缓存和结果列表
        buf_o = []  # 开盘价缓存
        buf_h = []  # 最高价缓存
        buf_l = []  # 最低价缓存
        buf_c = []  # 收盘价缓存
        buf_v = []  # 成交量缓存
        res = []    # 结果列表
        
        # 初始化各维度累计值
        volume_sum = 0
        volatility_sum = 0
        momentum_sum = 0
        microstructure_sum = 0
        
        # 确定有效的起始索引
        start_idx = 14  # 使用最大的窗口大小
        
        # 遍历原始数据生成新的时间桶
        for j in range(start_idx, len(df)):
            # 获取当前行数据
            timestamp = df['datetime'].iloc[j]
            open_price = df['open'].iloc[j]
            high_price = df['high'].iloc[j]
            low_price = df['low'].iloc[j]
            close_price = df['close'].iloc[j]
            volume = df['volume'].iloc[j]
            
            # 获取当前各维度指标值
            vol_int = volume_intensity.iloc[j]
            volat = volatility.iloc[j]
            mom = momentum.iloc[j]
            micro = microstructure.iloc[j]
            
            # 处理NaN值
            vol_int = 0 if np.isnan(vol_int) else vol_int
            volat = 0 if np.isnan(volat) else volat
            mom = 0 if np.isnan(mom) else mom
            micro = 0 if np.isnan(micro) else micro
            
            # 将当前数据添加到缓存
            buf_o.append(open_price)
            buf_h.append(high_price)
            buf_l.append(low_price)
            buf_c.append(close_price)
            buf_v.append(volume)
            
            # 累计各维度指标
            volume_sum += abs(vol_int)
            volatility_sum += abs(volat)
            momentum_sum += abs(mom)
            microstructure_sum += abs(micro)
            
            # 判断是否生成新的时间桶，根据组合类型
            generate_bar = False
            
            if combination_type == 'any':
                # 任一维度达标
                if (volume_sum >= current_threshold * volume_weight or 
                    volatility_sum >= current_threshold * volatility_weight or 
                    momentum_sum >= current_threshold * momentum_weight or 
                    microstructure_sum >= current_threshold * microstructure_weight) and len(buf_o) > 0:
                    generate_bar = True
            elif combination_type == 'all':
                # 所有维度达标
                if (volume_sum >= current_threshold * volume_weight and 
                    volatility_sum >= current_threshold * volatility_weight and 
                    momentum_sum >= current_threshold * momentum_weight and 
                    microstructure_sum >= current_threshold * microstructure_weight) and len(buf_o) > 0:
                    generate_bar = True
            else:  # weighted
                # 加权平均达标
                weighted_sum = (volume_sum * volume_weight + 
                              volatility_sum * volatility_weight + 
                              momentum_sum * momentum_weight + 
                              microstructure_sum * microstructure_weight)
                if weighted_sum >= current_threshold and len(buf_o) > 0:
                    generate_bar = True
            
            # 生成新的时间桶
            if generate_bar:
                # 计算新K线的OHLCV
                o = buf_o[0]      # 第一个开盘价
                h = max(buf_h)    # 最高价
                l = min(buf_l)    # 最低价
                c = buf_c[-1]     # 最后一个收盘价
                v = sum(buf_v)    # 总成交量
                
                # 计算加权综合指标
                weighted_sum = (volume_sum * volume_weight + 
                              volatility_sum * volatility_weight + 
                              momentum_sum * momentum_weight + 
                              microstructure_sum * microstructure_weight)
                
                # 添加到结果列表
                res.append({
                    'eob': timestamp,            # 结束时间
                    'open': o,                  # 开盘价
                    'high': h,                  # 最高价
                    'low': l,                   # 最低价
                    'close': c,                 # 收盘价
                    'volume': v,                # 成交量
                    'volume_sum': volume_sum,   # 累计交易量强度
                    'volatility_sum': volatility_sum,  # 累计波动率
                    'momentum_sum': momentum_sum,      # 累计动量
                    'microstructure_sum': microstructure_sum,  # 累计微观结构
                    'weighted_sum': weighted_sum,      # 加权综合指标
                    'index': j                  # 原始索引
                })
                
                # 重置缓存和累计值
                buf_o = []
                buf_h = []
                buf_l = []
                buf_c = []
                buf_v = []
                volume_sum = 0
                volatility_sum = 0
                momentum_sum = 0
                microstructure_sum = 0
        
        # 转换结果为DataFrame
        result_df = pd.DataFrame(res)
        
        # 分割训练集和测试集（以2023-01-01为界）
        test_mask = result_df['eob'] > '2023-01-01'
        test_count = len(result_df[test_mask])
        total_count = len(result_df)
        
        print(f"生成的时间桶总数: {total_count}, 测试集数量: {test_count}")
        
        # 保存结果到CSV文件
        output_file = f'./temp/RB99_1m_Hybrid_{combination_type}_{current_threshold}_{total_count}_{test_count}.csv'
        result_df.to_csv(output_file, index=False)
        generated_files.append(output_file)
        print(f"已保存文件: {output_file}")
        
        # 绘制收益率分布图表
        plot_return_distribution(result_df, output_file, i+1, current_threshold, total_count, test_count, combination_type)
        
        # 更新阈值用于下次迭代
        current_threshold += threshold_increment
    
    return generated_files

def plot_return_distribution(bars_df, file_path, iteration_num, threshold, total_count, test_count, combination_type):
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
    plt.title(f"Hybrid_{combination_type}_{threshold}_{total_count}_{test_count}_No{iteration_num}", 
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
    initial_threshold = 100     # 初始阈值
    threshold_increment = 50    # 每次迭代的阈值增量
    num_iterations = 5          # 迭代次数
    
    # 维度权重（可以根据需要调整）
    volume_weight = 0.25        # 交易量维度权重
    volatility_weight = 0.25    # 波动率维度权重
    momentum_weight = 0.25      # 动量维度权重
    microstructure_weight = 0.25 # 微观结构维度权重
    
    # 组合类型: 'any'（任一维度达标）, 'all'（所有维度达标）, 'weighted'（加权平均达标）
    combination_type = 'weighted'
    
    print("开始生成基于混合多维度组合的自定义时间桶...")
    print(f"原始数据文件: {file_path}")
    print(f"初始阈值: {initial_threshold}")
    print(f"每次增量: {threshold_increment}")
    print(f"迭代次数: {num_iterations}")
    print(f"组合类型: {combination_type}")
    print(f"维度权重分布:")
    print(f"- 交易量: {volume_weight}")
    print(f"- 波动率: {volatility_weight}")
    print(f"- 动量: {momentum_weight}")
    print(f"- 微观结构: {microstructure_weight}")
    
    # 执行生成
    files = generate_hybrid_bars(file_path, initial_threshold, threshold_increment, num_iterations,
                               volume_weight, volatility_weight, momentum_weight, 
                               microstructure_weight, combination_type)
    
    print(f"\n生成完成！共生成 {len(files)} 个文件。")
    print("混合多维度时间桶生成器的特点：")
    print("1. 综合多种市场维度，提供更全面的市场信息捕捉能力")
    print("2. 支持灵活的权重配置，可以根据策略需求调整各维度的重要性")
    print("3. 提供多种组合策略（任一维度达标、所有维度达标、加权平均达标）")
    print("4. 能够捕捉到更复杂的市场动态和信息流动")
    print("5. 特别适合需要综合考虑多种市场因素的复杂策略")
    print("6. 可以根据不同市场环境动态调整权重配置，提高策略的适应性")