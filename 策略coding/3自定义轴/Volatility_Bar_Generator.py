import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 基于波动率的自定义时间桶生成器
# 波动率维度能够更好地捕捉市场的不确定性和风险水平变化

def calculate_rolling_volatility(prices, window=5):
    """
    计算滚动波动率（使用对数收益率的标准差）
    
    参数:
    - prices: 价格序列
    - window: 滚动窗口大小
    
    返回:
    - 波动率序列
    """
    returns = np.log(prices).diff().dropna()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # 年化波动率
    return volatility

def calculate_atr(high, low, close, window=14):
    """
    计算真实波动幅度均值(ATR)
    
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - window: 窗口大小
    
    返回:
    - ATR序列
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
    
    return df['atr']

def generate_volatility_bars(file_path, initial_vol_threshold, vol_increment, num_iterations=5, 
                             method='atr', window=14):
    """
    生成基于波动率累计阈值的自定义时间桶
    
    参数:
    - file_path: 原始数据文件路径
    - initial_vol_threshold: 初始波动率阈值
    - vol_increment: 每次迭代的阈值增量
    - num_iterations: 迭代次数
    - method: 波动率计算方法 ('atr' 或 'std')
    - window: 波动率计算窗口
    
    返回:
    - 生成的文件列表
    """
    # 读取原始数据
    df = pd.read_csv(file_path)
    
    # 计算波动率
    if method == 'atr':
        # 使用ATR作为波动率指标
        vol_series = calculate_atr(df['high'], df['low'], df['close'], window)
        vol_name = 'ATR'
    else:
        # 使用滚动标准差作为波动率指标
        vol_series = calculate_rolling_volatility(df['close'], window)
        vol_name = 'StdDev'
    
    # 保存生成的文件路径
    generated_files = []
    
    current_threshold = initial_vol_threshold
    
    for i in range(num_iterations):
        print(f"\n处理第 {i+1} 次迭代，{vol_name}阈值: {current_threshold}")
        
        # 初始化缓存和结果列表
        buf_o = []  # 开盘价缓存
        buf_h = []  # 最高价缓存
        buf_l = []  # 最低价缓存
        buf_c = []  # 收盘价缓存
        buf_v = []  # 成交量缓存
        res = []    # 结果列表
        
        vol_sum = 0  # 累计波动率
        
        # 遍历原始数据生成新的时间桶
        for j in range(window, len(df)):  # 从window开始，因为需要计算波动率
            # 获取当前行数据
            timestamp = df['datetime'].iloc[j]
            open_price = df['open'].iloc[j]
            high_price = df['high'].iloc[j]
            low_price = df['low'].iloc[j]
            close_price = df['close'].iloc[j]
            volume = df['volume'].iloc[j]
            volatility = vol_series.iloc[j]
            
            # 将当前数据添加到缓存
            buf_o.append(open_price)
            buf_h.append(high_price)
            buf_l.append(low_price)
            buf_c.append(close_price)
            buf_v.append(volume)
            
            # 累计波动率
            if not np.isnan(volatility):
                vol_sum += volatility
            
            # 当累计波动率达到阈值时，生成一个新的时间桶
            if vol_sum >= current_threshold and len(buf_o) > 0:
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
                    'vol_sum': vol_sum, # 实际累计的波动率
                    'last_vol': volatility if not np.isnan(volatility) else 0,  # 最后一个波动率
                    'index': j          # 原始索引
                })
                
                # 重置缓存和累计值
                buf_o = []
                buf_h = []
                buf_l = []
                buf_c = []
                buf_v = []
                vol_sum = 0
        
        # 转换结果为DataFrame
        result_df = pd.DataFrame(res)
        
        # 分割训练集和测试集（以2023-01-01为界）
        test_mask = result_df['eob'] > '2023-01-01'
        test_count = len(result_df[test_mask])
        total_count = len(result_df)
        
        print(f"生成的时间桶总数: {total_count}, 测试集数量: {test_count}")
        
        # 保存结果到CSV文件
        output_file = f'./temp-Volatility_Bar_Generator/RB99_1m_Volatility_{vol_name}_{current_threshold}_{total_count}_{test_count}.csv'
        result_df.to_csv(output_file, index=False)
        generated_files.append(output_file)
        print(f"已保存文件: {output_file}")
        
        # 绘制收益率分布图表
        plot_return_distribution(result_df, output_file, i+1, current_threshold, total_count, test_count, vol_name)
        
        # 更新阈值用于下次迭代
        current_threshold += vol_increment
    
    return generated_files

def plot_return_distribution(bars_df, file_path, iteration_num, threshold, total_count, test_count, vol_name):
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
    plt.title(f"Volatility_{vol_name}_{threshold}_{total_count}_{test_count}_No{iteration_num}", 
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
    initial_vol = 100        # 初始波动率阈值
    vol_increment = 50       # 每次迭代的阈值增量
    num_iterations = 5       # 迭代次数
    method = 'atr'           # 波动率计算方法 ('atr' 或 'std')
    window = 14              # 波动率计算窗口
    
    print("开始生成基于波动率的自定义时间桶...")
    print(f"原始数据文件: {file_path}")
    print(f"初始波动率阈值: {initial_vol}")
    print(f"每次增量: {vol_increment}")
    print(f"迭代次数: {num_iterations}")
    print(f"波动率计算方法: {method.upper()}")
    print(f"计算窗口: {window}")
    
    # 执行生成
    files = generate_volatility_bars(file_path, initial_vol, vol_increment, num_iterations, method, window)
    
    print(f"\n生成完成!共生成 {len(files)} 个文件。")
