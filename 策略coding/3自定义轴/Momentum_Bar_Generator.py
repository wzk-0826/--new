import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 基于价格动量的自定义时间桶生成器
# 动量维度能够更好地捕捉市场的方向性变化和趋势强度

def calculate_momentum(prices, window=14):
    """
    计算价格动量指标
    
    参数:
    - prices: 价格序列
    - window: 计算窗口大小
    
    返回:
    - 动量序列
    """
    momentum = prices.diff(periods=window)
    return momentum

def calculate_rsi(prices, window=14):
    """
    计算相对强弱指标(RSI)
    
    参数:
    - prices: 价格序列
    - window: 计算窗口大小
    
    返回:
    - RSI序列
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # 避免除以零
    avg_loss = avg_loss.replace(0, 1e-10)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # 归一化RSI为[-1, 1]区间
    normalized_rsi = (rsi - 50) / 50
    
    return normalized_rsi

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    计算MACD指标
    
    参数:
    - prices: 价格序列
    - fast_period: 快线周期
    - slow_period: 慢线周期
    - signal_period: 信号线周期
    
    返回:
    - MACD柱状图序列
    """
    exp1 = prices.ewm(span=fast_period, adjust=False).mean()
    exp2 = prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    return macd_hist

def generate_momentum_bars(file_path, initial_momentum_threshold, momentum_increment, 
                          num_iterations=5, method='momentum', window=14):
    """
    生成基于价格动量累计阈值的自定义时间桶
    
    参数:
    - file_path: 原始数据文件路径
    - initial_momentum_threshold: 初始动量阈值
    - momentum_increment: 每次迭代的阈值增量
    - num_iterations: 迭代次数
    - method: 动量计算方法 ('momentum', 'rsi', 'macd')
    - window: 计算窗口
    
    返回:
    - 生成的文件列表
    """
    # 读取原始数据
    df = pd.read_csv(file_path)
    
    # 计算动量指标
    if method == 'rsi':
        # 使用RSI作为动量指标
        momentum_series = calculate_rsi(df['close'], window)
        momentum_name = 'RSI'
    elif method == 'macd':
        # 使用MACD柱状图作为动量指标
        momentum_series = calculate_macd(df['close'])
        momentum_name = 'MACD'
    else:
        # 使用价格变化作为动量指标
        momentum_series = calculate_momentum(df['close'], window)
        momentum_name = 'Momentum'
    
    # 保存生成的文件路径
    generated_files = []
    
    current_threshold = initial_momentum_threshold
    
    for i in range(num_iterations):
        print(f"\n处理第 {i+1} 次迭代，{momentum_name}阈值: {current_threshold}")
        
        # 初始化缓存和结果列表
        buf_o = []  # 开盘价缓存
        buf_h = []  # 最高价缓存
        buf_l = []  # 最低价缓存
        buf_c = []  # 收盘价缓存
        buf_v = []  # 成交量缓存
        res = []    # 结果列表
        
        pos_momentum_sum = 0  # 累计正向动量
        neg_momentum_sum = 0  # 累计负向动量
        
        # 确定有效的起始索引（基于方法类型）
        start_idx = max(window, 26)  # MACD需要至少26个数据点
        
        # 遍历原始数据生成新的时间桶
        for j in range(start_idx, len(df)):
            # 获取当前行数据
            timestamp = df['datetime'].iloc[j]
            open_price = df['open'].iloc[j]
            high_price = df['high'].iloc[j]
            low_price = df['low'].iloc[j]
            close_price = df['close'].iloc[j]
            volume = df['volume'].iloc[j]
            
            # 获取当前动量值
            momentum = momentum_series.iloc[j]
            if np.isnan(momentum):
                momentum = 0
            
            # 将当前数据添加到缓存
            buf_o.append(open_price)
            buf_h.append(high_price)
            buf_l.append(low_price)
            buf_c.append(close_price)
            buf_v.append(volume)
            
            # 累计动量，区分正负
            if momentum > 0:
                pos_momentum_sum += abs(momentum)
            else:
                neg_momentum_sum += abs(momentum)
            
            # 当累计动量达到阈值时，生成一个新的时间桶
            # 这里考虑正向和负向动量的绝对值累计
            if (pos_momentum_sum >= current_threshold or neg_momentum_sum >= current_threshold) and len(buf_o) > 0:
                # 计算新K线的OHLCV
                o = buf_o[0]      # 第一个开盘价
                h = max(buf_h)    # 最高价
                l = min(buf_l)    # 最低价
                c = buf_c[-1]     # 最后一个收盘价
                v = sum(buf_v)    # 总成交量
                
                # 添加到结果列表
                res.append({
                    'eob': timestamp,            # 结束时间
                    'open': o,                  # 开盘价
                    'high': h,                  # 最高价
                    'low': l,                   # 最低价
                    'close': c,                 # 收盘价
                    'volume': v,                # 成交量
                    'pos_momentum_sum': pos_momentum_sum,  # 累计正向动量
                    'neg_momentum_sum': neg_momentum_sum,  # 累计负向动量
                    'last_momentum': momentum,              # 最后一个动量值
                    'index': j                  # 原始索引
                })
                
                # 重置缓存和累计值
                buf_o = []
                buf_h = []
                buf_l = []
                buf_c = []
                buf_v = []
                pos_momentum_sum = 0
                neg_momentum_sum = 0
        
        # 转换结果为DataFrame
        result_df = pd.DataFrame(res)
        
        # 分割训练集和测试集（以2023-01-01为界）
        test_mask = result_df['eob'] > '2023-01-01'
        test_count = len(result_df[test_mask])
        total_count = len(result_df)
        
        print(f"生成的时间桶总数: {total_count}, 测试集数量: {test_count}")
        
        # 保存结果到CSV文件
        output_file = f'./temp-Momentum_Bar_Generator/RB99_1m_Momentum_{momentum_name}_{current_threshold}_{total_count}_{test_count}.csv'
        result_df.to_csv(output_file, index=False)
        generated_files.append(output_file)
        print(f"已保存文件: {output_file}")
        
        # 绘制收益率分布图表
        plot_return_distribution(result_df, output_file, i+1, current_threshold, total_count, test_count, momentum_name)
        
        # 更新阈值用于下次迭代
        current_threshold += momentum_increment
    
    return generated_files

def plot_return_distribution(bars_df, file_path, iteration_num, threshold, total_count, test_count, momentum_name):
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
    plt.title(f"Momentum_{momentum_name}_{threshold}_{total_count}_{test_count}_No{iteration_num}", 
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
    # 使用相对路径或绝对路径确保能找到数据文件
    file_path = "./RB99_1m_2010-01-05_2025-10-27.csv"  
    # 或者使用绝对路径
    # file_path = "d:/desk/AI策略coding/3自定义轴/RB99_1m_2010-01-05_2025-10-27.csv"
    initial_momentum = 500     # 初始动量阈值
    momentum_increment = 250   # 每次迭代的阈值增量
    num_iterations = 5         # 迭代次数
    method = 'momentum'        # 动量计算方法 ('momentum', 'rsi', 'macd')
    window = 14                # 动量计算窗口
    
    print("开始生成基于价格动量的自定义时间桶...")
    print(f"原始数据文件: {file_path}")
    print(f"初始动量阈值: {initial_momentum}")
    print(f"每次增量: {momentum_increment}")
    print(f"迭代次数: {num_iterations}")
    print(f"动量计算方法: {method.upper()}")
    print(f"计算窗口: {window}")
    
    # 执行生成
    files = generate_momentum_bars(file_path, initial_momentum, momentum_increment, 
                                  num_iterations, method, window)
    
    print(f"\n生成完成！共生成 {len(files)} 个文件。")
    print("价格动量时间桶生成器的特点：")
    print("1. 能够更好地捕捉市场的方向性变化和趋势强度")
    print("2. 在价格快速变动时期自动生成更密集的时间桶")
    print("3. 特别适合趋势跟踪类策略，能够在趋势形成过程中提供更细粒度的数据")
    print("4. 可以使用不同的动量计算方法（价格变化、RSI、MACD等）来适应不同的市场特性")
    print("5. 能够自动调整时间粒度以适应市场的节奏变化，捕捉更多的趋势转折点")