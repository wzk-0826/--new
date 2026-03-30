import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置初始收盘价变动绝对值累计阈值和增量
size = 50    # 初始收盘价变动绝对值累计阈值
add_size = 20   # 每次累加的阈值增量
baseline = "RB99_1m_PriceChange_new_axis_"  # 保存文件的前缀

for j in range(5):  # 循环测试不同的阈值
    # 读取原始数据
    file_name = 'RB99_1m_2010-01-05_2025-10-27.csv'  # 原始数据文件
    df = pd.read_csv(file_name)
    
    # 初始化缓存和结果列表
    buf_o = []  # 开盘价缓存
    buf_h = []  # 最高价缓存
    buf_l = []  # 最低价缓存
    buf_c = []  # 收盘价缓存
    buf_v = []  # 成交量缓存
    res = []    # 结果列表
    
    pos_sum = 0        # 价格变动累计值
    prev_close = None  # 前一个收盘价
    
    # 遍历原始数据生成新的时间桶
    for i in range(len(df)):
        p_t = df['datetime'].iloc[i]     # 时间
        p_o = df['open'].iloc[i]         # 开盘价
        p_h = df['high'].iloc[i]         # 最高价
        p_l = df['low'].iloc[i]          # 最低价
        p_c = df['close'].iloc[i]        # 收盘价
        p_v = df['volume'].iloc[i]       # 成交量
        
        di = df.index.values[i]  # 当前行索引
        
        # 将当前数据添加到缓存
        buf_o.append(p_o)
        buf_h.append(p_h)
        buf_l.append(p_l)
        buf_c.append(p_c)
        buf_v.append(p_v)
        
        # 计算收盘价变动绝对值并累计（跳过第一根K线）
        if prev_close is not None:
            price_change = abs(p_c - prev_close)
            pos_sum += price_change
        
        # 更新前一个收盘价
        prev_close = p_c
        
        # 当累计价格变动达到阈值时，生成一个新的时间桶
        if pos_sum >= size and prev_close is not None:  # 确保至少有一个价格变动
            # 计算新K线的OHLCV
            o = buf_o[0]      # 第一个开盘价
            h = max(buf_h)    # 最高价
            l = min(buf_l)    # 最低价
            c = buf_c[-1]     # 最后一个收盘价
            v = sum(buf_v)    # 总成交量
            p = pos_sum       # 累计价格变动
            
            # 添加到结果列表
            res.append({
                'eob': p_t,     # 结束时间
                'open': o,      # 开盘价
                'high': h,      # 最高价
                'low': l,       # 最低价
                'close': c,     # 收盘价
                'volume': v,    # 成交量
                'price_change_sum': p,  # 累计价格变动（重命名更清晰）
                'index': di     # 原始索引
            })
            
            # 重置缓存和累计值
            buf_o = []
            buf_h = []
            buf_l = []
            buf_c = []
            buf_v = []
            pos_sum = 0
    
    # 转换结果为DataFrame
    result_df = pd.DataFrame(res)
    
    # 分割训练集和测试集（以2023-01-01为界）
    test_mask = result_df['eob'] > '2023-01-01'
    test_count = len(result_df[test_mask])
    total_count = len(result_df)
    
    # 保存结果到CSV文件
    output_file = f'./temp/{baseline}{size}_{total_count}_{test_count}.csv'
    result_df.to_csv(output_file, index=False)
    print(f'已保存文件: {output_file}, 总条数: {total_count}, 测试集条数: {test_count}')
    
    # 更新阈值用于下次循环
    size += add_size
    
    # 读取生成的数据进行分析
    bars = pd.read_csv(output_file)
    bars.set_index('eob', inplace=True)
    
    # 计算不同周期的对数收益率
    returns_1 = np.log(bars['close']).diff().dropna()
    returns_2 = np.log(bars['close']).diff(periods=2).dropna()
    returns_3 = np.log(bars['close']).diff(periods=3).dropna()
    returns_4 = np.log(bars['close']).diff(periods=4).dropna()
    returns_5 = np.log(bars['close']).diff(periods=5).dropna()
    
    # 标准化收益率
    standard_1 = (returns_1 - returns_1.mean()) / returns_1.std()
    standard_2 = (returns_2 - returns_2.mean()) / returns_2.std()
    standard_3 = (returns_3 - returns_3.mean()) / returns_3.std()
    standard_4 = (returns_4 - returns_4.mean()) / returns_4.std()
    standard_5 = (returns_5 - returns_5.mean()) / returns_5.std()
    
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
    plt.title(f"{baseline}{size-add_size}_{total_count}_{test_count}_No{j+1}", 
              loc='center', fontsize=20, fontweight="bold", fontname="Times New Roman")
    plt.xlim(-5, 5)
    plt.grid(True)
    
    # 保存图表
    plt.savefig(f"{output_file}_No{j+1}.jpg")
    plt.close()
    print(f'已保存图表: {output_file}_No{j+1}.jpg')

print("基于收盘价变动绝对值累计的时间桶生成完成！")