import pandas as pd
import numpy as np
from pytdx.hq import TdxHq_API
import matplotlib.pyplot as plt
import datetime
import os

# --------------------------
# 设置股票池示例（沪深A股前100只）
# --------------------------
stock_list = []
with TdxHq_API() as api:
    if api.connect('119.147.212.81', 7709):
        for market in [0, 1]:  # 0=深圳, 1=上海
            for code in range(0, 100):
                stock_code = f'{code:06d}'
                stock_list.append((market, stock_code))

# --------------------------
# 指标计算函数
# --------------------------
def SMA(series, period):
    return series.rolling(period).mean()

def WR(high, low, close, period):
    HHV = high.rolling(period).max()
    LLV = low.rolling(period).min()
    return 100 * (HHV - close) / (HHV - LLV)

def RSI(close, period=9):
    diff = close.diff()
    up = np.where(diff > 0, diff, 0)
    down = np.abs(np.where(diff < 0, diff, 0))
    return SMA(pd.Series(up), period) / (SMA(pd.Series(up), period) + SMA(pd.Series(down), period)) * 100

# --------------------------
# 主逻辑：计算每只股票是否符合XG条件
# --------------------------
results = []

for market, code in stock_list:
    with TdxHq_API() as api:
        if not api.connect('119.147.212.81', 7709):
            continue
        data = api.get_security_bars(9, market, code, 0, 60)  # 最近60日K线
        if not data:
            continue
        df = pd.DataFrame(data)
        df.rename(columns={'close':'CLOSE','high':'HIGH','low':'LOW'}, inplace=True)
        df = df[::-1]  # 正序排列

        df['WR1'] = WR(df['HIGH'], df['LOW'], df['CLOSE'], 10)
        df['WR2'] = WR(df['HIGH'], df['LOW'], df['CLOSE'], 20)
        df['WR3'] = (df['WR1'] < 20) & (df['WR2'] < 20)
        df['RSI3'] = RSI(df['CLOSE'],9) > 70

        # 市值及天数简化条件
        df['市值及天数'] = True

        df['XGTJ'] = df['WR3'] & df['RSI3'] & df['市值及天数']
        df['XG'] = df['XGTJ'].rolling(21).sum() > 4

        if df['XG'].iloc[-1]:
            results.append(code)

# --------------------------
# 保存当日选股数量
# --------------------------
os.makedirs('output', exist_ok=True)
today_str = datetime.datetime.now().strftime("%Y-%m-%d")
count_file = 'output/XG_21days.csv'

# 读取历史数据
if os.path.exists(count_file):
    count_df = pd.read_csv(count_file)
else:
    count_df = pd.DataFrame(columns=['Date','Count'])

# 添加今天的数据
count_df = pd.concat([count_df, pd.DataFrame([[today_str, len(results)]], columns=['Date','Count'])], ignore_index=True)
count_df = count_df.tail(21)  # 保留最近21天
count_df.to_csv(count_file, index=False)

# --------------------------
# 绘制折线图
# --------------------------
plt.figure(figsize=(12,6))
plt.plot(count_df['Date'], count_df['Count'], marker='o', color='blue', linewidth=2)
plt.xticks(rotation=45)
plt.xlabel("日期")
plt.ylabel("符合XG条件的股票数量")
plt.title("过去21天符合XG条件的股票数量趋势")
plt.grid(True)
plt.tight_layout()
plt.savefig('output/XG_21days.png')
plt.show()

print(f"今天符合条件的股票数量: {len(results)}")
