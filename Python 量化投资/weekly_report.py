import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# SQLite数据库路径
DB_PATH = '/home/bxdm/PycharmProjects/PythonProject/stock_data.db'

def get_last_week_range():
    """
    获取上周的日期范围（周一到周五）
    """
    today = datetime.now()
    # 找到本周周一
    days_since_monday = today.weekday()
    this_monday = today - timedelta(days=days_since_monday)

    # 上周的范围
    last_monday = this_monday - timedelta(days=7)
    last_friday = last_monday + timedelta(days=4)

    return last_monday.strftime('%Y%m%d'), last_friday.strftime('%Y%m%d')

def get_all_stocks():
    """
    获取数据库中所有股票列表
    """
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT DISTINCT stock_code, name FROM stock_price ORDER BY stock_code"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_stock_week_data(stock_code, start_date, end_date):
    """
    获取指定股票上周的交易数据
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT trade_date, stock_code, name, open, close, high, low, volume, turnover, change, pct_chg
    FROM stock_price 
    WHERE stock_code = ? AND trade_date >= ? AND trade_date <= ?
    ORDER BY trade_date ASC
    """
    df = pd.read_sql_query(query, conn, params=(stock_code, start_date, end_date))
    conn.close()
    return df

def calculate_week_performance(df):
    """
    计算一周的表现指标
    """
    if df.empty:
        return None

    # 基本数据
    first_day = df.iloc[0]
    last_day = df.iloc[-1]

    # 周开盘价和周收盘价
    week_open = first_day['open']
    week_close = last_day['close']

    # 周最高价和最低价
    week_high = df['high'].max()
    week_low = df['low'].min()

    # 周涨跌幅
    week_change = week_close - week_open
    week_change_pct = (week_change / week_open * 100) if week_open != 0 else 0

    # 周成交量和成交额
    week_volume = df['volume'].sum()
    week_turnover = df['turnover'].sum()

    # 交易天数
    trading_days = len(df)

    # 日均成交量
    avg_daily_volume = week_volume / trading_days if trading_days > 0 else 0

    # 振幅
    amplitude = ((week_high - week_low) / week_open * 100) if week_open != 0 else 0

    return {
        'stock_code': first_day['stock_code'],
        'name': first_day['name'],
        'week_open': week_open,
        'week_close': week_close,
        'week_high': week_high,
        'week_low': week_low,
        'week_change': week_change,
        'week_change_pct': week_change_pct,
        'week_volume': week_volume,
        'week_turnover': week_turnover,
        'avg_daily_volume': avg_daily_volume,
        'amplitude': amplitude,
        'trading_days': trading_days
    }

def generate_weekly_report():
    """
    生成上周股市周报告
    """
    print("=== 开始生成股市周报告 ===")

    # 获取上周日期范围
    start_date, end_date = get_last_week_range()
    print(f"分析时间范围: {start_date} 到 {end_date}")

    # 获取所有股票
    all_stocks = get_all_stocks()
    print(f"数据库中共有 {len(all_stocks)} 只股票")

    # 存储所有股票的周表现
    weekly_performances = []

    # 处理每只股票
    for index, stock in all_stocks.iterrows():
        stock_code = stock['stock_code']
        stock_name = stock['name']

        try:
            # 获取该股票上周数据
            week_data = get_stock_week_data(stock_code, start_date, end_date)

            if not week_data.empty:
                # 计算周表现
                performance = calculate_week_performance(week_data)
                if performance:
                    weekly_performances.append(performance)

            # 进度提示
            if (index + 1) % 100 == 0:
                print(f"已处理 {index + 1}/{len(all_stocks)} 只股票...")

        except Exception as e:
            print(f"处理股票 {stock_code} ({stock_name}) 时出错: {e}")
            continue

    print(f"成功分析了 {len(weekly_performances)} 只股票的周表现")

    # 转换为DataFrame并排序
    df_performance = pd.DataFrame(weekly_performances)

    if df_performance.empty:
        print("没有获取到有效数据")
        return

    # 按周涨跌幅排序（从高到低）
    df_performance = df_performance.sort_values('week_change_pct', ascending=False).reset_index(drop=True)

    # 生成报告
    report_filename = f"股市周报告_{start_date}至{end_date}.txt"

    with open(report_filename, 'w', encoding='utf-8') as f:
        # 报告标题
        f.write("=" * 80 + "\n")
        f.write(f"股市周报告 ({start_date} 至 {end_date})\n")
        f.write("=" * 80 + "\n\n")

        # 市场概况
        f.write("【市场概况】\n")
        f.write(f"统计股票数量: {len(df_performance)} 只\n")
        f.write(f"上涨股票数量: {len(df_performance[df_performance['week_change_pct'] > 0])} 只\n")
        f.write(f"下跌股票数量: {len(df_performance[df_performance['week_change_pct'] < 0])} 只\n")
        f.write(f"平盘股票数量: {len(df_performance[df_performance['week_change_pct'] == 0])} 只\n")
        f.write(f"平均涨跌幅: {df_performance['week_change_pct'].mean():.2f}%\n")
        f.write(f"涨跌幅中位数: {df_performance['week_change_pct'].median():.2f}%\n")
        f.write(f"总成交量: {df_performance['week_volume'].sum():,.0f} 手\n")
        f.write(f"总成交额: {df_performance['week_turnover'].sum()/100000000:.2f} 亿元\n\n")

        # 涨幅榜 TOP 20
        f.write("【涨幅榜 TOP 20】\n")
        f.write(f"{'排名':<4} {'股票代码':<8} {'股票名称':<12} {'周涨跌幅':<8} {'周涨跌额':<8} {'成交量(万手)':<12} {'振幅':<8}\n")
        f.write("-" * 80 + "\n")

        top_gainers = df_performance.head(20)
        for i, row in top_gainers.iterrows():
            f.write(f"{i+1:<4} {row['stock_code']:<8} {row['name']:<12} "
                   f"{row['week_change_pct']:>7.2f}% {row['week_change']:>7.2f} "
                   f"{row['week_volume']/10000:>10.1f} {row['amplitude']:>7.2f}%\n")

        f.write("\n")

        # 跌幅榜 TOP 20
        f.write("【跌幅榜 TOP 20】\n")
        f.write(f"{'排名':<4} {'股票代码':<8} {'股票名称':<12} {'周涨跌幅':<8} {'周涨跌额':<8} {'成交量(万手)':<12} {'振幅':<8}\n")
        f.write("-" * 80 + "\n")

        top_losers = df_performance.tail(20).iloc[::-1]  # 倒序显示最大跌幅
        for i, (idx, row) in enumerate(top_losers.iterrows()):
            f.write(f"{i+1:<4} {row['stock_code']:<8} {row['name']:<12} "
                   f"{row['week_change_pct']:>7.2f}% {row['week_change']:>7.2f} "
                   f"{row['week_volume']/10000:>10.1f} {row['amplitude']:>7.2f}%\n")

        f.write("\n")

        # 成交量排行 TOP 20
        volume_top = df_performance.nlargest(20, 'week_volume')
        f.write("【成交量排行 TOP 20】\n")
        f.write(f"{'排名':<4} {'股票代码':<8} {'股票名称':<12} {'成交量(万手)':<12} {'成交额(亿元)':<12} {'周涨跌幅':<8}\n")
        f.write("-" * 80 + "\n")

        for i, (idx, row) in enumerate(volume_top.iterrows()):
            f.write(f"{i+1:<4} {row['stock_code']:<8} {row['name']:<12} "
                   f"{row['week_volume']/10000:>10.1f} {row['week_turnover']/100000000:>10.2f} "
                   f"{row['week_change_pct']:>7.2f}%\n")

        f.write("\n")

        # 振幅排行 TOP 20
        amplitude_top = df_performance.nlargest(20, 'amplitude')
        f.write("【振幅排行 TOP 20】\n")
        f.write(f"{'排名':<4} {'股票代码':<8} {'股票名称':<12} {'振幅':<8} {'周涨跌幅':<8} {'成交量(万手)':<12}\n")
        f.write("-" * 80 + "\n")

        for i, (idx, row) in enumerate(amplitude_top.iterrows()):
            f.write(f"{i+1:<4} {row['stock_code']:<8} {row['name']:<12} "
                   f"{row['amplitude']:>7.2f}% {row['week_change_pct']:>7.2f}% "
                   f"{row['week_volume']/10000:>10.1f}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")

    print(f"✓ 周报告已生成: {report_filename}")

    # 打印简要统计信息
    print(f"\n=== 简要统计 ===")
    print(f"上涨股票: {len(df_performance[df_performance['week_change_pct'] > 0])} 只")
    print(f"下跌股票: {len(df_performance[df_performance['week_change_pct'] < 0])} 只")
    print(f"平均涨跌幅: {df_performance['week_change_pct'].mean():.2f}%")
    print(f"最大涨幅: {df_performance['week_change_pct'].max():.2f}% ({df_performance.iloc[0]['name']})")
    print(f"最大跌幅: {df_performance['week_change_pct'].min():.2f}% ({df_performance.iloc[-1]['name']})")

if __name__ == "__main__":
    generate_weekly_report()