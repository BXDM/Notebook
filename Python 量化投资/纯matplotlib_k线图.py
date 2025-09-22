import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib

# SQLite数据库路径
DB_PATH = '/home/bxdm/PycharmProjects/PythonProject/stock_data.db'

# 设置中文字体
chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']

def get_stock_list():
    """从数据库获取股票列表"""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT DISTINCT name, stock_code FROM stock_price ORDER BY name"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_stock_data(stock_code, days=100):
    """从数据库获取股票历史数据"""
    conn = sqlite3.connect(DB_PATH)
    
    # 获取最新的N天数据
    query = """
    SELECT trade_date, stock_code, name, open, close, low, high, volume, turnover, change, pct_chg
    FROM stock_price 
    WHERE stock_code = ? 
    ORDER BY trade_date DESC 
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(stock_code, days))
    conn.close()
    
    if not df.empty:
        # 按日期正序排列（最早的在前面）
        df = df.sort_values('trade_date').reset_index(drop=True)
        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        # 重命名列以匹配原有逻辑
        df = df.rename(columns={
            'trade_date': '日期',
            'open': '开盘',
            'close': '收盘',
            'high': '最高',
            'low': '最低',
            'volume': '成交量'
        })
    
    return df

def plot_candlestick(ax, data, width=0.6):
    """
    绘制K线图（蜡烛图）- 专业版本
    影线不穿过实体，符合标准K线图规范
    """
    for i, (index, row) in enumerate(data.iterrows()):
        open_price = row['开盘']
        close_price = row['收盘']
        high_price = row['最高']
        low_price = row['最低']
        
        # 确定颜色：红涨绿跌（中国习惯）
        if close_price >= open_price:
            color = 'red'  # 上涨用红色
            edge_color = 'red'
            face_color = 'white'  # 上涨空心
        else:
            color = 'green'  # 下跌用绿色
            edge_color = 'green'
            face_color = 'green'  # 下跌实心
        
        # 计算实体的上下边界
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        body_height = body_top - body_bottom
        
        # 绘制上影线（从实体顶部到最高价）
        if high_price > body_top:
            ax.plot([i, i], [body_top, high_price], color=color, linewidth=1.5, solid_capstyle='round')
        
        # 绘制下影线（从实体底部到最低价）
        if low_price < body_bottom:
            ax.plot([i, i], [low_price, body_bottom], color=color, linewidth=1.5, solid_capstyle='round')
        
        # 绘制实体
        if body_height > 0:  # 有实体（开盘价不等于收盘价）
            rect = patches.Rectangle((i - width/2, body_bottom), width, body_height,
                                   linewidth=1.5, edgecolor=edge_color, facecolor=face_color)
            ax.add_patch(rect)
        else:
            # 十字星：开盘价等于收盘价，画十字线
            ax.plot([i - width/2, i + width/2], [open_price, open_price], 
                   color=color, linewidth=2, solid_capstyle='round')
            # 如果有影线，十字星的竖线就是影线
            if high_price > open_price or low_price < open_price:
                ax.plot([i, i], [low_price, high_price], color=color, linewidth=1.5, solid_capstyle='round')

def plot_stock_kline_matplotlib(stock_name, days=100):
    """
    使用纯 matplotlib 绘制K线图 - SQLite数据源版本
    """
    try:
        # 从数据库获取股票列表
        print(f"搜索股票: {stock_name}")
        stock_info = get_stock_list()
        
        # 精确匹配
        matched_stocks = stock_info[stock_info['name'] == stock_name]
        
        # 如果没有精确匹配，尝试模糊匹配
        if matched_stocks.empty:
            print(f"精确匹配未找到，尝试模糊搜索...")
            matched_stocks = stock_info[stock_info['name'].str.contains(stock_name, na=False)]
            
            if matched_stocks.empty:
                print(f"股票 '{stock_name}' 未找到!")
                print("数据库中可用的股票:")
                print(stock_info.head(10).to_string(index=False))
                return
            else:
                print(f"找到相似股票:")
                for _, stock in matched_stocks.head(3).iterrows():
                    print(f"  {stock['name']} ({stock['stock_code']})")
                
                # 使用第一个匹配的股票
                matched_stocks = matched_stocks.head(1)
        
        stock_code = matched_stocks.iloc[0]['stock_code']
        actual_name = matched_stocks.iloc[0]['name']
        print(f"使用股票: {actual_name} (代码: {stock_code})")
        
        # 从数据库获取历史数据
        print("从数据库获取历史数据...")
        df = get_stock_data(stock_code, days)
        
        if df.empty:
            print("未获取到数据")
            return
        
        print("绘制K线图...")
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制K线图
        plot_candlestick(ax1, df)
        
        # 设置主图表
        ax1.set_title(f'{actual_name} ({stock_code}) K线图', fontsize=16, fontweight='bold')
        ax1.set_ylabel('价格 (CNY)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 设置x轴标签（日期）
        date_labels = []
        date_positions = []
        for i in range(0, len(df), max(1, len(df)//10)):  # 显示约10个日期标签
            date_labels.append(df.iloc[i]['日期'].strftime('%m-%d'))
            date_positions.append(i)
        
        ax1.set_xticks(date_positions)
        ax1.set_xticklabels(date_labels)
        ax1.set_xlim(-1, len(df))
        
        # 绘制成交量
        volume_colors = ['red' if df.iloc[i]['收盘'] >= df.iloc[i]['开盘'] else 'green' 
                        for i in range(len(df))]
        
        ax2.bar(range(len(df)), df['成交量'], color=volume_colors, alpha=0.7)
        ax2.set_ylabel('成交量', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-1, len(df))
        ax2.set_xticks(date_positions)
        ax2.set_xticklabels(date_labels)
        
        # 添加移动平均线
        if len(df) >= 5:
            ma5 = df['收盘'].rolling(window=5).mean()
            ax1.plot(range(len(df)), ma5, color='blue', linewidth=1, label='MA5', alpha=0.8)
        
        if len(df) >= 20:
            ma20 = df['收盘'].rolling(window=20).mean()
            ax1.plot(range(len(df)), ma20, color='orange', linewidth=1, label='MA20', alpha=0.8)
        
        ax1.legend()
        
        plt.tight_layout()
        
        # 保存图片
        filename = f"{actual_name}_matplotlib_k线图.png"
        # plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"K线图已保存为: {filename}")
        
        # 显示图表
        plt.show()
        
        # 显示最新信息
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        change = latest['收盘'] - previous['收盘']
        change_pct = (change / previous['收盘']) * 100 if previous['收盘'] != 0 else 0
        
        print(f"\n--- {actual_name} ({stock_code}) 最新信息 ---")
        print(f"最新价格: {latest['收盘']:.2f} CNY")
        print(f"涨跌额: {change:.2f} CNY ({change_pct:.2f}%)")
        print(f"最高价: {latest['最高']:.2f} CNY")
        print(f"最低价: {latest['最低']:.2f} CNY")
        print(f"成交量: {latest['成交量']:,.0f}")
        print(f"数据范围: {df['日期'].min().strftime('%Y-%m-%d')} 至 {df['日期'].max().strftime('%Y-%m-%d')}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 测试
    plot_stock_kline_matplotlib("卧龙电驱", days=90)
