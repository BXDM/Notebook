import requests
import json
import pandas as pd
import time
from datetime import datetime

class IFindHTTPFetcher:
    def __init__(self, access_token):
        self.access_token = access_token
        self.headers = {
            "Content-Type": "application/json",
            "access_token": self.access_token
        }
        self.base_url = 'https://quantapi.51ifind.com/api/v1'

    def get_all_a_stocks(self):
        """获取所有沪深A股代码列表"""
        thsUrl = f'{self.base_url}/data_pool'
        thsPara = {
            "reportname": "p03425",  # 股票基本信息报表
            "functionpara": {
                "date": datetime.now().strftime("%Y%m%d"),
                "blockname": "001005010",  # 沪深A股板块
                "iv_type": "allcontract"
            },
            "outputpara": "p03291_f001,p03291_f002"  # 股票代码和名称
        }

        try:
            response = requests.post(url=thsUrl, json=thsPara, headers=self.headers, timeout=30)
            data = json.loads(response.content)

            if data.get('code') != 0:
                print(f"获取股票列表失败: {data.get('message')}")
                return pd.DataFrame()

            # 解析返回的数据
            stocks_data = data.get('tables', [{}])[0].get('data', [])
            if not stocks_data:
                return pd.DataFrame()

            # 转换为DataFrame
            stocks_df = pd.DataFrame(stocks_data, columns=['thscode', 'securityname'])
            print(f"共获取到 {len(stocks_df)} 只A股")
            return stocks_df

        except Exception as e:
            print(f"获取股票列表时发生异常: {e}")
            return pd.DataFrame()

    def get_stock_history_batch(self, stock_codes, start_date, end_date, batch_size=50):
        """
        批量获取多只股票的历史行情数据
        """
        all_data = []

        # 分批处理，避免请求过大
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i + batch_size]
            codes_str = ",".join(batch_codes)

            print(f"正在获取第 {i // batch_size + 1} 批数据，共 {len(batch_codes)} 只股票...")

            thsUrl = f'{self.base_url}/cmd_history_quotation'
            thsPara = {
                "codes": codes_str,
                "indicators": "open,high,low,close,changeRatio,change,volume,turnoverRatio",
                "startdate": start_date,
                "enddate": end_date,
                "functionpara": {
                    "Fill": "Previous",  # 使用前值填充
                    "Adjust": "0"  # 不复权
                }
            }

            try:
                response = requests.post(url=thsUrl, json=thsPara, headers=self.headers, timeout=60)
                data = json.loads(response.content)

                if data.get('code') != 0:
                    print(f"批量获取失败: {data.get('message')}")
                    continue

                # 解析数据
                table_data = data.get('tables', [])
                if table_data:
                    df = pd.DataFrame(table_data[0].get('data', []))
                    all_data.append(df)

                # 避免请求过于频繁
                time.sleep(1)

            except Exception as e:
                print(f"获取批次 {i // batch_size + 1} 时发生异常: {e}")
                continue

        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            return result_df
        else:
            return pd.DataFrame()

    def get_single_stock_history(self, stock_code, start_date, end_date):
        """获取单只股票的历史行情数据"""
        thsUrl = f'{self.base_url}/cmd_history_quotation'
        thsPara = {
            "codes": stock_code,
            "indicators": "open,high,low,close,changeRatio,change,volume,turnoverRatio",
            "startdate": start_date,
            "enddate": end_date,
            "functionpara": {
                "Fill": "Previous",
                "Adjust": "0"
            }
        }

        try:
            response = requests.post(url=thsUrl, json=thsPara, headers=self.headers, timeout=30)
            data = json.loads(response.content)

            if data.get('code') != 0:
                print(f"获取 {stock_code} 数据失败: {data.get('message')}")
                return pd.DataFrame()

            # 解析数据
            table_data = data.get('tables', [])
            if table_data:
                df = pd.DataFrame(table_data[0].get('data', []))
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"获取 {stock_code} 数据时发生异常: {e}")
            return pd.DataFrame()


# 使用示例
def main():
    # 您的 access_token
    with open ('token.txt', 'r') as file:
        ACCESS_TOKEN = file.read()

    # 创建数据获取器实例
    fetcher = IFindHTTPFetcher(ACCESS_TOKEN)

    # 1. 获取所有A股代码
    print("正在获取所有A股代码...")
    all_stocks = fetcher.get_all_a_stocks()

    if all_stocks.empty:
        print("未能获取到股票列表，请检查网络连接或token有效性")
        return

    # 2. 获取历史行情数据（这里以获取前10只股票为例）
    sample_stocks = all_stocks['thscode'].head(10).tolist()
    print(f"\n开始获取 {len(sample_stocks)} 只股票的历史数据...")

    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # 批量获取数据
    history_data = fetcher.get_stock_history_batch(
        sample_stocks,
        start_date,
        end_date,
        batch_size=5  # 每批5只股票
    )

    if not history_data.empty:
        print(f"\n数据获取完成！共获取 {len(history_data)} 条记录")
        print("\n数据预览:")
        print(history_data.head())

        # 保存到CSV
        output_file = f'stock_data_{start_date}_to_{end_date}.csv'
        history_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存到: {output_file}")
    else:
        print("未能获取到历史数据")


if __name__ == '__main__':
    main()
