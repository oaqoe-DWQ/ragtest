import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)

import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel('shuju/回顾_20260421_194226.xlsx')
print('=== 数据概览 ===')
print(f'行数: {len(df)}, 列数: {len(df.columns)}')
print()
print('=== 列名 ===')
print(df.columns.tolist())
print()
print('=== 前20行数据 ===')
print(df.head(20).to_string())
print()
print('=== 数据类型 ===')
print(df.dtypes)
print()
print('=== 数值列统计 ===')
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if numeric_cols:
    print(df[numeric_cols].describe())
