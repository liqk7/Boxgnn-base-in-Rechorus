import pandas as pd

# 读取 CSV 文件
csv_file_path = './dev.csv'
df = pd.read_csv(csv_file_path)

# 将 DataFrame 写入到 .txt 文件
txt_file_path = './dev.txt'
df.to_csv(txt_file_path, sep='\t', index=False)  # 使用 tab 作为分隔符，去掉行索引