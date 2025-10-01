# check_depth_files.py
from pathlib import Path
from PIL import Image
import pandas as pd

DATA_ROOT = Path('Nutrition5K/Nutrition5K')
CSV_FILE = DATA_ROOT / 'nutrition5k_train.csv'

df = pd.read_csv(CSV_FILE)
corrupted_files = []

print("检查深度图文件...")
for idx, row in df.iterrows():
    dish_id = row['ID']
    depth_path = DATA_ROOT / 'train' / 'depth_raw' / dish_id / 'depth_raw.png'
    
    if not depth_path.exists():
        corrupted_files.append(dish_id)
        print(f"文件不存在: {dish_id}")
        continue
    
    try:
        img = Image.open(depth_path)
        img.load()  # 强制加载确认文件可读
    except Exception as e:
        corrupted_files.append(dish_id)
        print(f"损坏文件: {dish_id} - {e}")

print(f"\n总共检查: {len(df)} 个文件")
print(f"损坏文件: {len(corrupted_files)} 个")

if corrupted_files:
    # 保存损坏文件列表
    with open('corrupted_files.txt', 'w') as f:
        for dish_id in corrupted_files:
            f.write(f"{dish_id}\n")
    print("损坏文件列表已保存到 corrupted_files.txt")
    
    # 创建清理后的CSV（排除损坏样本）
    clean_df = df[~df['ID'].isin(corrupted_files)]
    clean_df.to_csv(DATA_ROOT / 'nutrition5k_train_clean.csv', index=False)
    print(f"清理后的CSV已保存，包含 {len(clean_df)} 个样本")