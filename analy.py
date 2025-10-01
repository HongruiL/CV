import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('validation_analysis.csv')

# 基本统计
print("="*60)
print("错误分析")
print("="*60)
print(f"平均绝对误差: {df['abs_error'].mean():.2f}")
print(f"中位数误差: {df['abs_error'].median():.2f}")
print(f"最大误差: {df['abs_error'].max():.2f}")

# 按卡路里范围统计
print("\n按卡路里范围分组:")
range_stats = df.groupby('calorie_range')['abs_error'].agg(['count', 'mean', 'median'])
print(range_stats)

# 查看最差的10个
print("\n预测最差的10个样本:")
worst_10 = df.nlargest(10, 'abs_error')[['dish_id', 'true_calories', 'pred_calories', 'abs_error']]
print(worst_10)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 真实值 vs 预测值散点图
axes[0, 0].scatter(df['true_calories'], df['pred_calories'], alpha=0.5)
axes[0, 0].plot([0, df['true_calories'].max()], [0, df['true_calories'].max()], 'r--', label='Perfect')
axes[0, 0].set_xlabel('True Calories')
axes[0, 0].set_ylabel('Predicted Calories')
axes[0, 0].set_title('Predictions vs Ground Truth')
axes[0, 0].legend()

# 2. 误差分布
axes[0, 1].hist(df['abs_error'], bins=50, edgecolor='black')
axes[0, 1].set_xlabel('Absolute Error (calories)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Error Distribution')

# 3. 误差 vs 真实值
axes[1, 0].scatter(df['true_calories'], df['abs_error'], alpha=0.5)
axes[1, 0].set_xlabel('True Calories')
axes[1, 0].set_ylabel('Absolute Error')
axes[1, 0].set_title('Error vs True Calories')

# 4. 按范围的箱线图
df.boxplot(column='abs_error', by='calorie_range', ax=axes[1, 1])
axes[1, 1].set_xlabel('Calorie Range')
axes[1, 1].set_ylabel('Absolute Error')
axes[1, 1].set_title('Error by Calorie Range')

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150)
print("\n可视化已保存到 error_analysis.png")
plt.show()