import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 设置中文字体和全局样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

# 读取数据
df = pd.read_excel("VGG准确率.xlsx")

x = df['修改次数/x']
y = df['准确率/y']

# 创建图形和轴
fig, ax = plt.subplots(facecolor='white')

# 1. 绘制散点图（数据点）
scatter = ax.scatter(x, y, color='#2E86AB', marker='o', edgecolors='white', 
                    linewidth=1.5, alpha=0.8, label='实际数据')
ax.plot(x, y, color='#6A8A82', linestyle=':', alpha=0.7, linewidth=1.5)

# 2. 绘制趋势线
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
trend_line = slope * x + intercept
ax.plot(x, trend_line, color='#A23B72', linestyle='--', alpha=0.8, 
        label=f'趋势线 (R²={r_value**2:.3f})')

# 3. 标注关键数据点（最高和最低准确率）
max_idx = y.idxmax()
min_idx = y.idxmin()


# 4. 设置坐标轴和标题
ax.set_xlabel('修改次数', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('准确率，单位/%', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('准确率变化图', fontsize=16, fontweight='bold', pad=20)

# 5. 设置坐标轴范围和刻度
ax.set_xlim(-1,x.max() + 1)
#ax.set_ylim(0.46, 0.60)
ax.set_xticks(range(0, int(x.max()) + 1, 1))
#ax.set_yticks(np.arange(0.46, 0.61, 0.02))

# 6. 添加网格
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
ax.set_axisbelow(True)

# 7. 添加图例
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.9)

# 调整布局并保存
plt.tight_layout()
plt.show()

print(f"\n关键统计信息:")
print(f"1. 平均准确率: {y.mean():.4f}")
print(f"2. 最高准确率: {y.max():.4f} (修改次数: {x[y.idxmax()]})")
print(f"3. 最低准确率: {y.min():.4f} (修改次数: {x[y.idxmin()]})")
print(f"4. 相关性系数 R: {r_value:.4f}")
print(f"5. 决定系数 R²: {r_value**2:.4f}")
print(f"6. 趋势线方程: y = {slope:.6f}x + {intercept:.4f}")