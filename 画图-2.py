import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df_params = pd.read_excel('VGG参数量.xlsx') 
df_flops = pd.read_excel('VGG计算量.xlsx')   
df_acc = pd.read_excel('VGG准确率.xlsx')     

df_params = df_params.dropna(subset=['修改次数/x', '参数量/y']).reset_index(drop=True)
df_flops = df_flops.dropna(subset=['修改次数/x', '计算量/y']).reset_index(drop=True)
df_acc = df_acc.dropna(subset=['修改次数/x', '准确率/y']).reset_index(drop=True)

#获取各数据的最大索引
max_idx_params = len(df_params) - 1
max_idx_flops = len(df_flops) - 1
max_idx_acc = len(df_acc) - 1
max_idx_overall = max(max_idx_params, max_idx_flops, max_idx_acc)

#创建图表
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
fig.suptitle('VGG相关指标变化', fontsize=16, fontweight='bold', y=0.95)

#定义颜色
colors = {
    'params': '#2E86AB',    # 深蓝色（参数量）
    'flops': '#A23B72',     # 深粉色（计算量）
    'acc': '#F18F01'        # 橙色（准确率）
}

#参数量变化图
ax1.plot(df_params['修改次数/x'], df_params['参数量/y'], 
         color=colors['params'], linewidth=3, marker='o', markersize=6, 
         markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors['params'])
ax1.set_ylabel('参数量 (M)', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(-0.5, max_idx_overall + 0.5)
ax1.set_xticks(range(max_idx_overall + 1))

#关键节点标签
key_points_params = [0, 3, 6, 8]  
for i in key_points_params:
    if i <= max_idx_params:
        ax1.annotate(f'{df_params.iloc[i]["参数量/y"]:.4f}', 
                    xy=(df_params.iloc[i]["修改次数/x"], df_params.iloc[i]["参数量/y"]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=colors['params']))

#计算量变化图
ax2.plot(df_flops['修改次数/x'], df_flops['计算量/y'], 
         color=colors['flops'], linewidth=3, marker='s', markersize=6, 
         markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors['flops'])
ax2.set_ylabel('计算量 (G)', fontsize=12)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(-0.5, max_idx_overall + 0.5)
ax2.set_xticks(range(max_idx_overall + 1))

#关键节点标签
key_points_flops = [0, 3, 6, 8]
for i in key_points_flops:
    if i <= max_idx_flops:
        ax2.annotate(f'{df_flops.iloc[i]["计算量/y"]:.6f}', 
                    xy=(df_flops.iloc[i]["修改次数/x"], df_flops.iloc[i]["计算量/y"]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=colors['flops']))

#准确率变化图
#动态调整y轴范围，确保数据显示完整
y_acc_percent = df_acc['准确率/y'] * 100
y_acc_min = y_acc_percent.min() - 1
y_acc_max = y_acc_percent.max() + 1

ax3.plot(df_acc['修改次数/x'], y_acc_percent, 
         color=colors['acc'], linewidth=3, marker='^', markersize=6, 
         markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors['acc'])
ax3.set_xlabel('修改次数', fontsize=12)
ax3.set_ylabel('准确率 (%)', fontsize=12)
ax3.set_ylim(y_acc_min, y_acc_max)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xlim(-0.5, max_idx_overall + 0.5)
ax3.set_xticks(range(max_idx_overall + 1))

#所有节点添加标签
for i in range(len(df_acc)):
    ax3.annotate(f'{df_acc.iloc[i]["准确率/y"]*100:.2f}%', 
                xy=(df_acc.iloc[i]["修改次数/x"], df_acc.iloc[i]["准确率/y"]*100),
                xytext=(0, 8), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=colors['acc']))

#调整布局并保存
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('VGG相关指标变化.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()