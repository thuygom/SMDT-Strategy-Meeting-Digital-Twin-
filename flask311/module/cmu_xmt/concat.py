import matplotlib.pyplot as plt

# 공통 모델 이름
model_names = ["Concat+MLP", "EmbraceNet", "Pairwise CMT", "MRO"]

# CMU-MOSEI 데이터: (모델명, Acc-7, Acc-2, FLOPs, 색상, 마커)
cmu_data = [
    (model_names[0], 41.20, 77.85, 22.31, 'gray', 'o'),
    (model_names[1], 44.30, 79.10, 26.78, 'gray', 's'),
    (model_names[2], 50.90, 84.20, 254.94, 'blue', '^'),
    (model_names[3], 55.30, 86.45, 173.48, 'orange', 'D'),
]

# MELD 데이터: (모델명, Acc-7, Acc-2, FLOPs, 색상, 마커)
meld_data = [
    (model_names[0], 61.83, 85.67, 23.90, 'gray', 'o'),
    (model_names[1], 63.92, 87.04, 29.20, 'gray', 's'),
    (model_names[2], 66.14, 89.36, 378.82, 'blue', '^'),
    (model_names[3], 67.07, 90.32, 297.36, 'orange', 'D'),
]

# 스타일 설정
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'font.family': 'serif'
})

# ⭕● CMU-MOSEI 그래프
fig1, ax1 = plt.subplots(figsize=(7, 5))
for label, acc7, acc2, flops, color, marker in cmu_data:
    ax1.scatter(flops, acc7, color=color, marker=marker, s=100,
                edgecolors='black', label=f"{label} (Acc-7)")
    ax1.scatter(flops, acc2, facecolors='none', edgecolors=color, marker=marker, s=100,
                label=f"{label} (Acc-2)")
ax1.set_xlabel('MFLOPs')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('CMU-MOSEI: Acc-7 (●) vs Acc-2 (○)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
fig1.tight_layout()

# ⭕● MELD 그래프
fig2, ax2 = plt.subplots(figsize=(7, 5))
for label, acc7, acc2, flops, color, marker in meld_data:
    ax2.scatter(flops, acc7, color=color, marker=marker, s=100,
                edgecolors='black', label=f"{label} (Acc-7)")
    ax2.scatter(flops, acc2, facecolors='none', edgecolors=color, marker=marker, s=100,
                label=f"{label} (Acc-2)")
ax2.set_xlabel('MFLOPs')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('MELD: Acc-7 (●) vs Acc-2 (○)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
fig2.tight_layout()

plt.show()
