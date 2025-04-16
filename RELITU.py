import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
# 示例数据
models = ['AUGADAPTCSODE', 'AUGCSODE', 'TRANSFORMER']
mae = [0.26542762,
0.3674573,
0.48452348
]
mse = [0.15583855,
0.25672415,
0.42984366
]
rmse = [0.3947639156978706,
0.5066795328298981,
0.6556246366399844
]
r2 = [0.001509,
0.002607,
0.004189
]
metrics = [mae,mse,rmse, r2]
metric_labels = ['MAE','MSE','RMSE', '$1-R^2$']

# 数据整理为 DataFrame 格式
heatmap_data = pd.DataFrame(np.array(metrics).T, columns=metric_labels, index=models)

# 设置 Seaborn 风格
sns.set_theme(style="whitegrid", context="talk")

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".6f", cmap="coolwarm", cbar=True, linewidths=0.5)

# 添加标题和标签
plt.title("Heatmap of Model Metrics", fontsize=18, pad=15)
plt.ylabel("Models", fontsize=14)
plt.xlabel("Metrics", fontsize=14)

# 展示图像
plt.tight_layout()
plt.show()

