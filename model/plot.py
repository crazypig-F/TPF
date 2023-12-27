import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'


def plot_cm(name, y_test, y_p):
    cm = confusion_matrix(y_test, y_p)
    cm = pd.DataFrame(cm, index=["RC", "YF", "EF", "CC"], columns=["RC", "YF", "EF", "CC"])
    index_list = []
    column_list = []
    value_list = []
    for idx in cm.index:
        for col in cm.columns:
            index_list.append(idx)
            column_list.append(col)
            value_list.append(cm.loc[idx][col])
    pd.DataFrame({"True": index_list, "Predict": column_list, "Value": value_list}).to_csv(
        f"./data/temp/ml/cm/{name}.csv")
    seaborn.heatmap(cm, cmap='viridis', annot=True)
    plt.tick_params(axis='both', labelsize=10)
    plt.xlabel("Predict Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    # plt.show()
    plt.savefig(f'./pdf/ml/cm/{name}.pdf', dpi=600)
    plt.close()


def plot_feature(name, model, feat_labels):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    index = []
    values = []
    for i in range(len(feat_labels)):
        index.append(feat_labels[indices[i]])
        values.append(importance[indices[i]])
        print("%2d) %-*s %f" % (i + 1, 30, feat_labels[indices[i]], importance[indices[i]]))
    pd.DataFrame({"Value": values, "Name": index}, index=index).to_csv(f"./data/temp/ml/feature/{name}.csv")
    plt.figure(figsize=(12, 5))
    seaborn.barplot(x=values, y=index, errorbar=None, color='#3A923A', orient='h')
    plt.tick_params(axis='both', labelsize=10)
    plt.xlabel("Feature importance", fontsize=12)
    # 获取当前坐标轴对象
    # ax = plt.gca()
    # 设置 y 轴刻度标签的旋转角度为45度
    # ax.set_yticks(ax.get_yticks())
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=30, ha='right')
    # plt.show()
    plt.savefig(f'./pdf/ml/feature/classify/{name}.pdf', dpi=600)
    plt.close()
