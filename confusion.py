import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':

    # 定义混淆矩阵
    # confusion_matrix = np.array([[233, 3, 3],
    #                              [1, 143, 26],
    #                              [1, 15, 346]])
    confusion_matrix = np.array([[211, 6, 2],
                                 [7, 102, 9],
                                 [1, 12, 261]])

    # 创建一个热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Reds", cbar=False,
                annot_kws={"size": 30},  # 设置热力图上数字的大小
                xticklabels=['Non-Rumor', 'Rumor', 'Unverified Rumor'],
                yticklabels=['Non-Rumor', 'Rumor', 'Unverified Rumor'])

    # 添加标题和标签
    #plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label',fontsize=24)
    plt.ylabel('True Label',fontsize=24)

    # 设置xticklabels和yticklabels的大小
    plt.xticks(fontsize=20)  # 设置x轴标签大小
    plt.yticks(fontsize=20)  # 设置y轴标签大小
    #plt.savefig('weibo_confusion.png', dpi=600)
    plt.savefig('twitter_confusion.png', dpi=600)


    # 显示图像
    plt.show()
