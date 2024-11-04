import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime

def compute1(l):
    sum = 0
    for i in l:
        sum+=i
    return sum

def s2l(s):
    return [int(char) for char in s]


if __name__ == '__main__':

    with open('processed/01original_translation.json', 'r', encoding='utf-8') as f:
        jsdata = json.load(f)

    fvecs = []
    fstrs = []
    zh_sens =[]
    en_sens = []
    
    for d in jsdata:
        tmp = s2l(d['fvec'])
        if compute1(tmp) > 0:
            fstrs.append(d['fvec'])
            fvecs.append(tmp)
            zh_sens.append(d['zh'])
            en_sens.append(d['en'])

    # 聚类数量
    n_clusters = 8

    # 创建KMeans模型
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    # 进行聚类
    kmeans.fit(fvecs)

    # 获取dict聚类的结果标签
    labels = kmeans.labels_

    dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7:[]}

    for i in range(len(labels)):
        dict[labels[i]].append({'en': en_sens[i], 'fvec': fstrs[i], 'zh': zh_sens[i]})

    # 当前时间
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    with open(f'processed/02clustered_original_translation.json', 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)

    # 输出结果
    print("聚类结果标签:", labels)
    print("聚类中心:\n", kmeans.cluster_centers_)

    # 可视化聚类结果（如果适用）
    for i in range(n_clusters):
        plt.plot(kmeans.cluster_centers_[i], label=f'Cluster {i}')
    plt.legend()
    plt.show()


    # # 计算不同聚类数下的SSE
    # sse = []
    # k_range = range(1, 15)
    # for k in k_range:
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(fvecs)
    #     sse.append(kmeans.inertia_)  # inertia_ 是KMeans中的SSE指标

    # # 绘制肘部图
    # plt.plot(k_range, sse, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('SSE')
    # plt.title('Elbow Method for Optimal k')
    # plt.show()