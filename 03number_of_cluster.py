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
    
    for d in jsdata:
        tmp = s2l(d['fvec'])
        if compute1(tmp) > 0:
            fstrs.append(d['fvec'])
            fvecs.append(tmp)

    # 计算不同聚类数下的SSE
    sse = []
    k_range = range(1, 15)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(fvecs)
        sse.append(kmeans.inertia_)  # inertia_ 是KMeans中的SSE指标

    # 绘制肘部图
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k')
    plt.show()