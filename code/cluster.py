from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('../name.txt', 'r', encoding='utf8') as f:
        names = f.readline().split(' ')
    model = Word2Vec.load('model3.model')
    names = [name for name in names if name in model.wv]
    name_vectors = [model.wv[name] for name in names]
    tsne = TSNE()
    embedding = tsne.fit_transform(name_vectors)
    n = 5
    label = KMeans(n).fit(embedding).labels_
    plt.title('kmeans聚类结果')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(label)):
        if label[i] == 0:
            plt.plot(embedding[i][0], embedding[i][1], 'ro', )
        if label[i] == 1:
            plt.plot(embedding[i][0], embedding[i][1], 'go', )
        if label[i] == 2:
            plt.plot(embedding[i][0], embedding[i][1], 'yo', )
        if label[i] == 3:
            plt.plot(embedding[i][0], embedding[i][1], 'co', )
        if label[i] == 4:
            plt.plot(embedding[i][0], embedding[i][1], 'bo', )
        if label[i] == 5:
            plt.plot(embedding[i][0], embedding[i][1], 'mo', )
        plt.annotate(names[i], xy=(embedding[i][0], embedding[i][1]), xytext=(embedding[i][0]+0.1, embedding[i][1]+0.1))
    plt.savefig('cluster3.png')

