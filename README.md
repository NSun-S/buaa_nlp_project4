## 深度学习与自然语言处理第四次作业

<p align='right'> <strong>SY2106318 孙旭东 </strong></p>

**详细技术报告见**[SY2106318-深度学习和自然语言处理第四次作业](https://github.com/NSun-S/buaa_nlp_project4/raw/main/SY2106318-深度学习和自然语言处理第四次作业.pdf)

### 程序简介

#### 程序运行

代码文件共三个:

- dataprepare.py：数据准备，将小说分词处理，并保存
- main.py：Word2Vec模型的生成
- cluster.py：使用Word2Vec模型生成的词向量进行聚类

### 3.实验过程

#### 3.1数据准备

使用到的数据为金庸的16本武侠小说，对数据进行预处理的代码如下：

```python
def get_single_corpus(file_path):
    """
    获取file_path文件对应的内容
    :return: file_path文件处理结果
    """
    corpus = ''
    # unuseful items filter
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~「」『』（）]+'
    with open('../stopwords.txt', 'r', encoding='utf8') as f:
        stop_words = [word.strip('\n') for word in f.readlines()]
        f.close()
    # print(stop_words)
    with open(file_path, 'r', encoding='ANSI') as f:
        corpus = f.read()
        corpus = re.sub(r1, '', corpus)
        corpus = corpus.replace('\n', '')
        corpus = corpus.replace('\u3000', '')
        corpus = corpus.replace('本书来自免费小说下载站更多更新免费电子书请关注', '')
        f.close()
    words = list(jieba.cut(corpus))
    return [word for word in words if word not in stop_words]
```

在以utf8编码格式读取文件内容后，删除文章内的所有非中文字符，以及和小说内容无关的片段，得到字符串形式的语料库，然后使用jieba分词进行分词，并使用**百度停用词表**进行停用词的过滤，最终返回小说的分词列表。

因为后续需要分析词语之间的相关性，因此本次实验选取了《射雕英雄传》，《神雕侠侣》，《天龙八部》，《笑傲江湖》，《倚天屠龙记》五本小说，将分词列表按照每行50词保存在txt格式文件中，方便后续使用。

#### 3.2 训练Word2Vec模型

这里直接使用python库gensim中的Word2Vec类进行模型的训练，并选取5本小说中的代表性人物，分析训练后与该人物相关性最强的10个词。

```python
test_name = ['郭靖', '杨过', '段誉', '令狐冲', '张无忌']
model = Word2Vec(sentences=PathLineSentences(DATA_PATH), hs=1, min_count=10, window=5,
                 vector_size=200, sg=0, workers=16, epochs=200)
for name in test_name:
    print(name)
    for result in model.wv.similar_by_word(name, topn=10):
        print(result[0], '{:.6f}'.format(result[1]))
```

模型的参数释义如下：

- sentences：可以是一个list，此处使用的PathLineSentences为一个文件夹下的所有文件，另外有LineSentence对单个文件生效；
- hs：如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用；
- min_count：可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5；
- window：表示当前词与预测词在一个句子中的最大距离是多少；
- vector_size：是指特征向量的维度，默认为100，大的size需要更多的训练数据,但是效果会更好；
- sg：用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法；
- workers：线程数；
- epoches：训练迭代轮数。

#### 3.3 K-means聚类

为了进一步验证模型的有效性，使用TSNE将训练得到的模型中的词向量进行降维（方便展示效果），并使用K-means算法进行聚类。这里聚类用到的词为5本小说中的代表性人物。最终用散点图进行效果展示，部分代码如下：

```python
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
    plt.annotate(names[i], xy=(embedding[i][0], embedding[i][1]), xytext=(embedding[i][0]+0.1, embedding[i][1]+0.1))
plt.savefig('cluster3.png')
```

### 结论

本次作业使用Word2Vec模型对金庸的五本小说进行了词向量的构建和聚类，结果显示与某个词（选取小说主角）相似的词在原著中也有一定的联系，在原著中有联系的一系列词（以人物为例）构建而成的词向量距离较近，使用Kmeans聚类效果良好。可见词向量对后续任务的重要意义。

### 参考文档

[深入浅出Word2Vec原理解析](https://zhuanlan.zhihu.com/p/114538417)

[秒懂词向量Word2vec的本质](https://zhuanlan.zhihu.com/p/26306795)

[word2vec词向量中文语料处理gensim总结](https://blog.csdn.net/shuihupo/article/details/85162237)

[利用Word2Vec模型训练Word Embedding,并进行聚类分析](https://blog.csdn.net/weixin_42663984/article/details/116739799)



