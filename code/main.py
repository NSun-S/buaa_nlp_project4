from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences

DATA_PATH = '../dataset/'

if __name__ == '__main__':
    test_name = ['郭靖', '杨过', '段誉', '令狐冲', '张无忌']
    model = Word2Vec(sentences=PathLineSentences(DATA_PATH), hs=1, min_count=10, window=5,
                     vector_size=200, sg=0, workers=16, epochs=200)
    for name in test_name:
        print(name)
        for result in model.wv.similar_by_word(name, topn=10):
            print(result[0], '{:.6f}'.format(result[1]))
        print('----------------------')
    model.save('model3.model')
    # with open('../inf.txt', 'r') as inf:
    #     txt_list = inf.readline().split(',')
    #     for idx, name in enumerate(txt_list):
    #         file_name = 'train_' + name + '.txt'
    #         model = Word2Vec(sentences=LineSentence(DATA_PATH + file_name), hs=1, min_count=10, window=5,
    #                          vector_size=200, sg=0, epochs=200)
    #         print(file_name, test_name[idx])
    #         for result in model.wv.similar_by_word(test_name[idx], topn=10):
    #             print(result[0], '{:.6f}'.format(result[1]))


