import numpy as np
from feature_select import Fitness
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

input_image=loadmat('C:\\Users\\Huff\\Desktop\\yanxm16\\paper_data\\Indian_pines_corrected.mat')['indian_pines_corrected']
output_image=loadmat('C:\\Users\\Huff\\Desktop\\yanxm16\\paper_data\\Indian_pines_gt.mat')['indian_pines_gt']
new_datawithlabel_list = []
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] != 0:
            c2l = list(input_image[i][j])
            c2l.append(output_image[i][j])
            new_datawithlabel_list.append(c2l)
new_datawithlabel_array = np.array(new_datawithlabel_list)
data_D = preprocessing.StandardScaler().fit_transform(new_datawithlabel_array[:,:-1])
data_L = new_datawithlabel_array[:,-1]

features_num=200
status_num=2

class Graph(object):
    def __init__(self, features_num: int,status_num:int):

        self.features_num=features_num
        self.status_num=status_num
        self.pheromone = [[1 / (features_num/40) for j in range(features_num)] for i in range(status_num)]


class Feature_select_from_index(object):
    def __init__(self,features_index_set,data_D,data_L):
        self.features_index_set=features_index_set
        self.data_D=data_D
        self.data_L=data_L

    def feature_select(self):
        data_after_feature_selected=[]
        for i in self.features_index_set:
            data_after_feature_selected.append(self.data_D[:,i])
        data_after_feature_selected = np.array(data_after_feature_selected)
        data_after_feature_selected = data_after_feature_selected.transpose()
        return data_after_feature_selected


    def split_selected_features(self):
        data_after_feature_selected=self.feature_select()
        return train_test_split(data_after_feature_selected, self.data_L, test_size=0.3, random_state=34)


class ACO(object):
    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int,
                 strategy: int):
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy
        self.ant = _Ant

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]


    def _update_pheromone_init(self, graph: Graph,ant):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    def solve1(self, graph: Graph):
        best_classification_accuracy = 0
        best_solution = []
        gamma = 0.125
        c = 20
        for gen in range(self.generations):
            print('第{}代蚂蚁开始选择特征'.format(gen) + '。' * 80)
            ants = [_Ant(self, graph,p0=0.2) for i in range(self.ant_count)]  # 生成ant_count个_Ant对象
            for ant in ants:
                if gen == 0 and ants.index(ant) == 0:
                    print('第0代第0只蚂蚁进行特征选择初始化......')
                    features_index_set, features_status = ant.init_feature_set_and_setStatus()

                    print('第0代第0只蚂蚁选择的特征：features_index_set={},features_status={}'.format(features_index_set,
                                                                                          features_status))
                    print('第0代第0只蚂蚁选择的特征:len(features_index_set)={},len(features_status)={}'.format(
                        len(features_index_set), len(features_status)))

                    feature_selected_object = Feature_select_from_index(features_index_set, data_D, data_L)
                    train, test, train_result, test_result = feature_selected_object.split_selected_features()
                    # print (train.shape,test.shape)
                    best_classification_accuracy = Fitness.funcFitness(train, train_result, test, test_result, gamma, c)
                    print('第0代第0只蚂蚁选择的特征分类的精确度为：{}'.format(best_classification_accuracy))
                    ant._update_pheromone_delta(features_status)
                    self._update_pheromone_init(graph, ant)

                else:
                    print('第{}代第{}只蚂蚁开始选择'.format(gen, ants.index(ant)))
                    for j in range(graph.features_num):
                        ant._select_next_feature_status1(j)
                    features_index_set = ant.features_index_set
                    features_status = ant.features_status
                    print('len(features_index_set)={},len(features_status)={}'.format(len(features_index_set),
                                                                                      len(features_status)))
                    features = Feature_select_from_index(features_index_set, data_D, data_L)
                    train, test, train_result, test_result = features.split_selected_features()
                    fitness = Fitness.funcFitness(train, train_result, test, test_result, gamma, c)
                    if fitness > best_classification_accuracy:
                        print('本次分类准确率优于上次' + '*' * 30+'acc={}'.format(fitness))
                        best_classification_accuracy = fitness
                        best_solution = [] + ant.features_status
                    # update pheromone
                    ant._update_pheromone_delta(features_status)
            self._update_pheromone(graph, ants)
        return best_solution, best_classification_accuracy


class _Ant(object):
    def __init__(self, aco: ACO, graph: Graph,p0:0.2):
        self.colony = aco
        self.graph = graph
        self.pheromone_delta = []  # the local increase of pheromone
        self.allowed_features_index=[i+1 for i in range(graph.features_num)]

        self.features_index_set=[]
        self.features_status = []
        self.p0=p0


    def init_feature_set_and_setStatus(self):
        init_feature_status = list(np.random.rand(features_num))
        for j in range(self.graph.features_num):
            if init_feature_status[j]<self.p0:
                self.features_index_set.append(j)
                self.features_status.append(1)
            else:
                self.features_status.append(0)
        return self.features_index_set,self.features_status


    def _select_next_feature_status1(self,j):
        # print('选择第{}个波段开始'.format(j))
        init_feature_status = np.random.rand()
        if init_feature_status<=self.p0:
            pheromone_concentration=[]
            for i in range(self.graph.status_num):
                pheromone_concentration.append(self.graph.pheromone[i][j])
            status=pheromone_concentration.index(max(pheromone_concentration))
            self.features_status.append(status)
            if status==1:
                self.features_index_set.append(j)
        else:
            denominator=0
            pheromone_concentration_scaled = []
            for i in range(self.graph.status_num):
                denominator+=self.graph.pheromone[i][j]
            for i in range(self.graph.status_num):
                self.graph.pheromone[i][j]=self.graph.pheromone[i][j]/denominator
                pheromone_concentration_scaled.append(self.graph.pheromone[i][j])
            init_feature_status_again = np.random.rand()
            if init_feature_status_again>0 and init_feature_status_again<min(pheromone_concentration_scaled):
                status=pheromone_concentration_scaled.index(min(pheromone_concentration_scaled))
            else:
                status=pheromone_concentration_scaled.index(max(pheromone_concentration_scaled))
            self.features_status.append(status)
            if status==1:
                self.features_index_set.append(j)
        # print ('选择第{}个波段结束'.format(j))


    def _update_pheromone_delta(self,features_status):
        self.pheromone_delta = [[0 for j in range(self.graph.features_num)] for i in range(self.graph.status_num)]
        for _ in range(self.graph.features_num):
            i = features_status[_]
            j = list(range(self.graph.features_num))[_]
            if self.colony.update_strategy == 1:  # ant-quality system
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:  # ant-density system
                self.pheromone_delta[i][j] = self.colony.Q / self.graph.features_num