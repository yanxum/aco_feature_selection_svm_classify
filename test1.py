import numpy as np
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
print ('data_D.shape={}'.format(data_D.shape))
data_L = new_datawithlabel_array[:,-1]

features_num=200
status_num=2

class Feature_select_from_index(object):
    def __init__(self,features_index_set,data_D,data_L):
        self.features_index_set=features_index_set
        self.data_D=data_D
        self.data_L=data_L

    def feature_select(self):
        data_after_feature_selected=[]
        for i in self.features_index_set:
            data_after_feature_selected.append(self.data_D[:,i])
        data_after_feature_selected=np.array(data_after_feature_selected)
        data_after_feature_selected=data_after_feature_selected.transpose()
        return data_after_feature_selected


    def split_selected_features(self):
        data_after_feature_selected=self.feature_select()
        return train_test_split(data_after_feature_selected, self.data_L, test_size=0.3, random_state=34)


if __name__=='__main__':
    features_index_set = [2, 7, 18, 35, 36, 48, 54, 57, 67, 68, 69, 74, 78, 80, 96, 99, 103, 107, 110, 117, 130, 131,
                          133, 135, 141, 151, 159, 163, 166, 171, 174, 177, 181, 184, 193, 194, 195, 198]
    feature_select_object=Feature_select_from_index(features_index_set,data_D,data_L)
    data_after_feature_selected=feature_select_object.feature_select()
    # print ('type(data_after_feature_selected):{}'.format(type(data_after_feature_selected)))
    # print ('data_after_feature_selected.shape={}'.format(data_after_feature_selected.shape))
    train, test, train_result, test_result=feature_select_object.split_selected_features()

    # if train:
    # print ('测试集和训练集划分成功')
    print (train.shape,test.shape)
