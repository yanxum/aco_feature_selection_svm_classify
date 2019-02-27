def _update_pheromone_delta(features_status,features_num,status_num):
    pheromone_delta = [[0 for j in range(features_num)] for i in range(status_num)]
    for _ in range(features_num):
        i=features_status[_]
        j=list(range(features_num))[_]
        pheromone_delta[i][j] = 5 / 10
    return pheromone_delta


features_status=[0,0,1,0,1,0,1,1,0,0]
features_num=10
status_num=2

if __name__=='__main__':
    print (_update_pheromone_delta(features_status,features_num,status_num))
