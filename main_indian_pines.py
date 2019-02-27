from feature_select.aco_feature_selection import ACO,Graph

features_num=200
status_num=2
def main():
    aco = ACO(100, 10, 1.0, 10.0, 0.5, 10, 2)
    graph = Graph(features_num, status_num)
    best_solution,best_classification_accuracy = aco.solve(graph)
    print ('best_solution:{},best_classification_accuracy:{}'.format(best_solution,best_classification_accuracy))


if __name__=='__main__':
    main()