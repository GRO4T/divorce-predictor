from app import data_set as ds
from app import id3
from pprint import pprint

datasets = {
    "data/divorce.csv" : {"class": "Class"},
    "data/students/student-mat.csv" : {"class": "Walc"},
    "data/students/student-por.csv" : {"class": "Walc"}
}
test_size = 0.05
test_runs = 5

def test_id3():
    for path in datasets.keys():
        a_class = datasets[path]["class"]
        ds.load_dataset(path=path, test_size=test_size, a_class=a_class)
        sum_acc = 0
        min_acc = 1.0
        max_acc =  0
        print(f"data set: {path}")
        for i in range(test_runs):
            ds.resplit_dataset(test_size=test_size)
            id3_tree = id3.buildTree(ds.train_set)
            print(f"run {i}")
            print(id3_tree)
            acc = id3.test(id3_tree, ds.test_set)
            if (acc > max_acc):
                max_acc = acc
            if (acc < min_acc):
                min_acc = acc
            sum_acc += acc
            print(f"accuracy: {acc}")
        mean_acc = sum_acc / test_runs
        datasets[path]["mean_acc"] = mean_acc
        datasets[path]["min_acc"] = min_acc
        datasets[path]["max_acc"] = max_acc
    
    print(f"\ntotal summary (iterations={test_runs})\n")
    for path in datasets.keys():
        a_class = datasets[path]["class"]
        print(f"data set: {path} (class to predict: {a_class})")
        mean_acc = datasets[path]["mean_acc"]
        min_acc = datasets[path]["min_acc"]
        max_acc = datasets[path]["max_acc"]
        print(f"mean accuracy: {mean_acc}")
        print(f"max accuracy: {max_acc}")
        print(f"min accuracy: {min_acc}\n")


test_id3()
