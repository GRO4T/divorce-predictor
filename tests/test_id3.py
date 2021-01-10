from app import data_set as ds
from app import id3
from pprint import pprint

DATA_SETS = {
    "data/divorce.csv": {"class": "Class"},
    "data/students/student-mat.csv": {"class": "Dalc"},
    "data/students/student-por.csv": {"class": "Dalc"}
}
TEST_SIZE = 0.2
TEST_RUNS = 5


def test_id3():
    for path in DATA_SETS.keys():
        print(f"data set: {path}")
        class_name = DATA_SETS[path]["class"]
        ds.load_dataset(path=path, test_size=TEST_SIZE, class_name=class_name)
        class_range = len(id3.get_att_values(ds.data_set)[class_name])
        sum_acc = 0
        min_acc = 1.0
        max_acc = 0
        sum_rel_acc = 0
        for i in range(TEST_RUNS):
            ds.resplit_dataset(test_size=TEST_SIZE)
            id3_tree = id3.build_tree(train_set=ds.train_set, original_data_set=ds.data_set)
            print(f"run {i}")
            # print(id3_tree)
            acc, mse, me = id3.test(id3_tree, ds.test_set)
            if acc > max_acc:
                max_acc = acc
            if acc < min_acc:
                min_acc = acc
            sum_acc += acc
            rel_acc = 1 - me / class_range
            sum_rel_acc += rel_acc
            print(f"correctly classified: {acc}")
            print(f"mean squared error: {mse}")
            print(f"absolute mean error: {me}")
            print(f"relative acc: {rel_acc}")
        mean_acc = sum_acc / TEST_RUNS
        mean_rel_acc = sum_rel_acc / TEST_RUNS
        DATA_SETS[path]["mean_acc"] = mean_acc
        DATA_SETS[path]["min_acc"] = min_acc
        DATA_SETS[path]["max_acc"] = max_acc
        DATA_SETS[path]["mean_rel_acc"] = mean_rel_acc

    print(f"\ntotal summary (iterations={TEST_RUNS})\n")
    for path in DATA_SETS.keys():
        class_name = DATA_SETS[path]["class"]
        print(f"data set: {path} (class to predict: {class_name})")
        mean_acc = DATA_SETS[path]["mean_acc"]
        min_acc = DATA_SETS[path]["min_acc"]
        max_acc = DATA_SETS[path]["max_acc"]
        mean_rel_acc = DATA_SETS[path]["mean_rel_acc"]
        print(f"mean correctly classified: {mean_acc}")
        print(f"max correctly classified: {max_acc}")
        print(f"min correctly classified: {min_acc}")
        print(f"mean relative accuracy: {mean_rel_acc}")


test_id3()
