import id3
import data_set as ds
from pprint import pprint
import pdb

NUM_RUNS = 5
DIVORCE_DATA_SET = "data/students/student-mat.csv"
CLASS = "Dalc"
#DIVORCE_DATA_SET = "data/divorce.csv"
#CLASS = "Class"
TEST_SIZE = 0.2

ds.load_dataset(path=DIVORCE_DATA_SET, test_size=TEST_SIZE, a_class=CLASS)

class_range = len(id3.buildAttValues(ds.df)[CLASS])
sum_acc = 0
min_acc = 1.0
max_acc =  0
sum_rel_acc = 0
print("data set: " + DIVORCE_DATA_SET)
for i in range(NUM_RUNS):
    ds.resplit_dataset(test_size=TEST_SIZE)
    id3_tree = id3.buildTreeWrapper(train_set=ds.train_set, original_data_set=ds.df)
    print(f"run {i}")
    # pprint(id3_tree)
    acc, mse, me = id3.test(id3_tree, ds.test_set)
    if (acc > max_acc):
        max_acc = acc
    if (acc < min_acc):
        min_acc = acc
    sum_acc += acc
    rel_acc = 1 - me / class_range
    sum_rel_acc += rel_acc
    print(f"correctly classified: {acc}")
    print(f"mean squared error: {mse}")
    print(f"absolute mean error: {me}")
    print(f"relative acc: {rel_acc}")

mean_acc = sum_acc / NUM_RUNS
mean_rel_acc = sum_rel_acc / NUM_RUNS
print(f"\nsummary after {NUM_RUNS} iterations")
print(f"mean correctly classified: {mean_acc}")
print(f"max correctly classified: {max_acc}")
print(f"min correctly classified: {min_acc}")
print(f"mean relative accuracy: {mean_rel_acc}")


