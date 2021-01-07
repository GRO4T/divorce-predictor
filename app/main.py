import id3
import data_set as ds
from pprint import pprint
import pdb

NUM_RUNS = 1
DIVORCE_DATA_SET = "data/students/student-mat.csv"
TEST_SIZE = 0.2

ds.load_dataset(path=DIVORCE_DATA_SET, test_size=TEST_SIZE, a_class="Walc")
sum_acc = 0
min_acc = 1.0
max_acc =  0
print("test set: " + DIVORCE_DATA_SET)
for i in range(NUM_RUNS):
    ds.resplit_dataset(test_size=TEST_SIZE)
    id3_tree = id3.buildTree(ds.train_set)
    print(f"run {i}")
    pprint(id3_tree)
    acc = id3.test(id3_tree, ds.test_set)
    if (acc > max_acc):
        max_acc = acc
    if (acc < min_acc):
        min_acc = acc
    sum_acc += acc
    print(f"accuracy: {acc}")
mean_acc = sum_acc / NUM_RUNS
print(f"\nsummary after {NUM_RUNS} iterations")
print(f"mean accuracy: {mean_acc}")
print(f"max accuracy: {max_acc}")
print(f"min accuracy: {min_acc}")


