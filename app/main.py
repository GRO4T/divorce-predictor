import id3
import data_set as ds
from pprint import pprint
import pdb

NUM_RUNS = 1
#DIVORCE_DATA_SET = "data/students/student-mat.csv"
#CLASS = "Dalc"
DIVORCE_DATA_SET = "data/divorce.csv"
CLASS = "Class"
TEST_SIZE = 0.2
PRUNE_SIZE = 0.2

ds.load_dataset(path=DIVORCE_DATA_SET, test_size=TEST_SIZE, prune_size=PRUNE_SIZE, class_name=CLASS)
class_range = len(id3.get_att_values(ds.data_set)[CLASS])


def test(test_set, tree):
    acc, mse, me = id3.test(tree, ds.test_set)
    rel_acc = 1 - me / class_range
    print(f"correctly classified: {acc}")
    print(f"mean squared error: {mse}")
    print(f"absolute mean error: {me}")
    print(f"relative acc: {rel_acc}")


if __name__ == "__main__":
    print("data set: " + DIVORCE_DATA_SET)
    for i in range(NUM_RUNS):
        ds.resplit_dataset(test_size=TEST_SIZE, prune_size=PRUNE_SIZE)
        id3_tree = id3.build_id3(ds.train_set, ds.data_set)
        # pprint(id3_tree)
        c45_tree = id3.build_c45(ds.prune_set, ds.data_set, id3_tree, ds.train_set)
        print(f"run {i}")
        print("ID3")
        test(ds.test_set, id3_tree)
        print("C45")
        test(ds.test_set, c45_tree)
