import id3 as id3
from data_set import DataSet
from pprint import pprint
import pdb

NUM_RUNS = 4
DIVORCE_DATA_SET = "data/students/student-mat.csv"
CLASS = "Dalc"
# DIVORCE_DATA_SET = "data/divorce.csv"
# CLASS = "Class"
TEST_SIZE = 0.2
PRUNE_SIZE = 0.2


def test(test_set, tree):
    acc, mse, me = id3.test(tree, test_set)
    rel_acc = 1 - me / class_range
    print(f"correctly classified: {acc}")
    print(f"mean squared error: {mse}")
    print(f"absolute mean error: {me}")
    print(f"relative acc: {rel_acc}")


if __name__ == "__main__":
    data = DataSet(DIVORCE_DATA_SET, CLASS)
    class_range = len(id3.get_att_values(data.data_set)[CLASS])
    print("data set: " + DIVORCE_DATA_SET)
    for i in range(NUM_RUNS):
        data.resplit_dataset(test_size=TEST_SIZE, prune_size=PRUNE_SIZE)
        id3_tree = id3.build_id3(data.train_set, data.data_set)
        # pprint(id3_tree)
        c45_tree = id3.build_c45(data.prune_set, data.data_set, id3_tree, data.train_set)
        print(f"\nrun {i}")
        print("ID3")
        test(data.test_set, id3_tree)
        print("C45")
        test(data.test_set, c45_tree)
