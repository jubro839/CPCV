from itertools import combinations
class CPCVPath:
    def __init__(self, n_groups, n_test_groups):
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        # calculate test_groups combinations
        self.combination = list(combinations([x for x in range(n_groups)], n_test_groups))
        self._set_path_indexes()
    def _set_path_indexes(self):
        cnt_path = [0 for _ in range(self.n_groups)]
        self.pairs = {}
        for group in self.combination:
            temp_path = [0 for _ in range(len(group))]
            for idx, group_index in enumerate(group):
                temp_path[idx] = cnt_path[group_index]
                cnt_path[group_index] += 1
            self.pairs[group] = tuple(temp_path)
    def get_path(self, test_group_indexes):
        return self.pairs[test_group_indexes]
# if __name__ == '__main__':
#     n_groups = 6
#     n_test_groups = 3
#     paths = CPCVPath(n_groups, n_test_groups)
#     for group in paths.pairs:
#         print('test_groups: {} / path_indexes: {}