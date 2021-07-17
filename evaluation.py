def mean_reciprocal_rank(sort_data):
        sort_lable = [s_d[1] for s_d in sort_data]
        assert 1 in sort_lable
        return 1.0 / (1 + sort_lable.index(1))

def precision_at_position_1(sort_data):
    if sort_data[0][1] == 1:
        return 1
    else:
        return 0

def recall_at_position_k_in_10(sort_data, k):
    sort_lable = [s_d[1] for s_d in sort_data]
    select_lable = sort_lable[:k]
    return 1.0 * select_lable.count(1) / sort_lable.count(1)

def evaluate_list(result, y_true, negtive_sample=20):
    sum_hit_1 = 0
    sum_hit_3 = 0
    sum_mrr = 0
    data = []
    total_num = 0
    i = 0
    for a_data, a_true in zip(result, y_true):
        if i % negtive_sample == 0:
            data = []
        data.append((float(a_data), int(a_true)))
        if i % negtive_sample == negtive_sample - 1:
            total_num += 1
            sort_data = sorted(data, key=lambda x: x[0], reverse=True)
            hit_1 = recall_at_position_k_in_10(sort_data, 1)
            hit_3 = recall_at_position_k_in_10(sort_data, 3)
            mrr  = mean_reciprocal_rank(sort_data)
            sum_hit_1 += hit_1
            sum_hit_3 += hit_3
            sum_mrr += mrr
        i += 1

    return (1.0 * sum_hit_1 / total_num, 1.0 * sum_hit_3 / total_num, 1.0 * sum_mrr / total_num)