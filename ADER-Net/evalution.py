import numpy as np
import os
import math
from sklearn.metrics  import accuracy_score,f1_score,average_precision_score


def str_2_list(str_tmp):
    start_index = str_tmp.find('[')
    end_index = str_tmp.find(']')
    str = str_tmp[start_index + 1:end_index]
    list_tmp = str.split(', ')
    list_res = [float(i) for i in list_tmp]
    return list_res


file = open('results/test_result_5.txt',mode='r')
lines = file.readlines()
test_cnt = math.floor(len(lines) / 2)

y_ture = []
y_pred = []
for i in range(0, test_cnt):
    pred = str_2_list(lines[2 * i])
    y_pred.append(pred.index(max(pred)))
    target = str_2_list(lines[2 * i + 1])
    y_ture.append(target.index(max(target)))

print('-' * 100)
print('accuracy:                 {}'.format(accuracy_score(y_ture,y_pred)))
print('-' * 100)
print('macro f1:                 {}'.format(f1_score(y_ture,y_pred,average='macro')))

pred_res = [[],[],[],[],[],[]]
target_res = [[],[],[],[],[],[]]

for class_index in range(6):
    for i in range(test_cnt):
        pred = str_2_list(lines[2 * i])
        pred_res[class_index].append(pred[class_index])
        target = str_2_list(lines[2 * i + 1])
        target_res[class_index].append(target[class_index])
mAP = []
for i in range(6):
    mAP.append(average_precision_score(target_res[i],pred_res[i]))
print('-' * 100)
print('mean average precision:   {}'.format(np.mean(mAP)))



