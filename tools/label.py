import os
import xlrd
import math
import numpy as np

execl_dir = '../annotation/annotation.xls'
N = 16
def frame_location(video_length,num_tmp):
    num_location = []
    start_num = num_tmp
    end_num = video_length-1
    span = end_num-start_num
    step_lower = math.floor(span/(N-1))
    step_upper = math.ceil(span/(N-1))
    num_location.append(start_num)
    for i in range(int((N-2)/2)):
        frame_tmp = start_num+int((i+2)/2)*step_lower+int((i+1)/2)*step_upper
        num_location.append(frame_tmp)
    for i in range(int((N-2)/2)):
        frame_tmp = end_num-int((i+2)/2)*step_lower-int((i+1)/2)*step_upper
        num_location.append(frame_tmp)
    num_location.append(end_num)
    num_location.sort()

    return  num_location

########create txt that is used for ADER training and test #############
book = xlrd.open_workbook(execl_dir)
sheet = book.sheets()[0]
nrows = sheet.nrows

folder_list = os.listdir('../dataset/training_set_img')
for folder in folder_list:
    image_list = os.listdir('../dataset/training_set_img/'+folder)
    image_name = []
    for image in image_list:
        image_name.append(int(image[:-4]))
    index = (np.min(image_name))
    image_name.clear()
    frame_location_tmp = frame_location(len(image_list),index)
    class_index = 'c0'
    for i in range(0,nrows):
        if(sheet.row_values(i)[0] == folder):
            class_index = sheet.row_values(i)[2]
    frame_location_final = []
    frame_location_final.extend(frame_location_tmp)
    for i in range(0, 5):
        for j in range(0, N):
            frame_location_tmp[j] = frame_location_tmp[j] + 1
        frame_location_final.extend(frame_location_tmp)

    for i in range(0, 6):
        json_dic = {}
        json_dic['label'] = class_index
        json_dic['sequence'] = []
        for j in range(0, N):
            json_dic['sequence'].append(
                '/' + folder + '/' + str('%04d' % frame_location_final[N * i + j]) + '.jpg')
        with open('../dataset/' + str(N) + '_frames_training.txt', 'a') as f:
            f.write(str(json_dic) + '\n')

folder_list = os.listdir('../dataset/test_set_img')
for folder in folder_list:
    image_list = os.listdir('../dataset/test_set_img/'+folder)
    image_name = []
    for image in image_list:
        image_name.append(int(image[:-4]))
    index = (np.min(image_name))
    image_name.clear()
    frame_location_tmp = frame_location(len(image_list),index)
    class_index = 'c0'
    for i in range(0,nrows):
        if(sheet.row_values(i)[0] == folder):
            class_index = sheet.row_values(i)[2]
    frame_location_final = []
    frame_location_final.extend(frame_location_tmp)
    for i in range(0, 5):
        for j in range(0, N):
            frame_location_tmp[j] = frame_location_tmp[j] + 1
        frame_location_final.extend(frame_location_tmp)

    for i in range(0, 6):
        json_dic = {}
        json_dic['label'] = class_index
        json_dic['sequence'] = []
        for j in range(0, N):
            json_dic['sequence'].append(
                '/' + folder + '/' + str('%04d' % frame_location_final[N * i + j]) + '.jpg')
        with open('../dataset/' + str(N) + '_frames_test.txt', 'a') as f2:
            f2.write(str(json_dic) + '\n')
########create txt that is used for driver map training and test #############

att_folder_list = os.listdir('../dataset/training_att_set_img')
for folder2 in att_folder_list:
    att_image_list = os.listdir('../dataset/training_att_set_img/'+folder2)
    for att_image in att_image_list:
        att_image_dir = '../dataset/training_att_set_img/'+folder2+'/'+att_image
        tra_image_dir = '../dataset/training_set_img/'+folder2+'/'+att_image
        if os.path.exists(att_image_dir) and os.path.exists(tra_image_dir):
            write_str = '/training_set_img/'+folder2+'/'+att_image+','+'/training_att_set_img/'+folder2+'/'+att_image
            with open('../dataset/ '+'attention_train.txt', 'a') as f3:
                f3.write(write_str + '\n')

att_folder_list = os.listdir('../dataset/test_att_set_img')
for folder2 in att_folder_list:
    att_image_list = os.listdir('../dataset/test_att_set_img/'+folder2)
    for att_image in att_image_list:
        att_image_dir = '../dataset/test_att_set_img/'+folder2+'/'+att_image
        tra_image_dir = '../dataset/test_set_img/'+folder2+'/'+att_image
        if os.path.exists(att_image_dir) and os.path.exists(tra_image_dir):
            write_str = '/test_set_img/'+folder2+'/'+att_image+','+'/test_att_set_img/'+folder2+'/'+att_image
            with open('../dataset/ '+'attention_test.txt', 'a') as f4:
                f4.write(write_str + '\n')