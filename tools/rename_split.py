import os
import cv2
import math
import xlrd
import shutil

data_sour = 'E:/dataset/BDDA'#the dir of unzipped BDD-A file
execl_dir = '../annotation/annotation.xls'


##########################################rename videos############################################
book = xlrd.open_workbook(execl_dir)
sheet_train = book.sheets()[1]
nrows_train = sheet_train.nrows
sheet_test = book.sheets()[2]
nrows_test = sheet_test.nrows
sets = ['validation/','training/','test/']
if not os.path.exists('../dataset/training_set'):
    os.makedirs('../dataset/training_set')
for i in range(0, nrows_train):
    no_old = sheet_train.row_values(i)[1]
    no_new = sheet_train.row_values(i)[0]
    for set_tmp in sets:
        video_old_path = data_sour+'/'+set_tmp+'camera_videos/'+str(int(no_old))+'.mp4'
        video_new_path = '../dataset/training_set/' + no_new + '.mp4'
        if os.path.exists(video_old_path):
            shutil.copy(video_old_path, video_new_path)



if not os.path.exists('../dataset/test_set'):
    os.makedirs('../dataset/test_set')
for i in range(0, nrows_test):
    no_old = sheet_test.row_values(i)[1]
    no_new = sheet_test.row_values(i)[0]
    for set_tmp in sets:
        video_old_path = data_sour+'/'+set_tmp+'camera_videos/'+str(int(no_old))+'.mp4'
        video_new_path = '../dataset/test_set/' + no_new + '.mp4'
        if os.path.exists(video_old_path):
            shutil.copy(video_old_path,video_new_path)

#########################################rename attention videos############################################

if not os.path.exists('../dataset/training_att_set'):
    os.makedirs('../dataset/training_att_set')
for i in range(0, nrows_train):
    no_old = sheet_train.row_values(i)[1]
    no_new = sheet_train.row_values(i)[0]
    for set_tmp in sets:
        video_old_path = data_sour+'/'+set_tmp+'gazemap_videos/'+str(int(no_old))+'_pure_hm.mp4'
        video_new_path = '../dataset/training_att_set/' + no_new + '.mp4'
        if os.path.exists(video_old_path):
            shutil.copy(video_old_path, video_new_path)



if not os.path.exists('../dataset/test_att_set'):
    os.makedirs('../dataset/test_att_set')
for i in range(0, nrows_test):
    no_old = sheet_test.row_values(i)[1]
    no_new = sheet_test.row_values(i)[0]
    for set_tmp in sets:
        video_old_path = data_sour+'/'+set_tmp+'gazemap_videos/'+str(int(no_old))+'_pure_hm.mp4'
        video_new_path = '../dataset/test_att_set/' + no_new + '.mp4'
        if os.path.exists(video_old_path):
            shutil.copy(video_old_path,video_new_path)

##################################split videos and attention videos to frames#################################
video_train_list = os.listdir('../dataset/training_set')
for video_tmp in video_train_list:
    if not os.path.exists('../dataset/training_att_set_img/'+video_tmp[:-4]):
        os.makedirs('../dataset/training_att_set_img/'+video_tmp[:-4])
    if not os.path.exists('../dataset/training_set_img/'+video_tmp[:-4]):
        os.makedirs('../dataset/training_set_img/'+video_tmp[:-4])
    traffic_video_dir = '../dataset/training_set/'+video_tmp
    attention_video_dir = '../dataset/training_att_set/'+video_tmp
    traffic_video = cv2.VideoCapture(traffic_video_dir)
    attention_video = cv2.VideoCapture(attention_video_dir)
    frame_count_t = traffic_video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count_a = attention_video.get(cv2.CAP_PROP_FRAME_COUNT)
    alignment_factor = math.floor((frame_count_t - frame_count_a) / 2)

    c = alignment_factor + 1   # the number of traffic  frames is more than the number of attention maps in a BDD-A video
    if attention_video.isOpened():
        rval_a, frame_a = attention_video.read()
    else:
        rval_a = False
    while rval_a:
        rval_a, frame_a = attention_video.read()
        if rval_a:
            frame_tmp = frame_a[96:672, 0:1024]
            frame_res = cv2.resize(frame_tmp, (320, 192))
            cv2.imwrite( '../dataset/training_att_set_img/'+video_tmp[:-4]+'/' + str('%04d' % c) + '.jpg', frame_res)
            c = c + 1
    attention_video.release()

    d = 0
    if traffic_video.isOpened():
        rval_t, frame_t = traffic_video.read()
    else:
        rval_t = False
    while rval_t:
        rval_t, frame_t = traffic_video.read()
        if d>alignment_factor and d<c:
            if rval_t:
                frame_res = cv2.resize(frame_t, (320, 192))
                cv2.imwrite('../dataset/training_set_img/' + video_tmp[:-4] + '/' + str('%04d' % d) + '.jpg', frame_res)
        d = d + 1
    traffic_video.release()


video_test_list = os.listdir('../dataset/test_set')
for video_tmp in video_test_list:
    if not os.path.exists('../dataset/test_att_set_img/'+video_tmp[:-4]):
        os.makedirs('../dataset/test_att_set_img/'+video_tmp[:-4])
    if not os.path.exists('../dataset/test_set_img/'+video_tmp[:-4]):
        os.makedirs('../dataset/test_set_img/'+video_tmp[:-4])
    traffic_video_dir = '../dataset/test_set/'+video_tmp
    attention_video_dir = '../dataset/test_att_set/'+video_tmp
    traffic_video = cv2.VideoCapture(traffic_video_dir)
    attention_video = cv2.VideoCapture(attention_video_dir)
    frame_count_t = traffic_video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count_a = attention_video.get(cv2.CAP_PROP_FRAME_COUNT)
    alignment_factor = math.floor((frame_count_t - frame_count_a) / 2)

    c = alignment_factor + 1   # the number of traffic  frames is more than the number of attention maps in a BDD-A video
    if attention_video.isOpened():
        rval_a, frame_a = attention_video.read()
    else:
        rval_a = False
    while rval_a:
        rval_a, frame_a = attention_video.read()
        if rval_a:
            frame_tmp = frame_a[96:672, 0:1024]
            frame_res = cv2.resize(frame_tmp, (320, 192))
            cv2.imwrite( '../dataset/test_att_set_img/'+video_tmp[:-4]+'/' + str('%04d' % c) + '.jpg', frame_res)
            c = c + 1
    attention_video.release()
    d = 0
    if traffic_video.isOpened():
        rval_t, frame_t = traffic_video.read()
    else:
        rval_t = False
    while rval_t:
        rval_t, frame_t = traffic_video.read()
        if d>alignment_factor and d<c:
            if rval_t:
                frame_res = cv2.resize(frame_t, (320, 192))
                cv2.imwrite('../dataset/test_set_img/' + video_tmp[:-4] + '/' + str('%04d' % d) + '.jpg', frame_res)
        d = d + 1
    traffic_video.release()



