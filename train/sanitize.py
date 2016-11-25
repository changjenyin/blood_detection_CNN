import os
import re
import random
TRAIN_DIR = "/tmp3/changjenyin/fine_tuning/train"
VAL_DIR = '/tmp3/changjenyin/fine_tuning/val'
train_pos_video_dir = [f for f in os.listdir(TRAIN_DIR + '/pos')]
train_neg_video_dir = [f for f in os.listdir(TRAIN_DIR + '/neg')]
val_pos_video_dir = [f for f in os.listdir(VAL_DIR + '/pos')]
val_neg_video_dir = [f for f in os.listdir(VAL_DIR + '/neg')]

train_txt = open('/tmp3/changjenyin/fine_tuning/meta/train.txt', 'wb+')
val_txt = open('/tmp3/changjenyin/fine_tuning/meta/val.txt', 'wb+')
sample_rate = 0.2

def randomSample(directory):
    total_num = 0
    images = []
    for video in os.listdir(directory):
        #print video
        shots = {}
        for img in os.listdir(os.path.join(directory+'/'+video)):
            if img == '.DS_Store':
                continue
            shot_idx = re.search(r'image_(.*?)-.*', img).group(1)
            shot_idx = int(shot_idx)

            shot = shots.get(shot_idx, [])
            shot.append(img)
            shots[shot_idx] = shot

        for (idx, imgs) in shots.iteritems():
            img_num = len(imgs)
            sample_num = int(img_num*sample_rate)
            total_num += sample_num
            for i in random.sample(range(0, img_num), sample_num):
                images.append(directory + '/' + video + '/' + imgs[i])

    return total_num, images

train_pos_num, imgs = randomSample(TRAIN_DIR+'/pos')
for img in imgs:
    train_txt.write(img + " 1\n")

neg_num, imgs = randomSample(TRAIN_DIR+'/neg')
for idx in random.sample(range(0, neg_num), train_pos_num):
    train_txt.write(imgs[idx] + " 0\n")

val_pos_num, imgs = randomSample(VAL_DIR+'/pos')
for img in imgs:
    val_txt.write(img + " 1\n")

neg_num, imgs = randomSample(VAL_DIR+'/neg')
for idx in random.sample(range(0, neg_num), val_pos_num):
    val_txt.write(imgs[idx] + " 0\n")

