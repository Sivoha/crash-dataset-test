# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *


def check_video_sizes():
    files = glob.glob(INPUT_PATH + 'frames_50_video/*.mp4')
    print(len(files))

    res = dict()
    for f in files:
        video = read_video(f)
        print(os.path.basename(f), video.shape)
        if video.shape[:2] in res:
            res[video.shape[:2]] += 1
        else:
            res[video.shape[:2]] = 1

    print(res)


def check_train_data():
    s = pd.read_csv(INPUT_PATH + 'train.csv')
    print(s['target'].value_counts())


if __name__ == '__main__':
    # check_video_sizes()
    check_train_data()

