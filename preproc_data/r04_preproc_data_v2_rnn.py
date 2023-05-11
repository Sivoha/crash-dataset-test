# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *


def get_class(class_name):
    if class_name == 'before crush':
        return 0
    elif class_name == 'crush moment':
        return 1
    elif class_name == 'after crush':
        return 2
    else:
        return -1


def create_videos_hdf():
    files = glob.glob(INPUT_PATH + 'frames_50_video/*.mp4')
    out_folder = INPUT_PATH + 'pickles/'
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    for i in range(len(files)):
        f = files[i]
        name = os.path.basename(f)
        print('Go for [{}] {}'.format(i, name))
        video = read_video(f)
        data = pd.read_csv(f[:-4] + '.csv')
        print(video.shape, data.shape)
        if video.shape[0] != data.shape[0]:
            print('Some problem with CSV file: {} != {}. Check frame number'.format(video.shape[0], data.shape[0]))
            exit()

        all_targets = []
        for j in range(video.shape[0]):
            if data.values[j, 0] != j:
                print('Unexpected!', data.values[j, 0], j)
                exit()

            target = get_class(data.values[j, 1])
            if target == -1:
                print('Unknown crush type: {}'.format(data.values[j, 1]))
                exit()
            all_targets.append(target)
        all_targets = np.array(all_targets).astype(np.int8)

        save_in_file_fast((video, all_targets), out_folder + '{}.pkl'.format(name))


if __name__ == '__main__':
    # After extraction of 50 videos it will be ~46 GB
    create_videos_hdf()
