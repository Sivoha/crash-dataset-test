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


def split_videos_on_frames():
    files = glob.glob(INPUT_PATH + 'frames_original/*.mp4')
    out_folder = INPUT_PATH + 'frames/'
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    all_names = []
    all_frame_names = []
    all_frame_numbers = []
    all_targets = []
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

        for j in range(video.shape[0]):
            all_names.append(name)
            frame_name = '{}_frame_{:04d}.png'.format(name, j)
            if not os.path.isfile(out_folder + frame_name):
                cv2.imwrite(out_folder + frame_name, video[j])
            all_frame_names.append(frame_name)
            all_frame_numbers.append(j)
            if data.values[j, 0] != j:
                print('Unexpected!', data.values[j, 0], j)
                exit()

            target = get_class(data.values[j, 1])
            if target == -1:
                print('Unknown crush type: {}'.format(data.values[j, 1]))
                exit()
            all_targets.append(target)

    s = pd.DataFrame(all_names, columns=['video_id'])
    s['frame_name'] = all_frame_names
    s['frame_number'] = np.array(all_frame_numbers).astype(np.int32)
    s['target'] = all_targets
    s.to_csv(INPUT_PATH + 'train.csv', index=False)


if __name__ == '__main__':
    # After extraction of 50 videos it will be ~16 GB
    # After extraction of 118 videos it will be ~30 GB
    split_videos_on_frames()
