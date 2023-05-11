# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
from sklearn.model_selection import StratifiedKFold, KFold


def create_kfold_split_uniform(folds, seed=42):
    cache_path = OUTPUT_PATH + 'kfold_split_{}_{}.csv'.format(folds, seed)
    if not os.path.isfile(cache_path):
        kf = KFold(n_splits=folds, random_state=seed, shuffle=True)
        files = glob.glob(INPUT_PATH + 'frames_original/*.mp4')
        names = [os.path.basename(f) for f in files]

        s = pd.DataFrame(names, columns=['video_id'])
        s['fold'] = -1
        for i, (train_index, test_index) in enumerate(kf.split(s.index)):
            s.loc[test_index, 'fold'] = i
        s.to_csv(cache_path, index=False)
        print('No folds: {}'.format(len(s[s['fold'] == -1])))

        for i in range(folds):
            part = s[s['fold'] == i]
            print(i, len(part))
    else:
        print('File already exists: {}'.format(cache_path))


# We only add new data with fold different from 0
def append_to_old_split_with_no_zero(path):
    s = pd.read_csv(path)
    files = glob.glob(INPUT_PATH + 'frames_original/*.mp4')

    new_video_ids = []
    new_folds = []
    for f in files:
        video_id = os.path.basename(f)
        if video_id in list(s['video_id'].values):
            fold_id = s[s['video_id'] == video_id]['fold'].values[0]
        else:
            fold_id = random.randint(1, 5)
        new_video_ids.append(video_id)
        new_folds.append(fold_id)

    t = pd.DataFrame(new_video_ids, columns=['video_id'])
    t['fold'] = new_folds
    t.to_csv(path[:-4] + '_upd.csv', index=False)

    for i in range(5):
        part = t[t['fold'] == i]
        print(i, len(part))


if __name__ == '__main__':
    create_kfold_split_uniform(5, seed=42)
    # append_to_old_split_with_no_zero(OUTPUT_PATH + 'kfold_split_5_42.csv')