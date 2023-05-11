# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = "1"
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import tqdm
try:
    from .a01_settings import *
    from .a03_models_2D import *
except:
    from a01_settings import *
    from a03_models_2D import *
from torch.utils.data import DataLoader, Dataset, Sampler
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import torch
import math
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error, mean_squared_error


GLOBAL_AUG = None
GLOBAL_CACHE = dict()
KFOLD_NUMBER = 5
FOLD_LIST = [0]
DIR_PREFIX = os.path.basename(os.path.dirname(__file__)) + '_' + os.path.basename(__file__)
MODELS_PATH_LOCAL = MODELS_PATH + DIR_PREFIX + '_' + os.path.basename(KFOLD_SPLIT_FILE)[:-4] + '_' + str(RUN_NUMBER) +'_' + str(SHAPE_SIZE[-1]) + '/'
if not os.path.isdir(MODELS_PATH_LOCAL):
    os.mkdir(MODELS_PATH_LOCAL)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_target(df):
    ids = df['frame_name'].values
    answers = df['target'].values
    # Class 2 we make class 0 (because we need to classify only crash parts)
    answers[answers > 1.5] = 0
    return ids, answers


def get_train_transforms():
    return A.Compose(
        [
            A.ShiftScaleRotate(p=0.5, shift_limit=0.05, scale_limit=0.1, rotate_limit=45, border_mode=cv2.BORDER_REFLECT),
            A.RandomCropFromBorders(p=0.3, crop_value=0.1),
            A.OneOf([
                A.MedianBlur(p=1.0, blur_limit=7),
                A.Blur(p=1.0, blur_limit=7),
                A.GaussianBlur(p=1.0, blur_limit=7),
            ], p=0.3),
            A.Resize(height=SHAPE_SIZE[1], width=SHAPE_SIZE[2], p=1),
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0, brightness_limit=0.2, contrast_limit=0.2),
                A.RGBShift(p=1.0, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20)
            ], p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.05 * 255), p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            ], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ],
        p=1.0
    )


def get_train_transforms_simple():
    return A.Compose(
        [
            A.Resize(height=SHAPE_SIZE[1], width=SHAPE_SIZE[2], p=1),
            A.HorizontalFlip(p=0.5),
            # A.RandomRotate90(p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ],
        p=1.0
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=SHAPE_SIZE[1], width=SHAPE_SIZE[2], p=1),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ],
        p=1.0
    )


class DatasetTrain(Dataset):

    def __init__(self, image_ids, answers, root_path, transform=None, get_sum_pixels=False):
        self.image_ids = image_ids
        self.answers = answers
        self.root_path = root_path
        self.transform = transform
        self.soft_labels = False
        self.get_sum_pixels = get_sum_pixels

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_path = self.root_path + self.image_ids[index]
        image = cv2.imread(image_path)
        label = self.answers[index]

        if self.soft_labels is True:
            if label == 0:
                label = 0.0000001
            else:
                label = 0.9999999

        label = np.expand_dims(np.array(label), axis=-1)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, label, self.image_ids[index]


class DatasetTest(Dataset):

    def __init__(self, image_ids, root_path, transform=None, get_sum_pixels=False):
        self.image_ids = image_ids
        self.root_path = root_path
        self.transform = transform
        self.get_sum_pixels = get_sum_pixels

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_path = self.root_path + self.image_ids[index]
        image = cv2.imread(image_path)
        sum_pixels = image.sum().astype(np.int32)

        image_big = np.zeros((image.shape[0] + 128, image.shape[1] + 128, image.shape[2]), dtype=np.uint8)
        image_big[64:-64, 64:-64, :] = image
        image = image_big

        if self.transform is not None:
            image = self.transform(image=image)['image']

        if self.get_sum_pixels:
            return image, self.image_ids[index], sum_pixels

        return image, self.image_ids[index]


def get_datasets(fold_num):
    train = pd.read_csv(INPUT_PATH + 'train.csv')
    s = pd.read_csv(KFOLD_SPLIT_FILE, dtype={'video_id': str})

    part = s[s['fold'] != fold_num].copy()
    train_ids = part['video_id'].values
    train_df = train[train['video_id'].isin(train_ids)]
    image_ids_train, answers_train = get_target(train_df)

    part = s[s['fold'] == fold_num].copy()
    valid_ids = part['video_id'].values
    valid_df = train[train['video_id'].isin(valid_ids)]
    image_ids_valid, answers_valid = get_target(valid_df)

    tr = get_train_transforms_simple()
    if USE_HARD_AUG:
        tr = get_train_transforms()

    train_dataset = DatasetTrain(
        image_ids_train,
        answers_train,
        INPUT_PATH + 'frames/',
        tr,
    )

    valid_dataset = DatasetTrain(
        image_ids_valid,
        answers_valid,
        INPUT_PATH + 'frames/',
        get_valid_transforms()
    )

    return train_dataset, valid_dataset


class RandomSampler_uni_classes(Sampler):
    def __init__(self, image_ids, targets, num_samples):
        Sampler.__init__(self, image_ids)
        if num_samples is None:
            self.num_samples = len(image_ids)
        else:
            self.num_samples = num_samples
        self.ids_0 = []
        self.ids_1 = []
        for i, t in enumerate(targets):
            if t == 0:
                self.ids_0.append(i)
            else:
                self.ids_1.append(i)
        random.shuffle(self.ids_0)
        random.shuffle(self.ids_1)
        print('Target 0: {} Target 1: {}'.format(len(self.ids_0), len(self.ids_1)))

    def __iter__(self):
        for i in range(self.num_samples):
            # print('\tcalling Sampler:__iter__')
            if random.randint(0, 1) == 0:
                index = random.choice(self.ids_0)
            else:
                index = random.choice(self.ids_1)
            yield index

    def __len__(self):
        return self.num_samples


def train_step(model, device, train_loader, optimizer, epoch):
    start_time = time.time()
    loss_func = torch.nn.BCEWithLogitsLoss()
    model.train()
    tk0 = tqdm.tqdm(train_loader, total=int(len(train_loader)))
    running_loss = 0.0
    for batch_idx, (data, target, _) in enumerate(tk0):
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tk0.set_postfix(loss=(running_loss / (batch_idx + 1)))
    print('Time: {:.2f}'.format(time.time() - start_time))


def valid_step(model, device, valid_loader):
    start_time = time.time()
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = torch.nn.BCEWithLogitsLoss()
    full_output = []
    full_target = []
    full_ids = []
    with torch.no_grad():
        for data, target, img_ids in valid_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            output = model(data)

            test_loss += loss_func(output, target)
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            d1 = target.cpu().numpy().flatten()
            o1 = torch.sigmoid(output).cpu().numpy().flatten()
            full_target += list(d1)
            full_output += list(o1)
            full_ids += list(img_ids)

    test_loss /= len(valid_loader)

    if 0:
        print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset))
        )

    preds = np.array(full_output)
    valid_answs = np.array(full_target)
    full_ids = np.array(full_ids)

    preds_max = np.round(preds).astype(np.int32)
    score_acc = accuracy_score(valid_answs.astype(np.int32), preds_max)
    score_ll = log_loss(valid_answs, np.clip(preds, 0.0000001, 0.9999999))
    score_auc = roc_auc_score(valid_answs, preds)
    mae = mean_absolute_error(valid_answs, preds)
    rmse = mean_squared_error(valid_answs, preds, squared=False)
    if 1:
        # print("MAE: {:.6f} RMSE: {:.6f}".format(mae, rmse))
        print('Acc: {:.6f} LL: {:.6f} AUC: {:.6f} Time: {:.2f}'.format(score_acc, score_ll, score_auc, time.time() - start_time))
    return score_acc, score_auc, score_ll, preds, valid_answs, full_ids


def save_model_weights(model, path):
    model.eval()
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
        }, path)
    except:
        torch.save({
            'model_state_dict': model.module.state_dict(),
        }, path)


def convert_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict['model_state_dict'].items():
        if 'module.' in k:
            name = k[7:]  # remove `module.`
        else:
            name = k
        # print(k, name)
        new_state_dict[name] = v
    # load params
    return new_state_dict


def load_model_weights(model, path):
    model.eval()
    data = torch.load(path)
    try:
        model.load_state_dict(data['model_state_dict'])
    except:
        try:
            model.module.load_state_dict(data['model_state_dict'])
        except:
            new_state_dict = convert_state_dict(data)
            try:
                model.load_state_dict(new_state_dict)
            except:
                model.module.load_state_dict(new_state_dict)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def train_single_model(fold_num, model_path):
    epochs = EPOCHS
    optim_type = OPTIMIZER
    dropout = DROPOUT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {} Epochs: {} Batch size train: {}'.format(device, epochs, BATCH_SIZE_TRAIN))
    print('Go fold: {}'.format(fold_num))

    model_name = 'effnet_v2_l_pytorch'
    history_file = open(MODELS_PATH_LOCAL + 'history_{}.txt'.format(model_name), 'w')
    cnn_type = '{}_optim_{}_drop_{}'.format(model_name, optim_type, dropout)
    print('Creating and compiling {}...'.format(cnn_type))

    model = Model_2D_pretrained_effnet(dropout_val=dropout)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    if model_path is not None:
        print('Load weights: {}'.format(model_path))
        load_model_weights(
            model,
            model_path
        )
    model.to(device)

    train_dataset, valid_dataset = get_datasets(fold_num)

    batch_size_train = BATCH_SIZE_TRAIN * torch.cuda.device_count()
    batch_size_valid = BATCH_SIZE_VALID * torch.cuda.device_count()

    if 0:
        random_sampler_train = RandomSampler_uni_classes(image_ids_train, answers_train,
                                                         batch_size_train * STEPS_PER_EPOCH)
        train_loader = DataLoader(train_dataset,
                               sampler=random_sampler_train,
                               batch_size=batch_size_train,
                               num_workers=THREADS,
                               shuffle=False)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            num_workers=THREADS,
            shuffle=True
        )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size_valid,
        num_workers=THREADS,
        shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if not USE_ONE_CYCLE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=3,
            verbose=True,
        )

    else:
        lf = one_cycle(START_LEARNING_RATE, 0.01, epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if 0:
        for epoch in range(epochs):
            scheduler.step()
            print(scheduler.get_last_lr())
    scheduler.last_epoch = 0

    best_auc = -1
    best_acc = -1
    best_ll = 100
    best_rmse = 100000
    best_mae = 100000
    last_improvement = 0
    for epoch in range(1, epochs + 1):
        print('Start train epoch: {} from {} LR: {}'.format(epoch, epochs, scheduler.get_last_lr()))
        if epoch - last_improvement > (EARLY_STOPPING + 1):
            print('No improvements for {} epochs. Stop training!'.format(EARLY_STOPPING))
            break
        train_step(model, device, train_loader, optimizer, epoch)
        acc, auc, ll, _, _, _ = valid_step(model, device, valid_loader)
        if not USE_ONE_CYCLE_SCHEDULER:
            scheduler.step(rmse)

        out_path = MODELS_PATH_LOCAL + '{}_fold_{}-{}px-auc_{:.6f}-acc-{:.6f}-ll-{:.6f}-ep-{:02d}.pt'.format(model_name, fold_num, SHAPE_SIZE[-1], auc, acc, ll, epoch)
        save_model_weights(model, out_path)
        save_model_weights(model, MODELS_PATH_LOCAL + 'last.pt')
        history_file.write("Epoch {:20d} AUC: {:.6f} ACC: {:.6f}\n".format(epoch, auc, acc))
        history_file.flush()
        if auc > best_auc:
            best_auc = auc
            last_improvement = epoch
        if acc > best_acc:
            best_acc = acc
        if ll < best_ll:
            best_ll = ll
        if USE_ONE_CYCLE_SCHEDULER:
            scheduler.step()

    print('Best AUC: {:.6f}'.format(best_auc))
    history_file.close()
    return


def check_datasets():
    train_dataset, valid_dataset = get_datasets(0)
    for image, target, ids in train_dataset:
        print(image.shape, target)
        image = image.cpu().numpy().transpose((1, 2, 0))
        show_image(image)
    exit()


def find_start_weights(search_path, folds=5):
    all_models = []
    for fold in range(folds):
        files = glob.glob(search_path + '*_fold_{}*.pt'.format(fold))
        best_rmse = 1000000
        best_model = None
        for f in files:
            arr = f.split('auc_')
            rmse = float(arr[-1].split('-')[0])
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = f
        all_models.append(best_model)
    return all_models


if __name__ == '__main__':
    start_time = time.time()
    random.seed(start_time)
    seed_everything(42)

    # check_datasets()

    if RUN_NUMBER == 1:
        start_weights = [
            None,
            None,
            None,
            None,
            None,
        ]
    if RUN_NUMBER == 2:
        path = MODELS_PATH + 'net_v01_effnet_1_classes_r30_train_model.py_kfold_split_5_42_1_384/'
        start_weights = find_start_weights(path, 5)
    if RUN_NUMBER == 3:
        path = MODELS_PATH + 'net_v08_regression_effnet_max_r30_train_model.py_kfold_split_5_42_fix_2_384/'
        start_weights = find_start_weights(path, 5)
    if RUN_NUMBER == 4:
        path = MODELS_PATH + 'net_v08_regression_effnet_max_r30_train_model.py_kfold_split_5_42_fix_3_512/'
        start_weights = find_start_weights(path, 5)
    if RUN_NUMBER == 5:
        path = MODELS_PATH + 'net_v08_regression_effnet_max_r30_train_model.py_kfold_split_5_42_fix_4_512/'
        start_weights = find_start_weights(path, 5)
    if RUN_NUMBER == 6:
        path = MODELS_PATH + 'net_v08_regression_effnet_max_r30_train_model.py_kfold_split_5_42_fix_5_768/'
        start_weights = find_start_weights(path, 5)

    for i, s in enumerate(start_weights):
        if s is not None:
            print("Fold {}: {}".format(i, os.path.basename(s)))

    for kf in range(KFOLD_NUMBER):
        if kf not in FOLD_LIST:
            continue
        train_single_model(kf, start_weights[kf])
    print('Time: {:.0f} sec'.format(time.time() - start_time))


'''

'''