# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 2
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
import argparse
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import timm
import torch
import cv2
import time
import os
import tqdm
import numpy as np
import pandas as pd


USE_TTA = 0
FULL_TRAIN = True
SHAPE_SIZE = (3, 384, 384)
NUM_CLASSES = 1
BATCH_SIZE_VALID = 12
DROPOUT = 0.1
THREADS = 1
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + '/'
INPUT_PATH = ROOT_PATH + 'input/'
CACHE_PATH = ROOT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)


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


def Model_2D_pretrained_effnet(
    dropout_val=0.2,
    out_channels=1,
):
    m = timm.create_model(
        'tf_efficientnetv2_l_in21k',
        num_classes=out_channels,
        drop_rate=dropout_val,
        pretrained=True,
    )
    m.eval()
    return m


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


class DatasetInference(Dataset):

    def __init__(self, video, transform):
        self.video = video
        self.transform = transform

    def __len__(self):
        return self.video.shape[0]

    def __getitem__(self, index):
        image = self.video[index]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=SHAPE_SIZE[1], width=SHAPE_SIZE[2], p=1),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ],
        p=1.0
    )


def create_video(image_list, out_file, fps):
    height, width = image_list[0].shape[:2]
    # fourcc = cv2.VideoWriter_fourcc(*'DIB ')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    # fourcc = -1
    video = cv2.VideoWriter(out_file, fourcc, fps, (width, height), True)
    for im in image_list:
        if len(im.shape) == 2:
            im = np.stack((im, im, im), axis=2)
        video.write(im.astype(np.uint8))
    cv2.destroyAllWindows()
    video.release()


def read_video(f, verbose=False, output_info=False):
    cap = cv2.VideoCapture(f)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0
    frame_list = []
    if verbose:
        print('ID: {} Video length: {} Width: {} Height: {} FPS: {}'.format(os.path.basename(f), length, width, height, fps))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        frame_list.append(frame.copy())
        current_frame += 1

    frame_list = np.array(frame_list, dtype=np.uint8)
    cap.release()
    if output_info:
        return frame_list, fps

    return frame_list


def predict_with_model(options):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    # Read input video
    video, fps = read_video(options['input_video'], output_info=True)
    print("Video shape: {} FPS: {}".format(video.shape, fps))

    # Read input_csv if given
    csv = None
    if 'input_csv' in options:
        csv = pd.read_csv(options['input_csv'])
        csv['type_num'] = -1
        csv.loc[csv['type'] == 'before crush', 'type_num'] = 0
        csv.loc[csv['type'] == 'crush moment', 'type_num'] = 1
        csv.loc[csv['type'] == 'after crush', 'type_num'] = 0
        csv['type_num'] = csv['type_num'].astype(np.float32)

    # Prepare model
    model = Model_2D_pretrained_effnet(
        dropout_val=DROPOUT,
        out_channels=NUM_CLASSES
    )
    if not os.path.isfile(options['model']):
        print('Cant find model file at location: {}'.format(options['model']))
        exit()
    load_model_weights(model, options['model'])
    model.to(device)

    inference_dataset = DatasetInference(
        video,
        get_valid_transforms()
    )
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=BATCH_SIZE_VALID,
        num_workers=THREADS,
        shuffle=False
    )

    all_preds = []
    with torch.no_grad():
        for data in tqdm.tqdm(inference_loader):
            # print(data.shape)
            data = data.to(device, dtype=torch.float)
            output = model(data)
            # print(output.shape)
            output = torch.sigmoid(output).cpu().numpy().squeeze()
            all_preds.append(output)
    all_preds = np.concatenate(all_preds, axis=0)

    # Write CSV_File
    out_csv_file = options['input_video'] + '.csv'
    out = open(out_csv_file, 'w')
    out.write('frame_id,pred\n')
    for i, p in enumerate(all_preds):
        out.write('{},{:.6f}\n'.format(i, p))
    out.close()
    print('Predictions saved in: {}'.format(out_csv_file))

    # Calc metrics (we expect all frames goes in order from 0 to N-1 in CSV-file)
    if csv is not None:
        preds = all_preds
        answs = csv['type_num'].values
        answs = answs.astype(np.float32)
        preds_max = np.round(preds).astype(np.int32)
        score_acc = accuracy_score(answs.astype(np.int32), preds_max)
        score_ll = log_loss(answs, np.clip(preds, 0.0000001, 0.9999999))
        score_auc = roc_auc_score(answs, preds)
        mae = mean_absolute_error(answs, preds)
        rmse = mean_squared_error(answs, preds, squared=False)
        print('Acc: {:.6f} LL: {:.6f} AUC: {:.6f}'.format(score_acc, score_ll, score_auc))
        print("MAE: {:.6f} RMSE: {:.6f}".format(mae, rmse))

    if 'output_video' in options:
        if options['output_video']:
            updated_video = []
            for i in range(video.shape[0]):
                # print('Here {}'.format(i))
                frame = video[i]
                cv2.putText(
                    img=frame,
                    text="Crash prob: {:.4f}".format(all_preds[i]),
                    org=(50, 50),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1,
                    color=[1, 1, 1],
                    lineType=cv2.LINE_AA,
                    thickness=4
                )
                cv2.putText(
                    img=frame,
                    text="Crash prob: {:.4f}".format(all_preds[i]),
                    org=(50, 50),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1,
                    color=[255, 255, 255],
                    lineType=cv2.LINE_AA,
                    thickness=2
                )
                if csv is not None:
                    answ = csv['type_num'].values[i]
                    cv2.putText(
                        img=frame,
                        text="Crash real: {:.4f}".format(answ),
                        org=(50, 120),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1,
                        color=[1, 1, 1],
                        lineType=cv2.LINE_AA,
                        thickness=4
                    )
                    cv2.putText(
                        img=frame,
                        text="Crash real: {:.4f}".format(answ),
                        org=(50, 120),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1,
                        color=[255, 255, 255],
                        lineType=cv2.LINE_AA,
                        thickness=2
                    )

                updated_video.append(frame)
            output_video_path = options['output_csv'] + '.mp4'
            create_video(updated_video, output_video_path, fps)
            print('Video saved in: {}'.format(output_video_path))


if __name__ == '__main__':
    start_time = time.time()

    m = argparse.ArgumentParser()
    m.add_argument("--input_video", "-i", type=str, help="Input audio location", required=True)
    m.add_argument("--input_csv", "-ic", type=str, help="Input CSV location. If added metrics will be calculated.", required=False)
    m.add_argument("--output_csv", "-o", type=str, help="Output CSV-file location", required=True)
    m.add_argument("--model", "-m", type=str, help="Model weights file", required=True)
    m.add_argument("--width", type=int, help="Frame width", required=False, default=384)
    m.add_argument("--height", type=int, help="Frame height", required=False, default=384)
    m.add_argument("--output_video", action='store_true', help="Output video or not")
    options = m.parse_args().__dict__
    print("Options: ".format(options))
    for el in options:
        print('{}: {}'.format(el, options[el]))
    predict_with_model(options)
    print('Time: {:.0f} sec'.format(time.time() - start_time))


"""
Example:
    python net_v05_effnet_1_classes_more_input_data/r60_inference.py
    --input_video ../../input/frames_original/w2_1.mp4
    --input_csv ../../input/frames_original/w2_1.csv
    --output_csv ../../cache/w2_1_proc.mp4.csv
    --model ../../models/net_v01_effnet_v2_l_384/best/effnet_v2_l_pytorch_fold_0-384px-auc_0.700435-acc-0.794321-ll-0.473547-ep-11.pt
    --output_video
    --width 384
    --height 384
"""