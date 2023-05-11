# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 2
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

try:
    from .r30_train_model import *
except:
    from r30_train_model import *
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error, mean_squared_error

USE_TTA = 0
FULL_TRAIN = True


def store_debug(data, target, output):
    data = data.cpu().numpy()
    target = target.cpu().numpy()
    output = output.cpu().numpy()

    out = open(CACHE_PATH + 'debug.txt', 'w')
    for i in range(len(data)):
        for j in range(target.shape[1]):
            out.write(' {:.6f}'.format(target[i, j]))
        out.write('\n')
        for j in range(target.shape[1]):
            out.write(' {:.6f}'.format(output[i, j]))
        out.write('\n')
        out.write('\n')
    out.close()


def get_masks_tta8(data, model):
    data0 = data
    data1 = torch.rot90(data, 1, (2, 3))
    data2 = torch.rot90(data, 2, (2, 3))
    data3 = torch.rot90(data, 3, (2, 3))
    data4 = torch.flip(data, [2])
    data5 = torch.rot90(data4, 1, (2, 3))
    data6 = torch.rot90(data4, 2, (2, 3))
    data7 = torch.rot90(data4, 3, (2, 3))

    pred_masks = []
    for i, d in enumerate([data0, data1, data2, data3, data4, data5, data6, data7]):
        pred_mask = model(d)
        pred_masks.append(pred_mask.clone())
    pred_masks = torch.stack(pred_masks, 0)
    pred_mask = torch.mean(pred_masks, dim=0)
    return pred_mask


def valid_step_v1(model, device, valid_loader):
    start_time = time.time()
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = torch.nn.BCEWithLogitsLoss()
    full_output = []
    full_target = []
    full_ids = []
    full_counts_real = []
    full_counts_answ = []
    with torch.no_grad():
        for data, target, img_ids in valid_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            output = model(data)
            # print(data.shape, target.shape, output.shape)
            if 0:
                # print(pos_first_pred)
                # print(pos_first_real)
                mae = mean_absolute_error(pos_first_real, pos_first_pred)
                rmse = mean_squared_error(pos_first_real, pos_first_pred, squared=False)
                # print("MAE: {:.6f} RMSE: {:.6f}".format(mae, rmse))
                # store_debug(data, target, output)
            test_loss += loss_func(output, target)
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            d1 = target.cpu().numpy().flatten()
            o1 = torch.sigmoid(output).cpu().numpy()
            # o1 = o1.cpu().numpy().flatten()
            full_target += list(d1)
            full_output += list(o1)
            full_ids += list(img_ids)

    test_loss /= len(valid_loader)

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
        print("MAE: {:.6f} RMSE: {:.6f}".format(mae, rmse))
        print('Acc: {:.6f} LL: {:.6f} AUC: {:.6f} Time: {:.2f}'.format(score_acc, score_ll, score_auc, time.time() - start_time))
    return score_acc, score_auc, score_ll, preds, valid_answs, full_ids


def valid_model(model_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    all_answs = []
    all_preds = []
    all_ids = []
    raw_preds = dict()
    model_list_name = ''
    for fold_num in range(len(model_list)):
        fold_time = time.time()
        if len(model_list[fold_num]) == 0:
            print('Skip fold: {}'.format(fold_num))
            continue

        train_dataset, valid_dataset = get_datasets(fold_num)
        print('Fold length: {}'.format(len(valid_dataset)))

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE_VALID,
            num_workers=THREADS,
            shuffle=False
        )

        fold_preds = []
        fold_answers = None
        fold_ids = None
        for m in model_list[fold_num]:
            print('Load: {}'.format(m))
            model = Model_2D_pretrained_effnet(dropout_val=DROPOUT, out_channels=NUM_CLASSES)
            if m is not None:
                load_model_weights(model, m)
            model.to(device)
            model_list_name = m
            acc, auc, ll, preds, answers, ids_order = valid_step_v1(model, device, valid_loader)
            fold_preds.append(preds)
            if fold_ids:
                if tuple(fold_ids) != tuple(ids_order):
                    print('Different order of IDs!')
                    exit()
                if tuple(fold_answers) != tuple(answers):
                    print('Different order of Answers!')
                    exit()
            else:
                fold_ids = ids_order
                fold_answers = answers

        fold_preds = np.array(fold_preds).mean(axis=0)
        all_answs += list(fold_answers)
        all_preds += list(fold_preds)
        all_ids += list(fold_ids)

    all_answs = np.array(all_answs, dtype=np.int32)
    all_preds = np.array(all_preds, dtype=np.float32)
    preds_round = np.round(all_preds).astype(np.int32)

    score_acc = accuracy_score(all_answs.astype(np.int32), preds_round)
    score_ll = log_loss(all_answs, np.clip(all_preds, 0.0000001, 0.9999999))
    score_auc = roc_auc_score(all_answs, all_preds)

    feat = pd.DataFrame(all_ids, columns=['image_id'])
    feat['target'] = all_answs
    feat['pred'] = all_preds
    out_feat_file = FEATURES_PATH + '{}_auc_{:.4f}_acc_{:.4f}_ll_{:.4f}_{}_train.csv'.format(os.path.basename(model_list_name)[:-3], score_auc, score_acc, score_ll, USE_TTA)
    feat.to_csv(out_feat_file, index=False)
    save_in_file(raw_preds, out_feat_file[:-4] + '.pklz')

    print('All valid. Acc: {:.6f} LL: {:.6f} AUC: {:.6f} Time: {:.2f}'.format(score_acc, score_ll, score_auc, time.time() - start_time))
    return score_auc, score_acc, score_ll, out_feat_file


def get_best_model_list_advanced(path1, fold_num, max_models_per_fold):
    model_list = []
    for i in range(fold_num):
        files = glob.glob(path1 + '*fold_{}-*.pt'.format(i))
        if len(files) == 0:
            model_list.append([])
            continue
        models_data = dict()
        best_score = None
        for f in files:
            arr = os.path.basename(f).split('-')
            # print(arr)
            score = float(arr[2].split('_')[1])
            models_data[f] = score
        models_data = sort_dict_by_values(models_data, reverse=True)
        paths = []
        for j in range(min(max_models_per_fold, len(models_data))):
            paths.append(models_data[j][0])
        model_list.append(paths)
        print('Fold: {} Model: {} Score: {} Total models to use: {}'.format(i, os.path.basename(paths[0]), models_data[0][1], len(paths)))
    return model_list


if __name__ == '__main__':
    start_time = time.time()

    if 1:
        path1 = MODELS_PATH + 'net_v01_effnet_1_classes_r30_train_model.py_kfold_split_5_42_1_384/'
        model_list = get_best_model_list_advanced(path1, 5, 1)
        print(model_list)

    score_auc, score_acc, score_ll, out_feat_file = valid_model(model_list)
    print('Time: {:.0f} sec'.format(time.time() - start_time))
