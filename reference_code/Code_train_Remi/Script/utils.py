import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
import torch
import math
import itertools
irange = range
import pickle
from sklearn.metrics import *
import torch.nn.functional as F
import csv
from datetime import date
import torch.nn as nn


def normalize(x, method='standard', axis=None):
    '''Normalizes the input with specified method.

    Parameters
    ----------
    x : array-like
    method : string, optional
        Valid values for method are:
        - 'standard': mean=0, std=1
        - 'range': min=0, max=1
        - 'sum': sum=1
    axis : int, optional
        Axis perpendicular to which array is sliced and normalized.
        If None, array is flattened and normalized.

    Returns
    -------
    res : numpy.ndarray
        Normalized array.
    '''
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid


def nss_loss(y_true, y_pred):
    max_y_pred = K.repeat_elements(
        torch.unsqueeze(K.repeat_elements(torch.unsqueeze(K.max(K.max(y_pred, dim=2), dim=2)),
                                          shape_r_out, dim=-1)), shape_c_out, dim=-1)
    y_pred /= max_y_pred
    y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, dim=-1)
    y_mean = K.repeat_elements(torch.unsqueeze(K.repeat_elements(torch.unsqueeze(torch.unsqueeze(y_mean)),
                                                                 shape_r_out, dim=-1)), shape_c_out, dim=-1)

    y_std = K.std(y_pred_flatten, dim=-1)
    y_std = K.repeat_elements(torch.unsqueeze(K.repeat_elements(torch.unsqueeze(torch.unsqueeze(y_std)),
                                                                shape_r_out, dim=-1)), shape_c_out, dim=-1)

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    return -(K.sum(K.sum(y_true * y_pred, dim=2), dim=2) / K.sum(K.sum(y_true, dim=2), dim=2))


def denormalize(T, coords):
    return (0.5 * ((coords + 1.0) * T))


def bounding_box(x, y, size, color='w'):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def resize_array(x, size):
    # 3D and 4D tensors allowed only
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    # 4D Tensor
    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            img = array2img(x[i])
            img = img.resize((size, size))
            img = np.asarray(img, dtype='float32')
            img = np.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = np.concatenate(res)
        res = np.expand_dims(res, axis=1)
        return res

    # 3D Tensor
    img = array2img(x)
    img = img.resize((size, size))
    res = np.asarray(img, dtype='float32')
    res = np.expand_dims(res, axis=0)
    res /= 255.0
    return res


def img2array(data_path, desired_size=None, expand=False, view=False):
    """
    Util function for loading RGB image into a numpy array.

    Returns array of shape (1, H, W, C).
    """
    img = Image.open(data_path)
    img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype='float32')
    if expand:
        x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def array2img(x):
    """
    Util function for converting anumpy array to a PIL img.

    Returns PIL RGB img.
    """
    x = np.asarray(x)
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype('uint8'), 'RGB')


def plot_images(images, gd_truth):
    images = images.squeeze()
    assert len(images) == len(gd_truth) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i], cmap="Greys_r")

        xlabel = "{}".format(gd_truth[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def prepare_dirs(config):
    for path in [config.data_dir, config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def save_config(config):
    model_name = 'ram_{}_{}x{}_{}'.format(
        config.num_glimpses, config.patch_size,
        config.patch_size, config.glimpse_scale
    )
    filename = model_name + '_params.json'
    param_path = os.path.join(config.ckpt_dir, filename)

    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=2)
#
# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')
#
# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
#
# plt.show()

def save_obj(obj, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_obj(path):
    with open(path, 'rb') as fp:
        itemlist = pickle.load(fp)
        return itemlist


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def calc_auc(y_pred_proba, labels, exp_run_folder, classifier, fold):
    auc = roc_auc_score(labels, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(labels, y_pred_proba)
    curve_roc = np.array([fpr, tpr])
    # dataile_id = open(exp_run_folder + '/roc_{}_{}.txt'.format(classifier, fold), 'w+')
    # np.savetxt(dataile_id, curve_roc)
    # dataile_id.close()
    plt.plot(fpr, tpr, label='ROC curve: AUC={:0.2f} fold {}'.format(auc, str(fold)))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True)
    plt.title('ROC Fold {}'.format(fold))
    plt.legend(loc="lower left")
    if fold == 4:
        plt.savefig(exp_run_folder + '/roc_{}_{}.pdf'.format(classifier, fold), format='pdf')
    return auc


def metric_save(all_pred, config, fold):
    csv_path = '../evaluation/comparaison_global_all.csv'

    y_true = [int(el) for el in all_pred[:, 3]]
    y_pred = [int(el) for el in all_pred[:, 1]]
    y_score = [float(el) for el in all_pred[:, 2]]

    if not os.path.exists(os.path.join(csv_path)):
        with open(csv_path, 'w') as csv_file:
            pass
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file.readlines(), delimiter=',')
    else:

        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file.readlines(), delimiter=',')

    dataset_name = config.data_dir.split('/')[-2]

    if config.additional_desc != '':
        model_name = config.model_name + '_' + config.additional_desc
    else:
        model_name = config.model_name

    if config.optimizer != '':
        model_name = model_name + '_' + config.optimizer
    else:
        model_name = model_name

    output_dir = os.path.join('../evaluation', config.model_name, config.optimizer, dataset_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config_d = (vars(config))

    save_obj(config_d, os.path.join(output_dir, 'config'))

    metrics_to_write = []
    acuracy = accuracy_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_score = roc_auc_score(y_true, y_score)
    calc_auc(y_score, y_true, output_dir, model_name, fold)

    metrics_to_write.append(config.model_name)
    metrics_to_write.append(config.additional_desc)
    metrics_to_write.append(config.optimizer)
    metrics_to_write.append(fold)

    metrics_to_write.append(acuracy)
    metrics_to_write.append(F1)
    metrics_to_write.append(precision)
    metrics_to_write.append(recall)
    metrics_to_write.append(roc_score)

    metrics_to_write.append(config.data_dir)

    ckpt_dir = os.path.join('../ckpt', model_name, dataset_name)
    filename = model_name + '_' + dataset_name + '_' + str(fold) + '_ckpt.pth.tar'
    ckpt_path = os.path.join(ckpt_dir, 'last_epoch', filename)

    metrics_to_write.append(ckpt_path)
    metrics_to_write.append(date.today())
    metrics_to_write = [str(el) for el in metrics_to_write]
    with open(csv_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow([
            'Model', 'Description', 'Optimizer', 'Fold', 'Accuracy', 'F1 score', 'Precision', 'Recall', 'Roc score',
            'Dataset', 'Weights path', 'Date'])
        flag_replace = 0
        for row_number, row in enumerate(csv_reader):
            if row_number != 0:
                if row[10] == ckpt_path:
                    flag_replace = 1
                    csv_writer.writerow(metrics_to_write)
                else:
                    csv_writer.writerow(row)
        if flag_replace == 0:
            csv_writer.writerow(metrics_to_write)

    return metrics_to_write


def write_all_metrics(metrics_tab):
    accuracy = 0
    F1 = 0
    precision = 0
    recall = 0
    roc_score = 0
    for metrics in metrics_tab:
        model_name = metrics[0]
        description = metrics[1]
        optimizer = metrics[2]
        accuracy += float(metrics[4])
        F1 += float(metrics[5])
        precision += float(metrics[6])
        recall += float(metrics[7])
        roc_score += float(metrics[8])
        dataset = metrics[9]
        date = metrics[11]
    to_write_full = [model_name, description, optimizer, accuracy / len(metrics_tab), F1 / len(metrics_tab),
                     precision / len(metrics_tab), recall / len(metrics_tab), roc_score / len(metrics_tab), dataset,
                     date]
    csv_path = '../evaluation/comparaison_global.csv'

    if not os.path.exists(os.path.join(csv_path)):
        with open(csv_path, 'w') as csv_file:
            pass
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file.readlines(), delimiter=',')
    else:

        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file.readlines(), delimiter=',')

    with open(csv_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow([
            'Model', 'Description', 'Optimizer', 'Accuracy', 'F1 score', 'Precision', 'Recall', 'Roc score',
            'Dataset', 'Date'])
        flag_replace = 0
        for row_number, row in enumerate(csv_reader):
            if row_number != 0:
                if row[0] == model_name and row[1] == description and row[2] == optimizer and row[8] == dataset:
                    flag_replace = 1
                    csv_writer.writerow(to_write_full)
                else:
                    csv_writer.writerow(row)
        if flag_replace == 0:
            csv_writer.writerow(to_write_full)


def sort_output(output_dir):
    dir_NP = os.path.join(output_dir, 'sorted/NP')
    if not os.path.exists(dir_NP):
        os.makedirs(dir_NP)
    else:
        rmtree(dir_NP)
        os.makedirs(dir_NP)
    dir_P = os.path.join(output_dir, 'sorted/P')
    if not os.path.exists(dir_P):
        os.makedirs(dir_P)
    else:
        rmtree(dir_P)
        os.makedirs(dir_P)
    print('Average score calculation...')
    score_list = []
    for file_path in os.listdir(output_dir):
        score_sublist = []
        if file_path.endswith('.csv'):
            with open(os.path.join(output_dir, file_path)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count != 0:
                        score_sublist.append([row[0], float(row[2])])
                    line_count += 1
                score_sublist.sort()
                score_list.append(score_sublist)
                print(np.shape(score_sublist))
    score_list = np.transpose(score_list, (1, 0, 2))
    final_score_list = []
    for row in score_list:
        av_score = 0
        for column in row:
            av_score += float(column[1])
        final_score_list.append([row[0][0], int(np.around(av_score / score_list.shape[1], decimals=0))])

    print('Sorting them...')
    for row in final_score_list:
        src = row[0]
        if row[1] == 1:
            dest = os.path.join(dir_P, row[0].split('/')[-1])
        else:
            dest = os.path.join(dir_NP, row[0].split('/')[-1])
        copyfile(src, dest)


# def NSS(saliency_map, map_pred):
#     '''
# 	Normalized scanpath saliency of a saliency map,
# 	defined as the mean value of normalized (i.e., standardized) saliency map at fixation locations.
# 	You can think of it as a z-score. (Larger value implies better performance.)
# 	Parameters
# 	----------
# 	saliency_map : real-valued matrix
# 		If the two maps are different in shape, saliency_map will be resized to match fixation_map..
# 	fixation_map : binary matrix
# 		Human fixation map (1 for fixated location, 0 for elsewhere).
# 	Returns
# 	-------
# 	NSS : float, positive
# 	'''
#
#     map_pred_mean = torch.mean(map_pred)  # calculating the mean value of tensor
#     map_pred_mean = map_pred_mean.item()  # change the tensor into a number
#
#     map_pred_std = torch.std(map_pred)  # calculate the standard deviation
#     map_pred_std = map_pred_std.item()  # change the tensor into a number
#
#     s_map = np.array(saliency_map, copy=False)
#     f_map = np.array(fixation_map, copy=False) > 0.5
#     if s_map.shape != f_map.shape:
#         s_map = resize(s_map, f_map.shape)
#     # Normalize saliency map to have zero mean and unit std
#     s_map = normalize(s_map, method='standard')
#     # Mean saliency value at fixation locations
#     return np.mean(s_map[f_map])


def cc_loss(map_pred, map_gtd):
    epsilon = 1e-8
    map_pred = map_pred.float()
    map_gtd = map_gtd.float()

    map_pred = map_pred.view(1, -1)  # change the map_pred into a tensor with n rows and 1 cols
    map_gtd = map_gtd.view(1, -1)  # change the map_pred into a tensor with n rows and 1 cols

    min1 = torch.min(map_pred)
    max1 = torch.max(map_pred)
    map_pred = (map_pred - min1) / (max1 - min1 + epsilon)  # min-max normalization for keeping KL loss non-NAN

    min2 = torch.min(map_gtd)
    max2 = torch.max(map_gtd)
    map_gtd = (map_gtd - min2) / (max2 - min2 + epsilon)  # min-max normalization for keeping KL loss non-NAN

    map_pred_mean = torch.mean(map_pred)  # calculating the mean value of tensor
    map_pred_mean = map_pred_mean.item()  # change the tensor into a number

    map_gtd_mean = torch.mean(map_gtd)  # calculating the mean value of tensor
    map_gtd_mean = map_gtd_mean.item()  # change the tensor into a number
    # print("map_gtd_mean is :", map_gtd_mean)

    map_pred_std = torch.std(map_pred)  # calculate the standard deviation
    map_pred_std = map_pred_std.item()  # change the tensor into a number
    map_gtd_std = torch.std(map_gtd)  # calculate the standard deviation
    map_gtd_std = map_gtd_std.item()  # change the tensor into a number

    map_pred = (map_pred - map_pred_mean) / (map_pred_std + epsilon)  # normalization
    map_gtd = (map_gtd - map_gtd_mean) / (map_gtd_std + epsilon)  # normalization

    map_pred_mean = torch.mean(map_pred)  # re-calculating the mean value of normalized tensor
    map_pred_mean = map_pred_mean.item()  # change the tensor into a number

    map_gtd_mean = torch.mean(map_gtd)  # re-calculating the mean value of normalized tensor
    map_gtd_mean = map_gtd_mean.item()  # change the tensor into a number

    CC_1 = torch.sum((map_pred - map_pred_mean) * (map_gtd - map_gtd_mean))
    CC_2 = torch.rsqrt(torch.sum(torch.pow(map_pred - map_pred_mean, 2))) * torch.rsqrt(
        torch.sum(torch.pow(map_gtd - map_gtd_mean, 2))) + epsilon
    CC = CC_1 * CC_2
    # print("CC loss is :", CC)
    # CC = -CC  # the bigger CC, the better

    # we put the L1 loss with CC together for avoiding building a new class
    # L1_loss =  torch.mean( torch.abs(map_pred - map_gtd) )
    # print("CC and L1 are :", CC, L1_loss)
    # CC = CC + L1_loss

    return CC


def NSS(saliency_map, fixation_map):
    '''
	Normalized scanpath saliency of a saliency map,
	defined as the mean value of normalized (i.e., standardized) saliency map at fixation locations.
	You can think of it as a z-score. (Larger value implies better performance.)
	Parameters
	----------
	saliency_map : real-valued matrix
		If the two maps are different in shape, saliency_map will be resized to match fixation_map..
	fixation_map : binary matrix
		Human fixation map (1 for fixated location, 0 for elsewhere).
	Returns
	-------
	NSS : float, positive
	'''
    s_shape = np.shape(saliency_map)
    # print(print(np.shape(saliency_map.view(s_shape[0], 1, -1))))
    mean = saliency_map.mean(dim=2).mean(dim=2).squeeze()
    std = torch.std(saliency_map.contiguous().view(s_shape[0], 1, -1), dim=2).squeeze()
    # sum = saliency_map.sum(dim=2).sum(dim=2).squeeze()
    saliency_map = (saliency_map - mean[:, None, None, None]) / std[:, None, None, None]
    # saliency_map = saliency_map / sum[:, None, None, None]
    nss = saliency_map[fixation_map.bool()]
    return nss.mean()


def kl_divergence(y_true, y_pred):
    _EPSILON = 1e-7
    shape_c_out = 256
    shape_r_out = 256
    max_y_pred, _ = torch.max(y_pred, dim=2)
    max_y_pred, _ = torch.max(max_y_pred, dim=2)
    max_y_pred = torch.repeat_interleave(
        torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(max_y_pred, dim=-1),
                                                shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)
    y_pred /= max_y_pred

    sum_y_true = torch.repeat_interleave(
        torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(torch.sum(torch.sum(y_true, dim=2), dim=2), dim=-1),
                                                shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)
    sum_y_pred = torch.repeat_interleave(
        torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(torch.sum(torch.sum(y_pred, dim=2), dim=2), dim=-1),
                                                shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)
    y_true /= (sum_y_true + _EPSILON)
    y_pred /= (sum_y_pred + _EPSILON)

    y_true = y_true.cuda()


    return 10 * torch.sum(torch.sum(y_true * torch.log((y_true / (y_pred + _EPSILON)) + _EPSILON), dim=-1), dim=-1)


def kl_divergence2(y_true, y_pred,im_size = (256, 256)):
    _EPSILON = 1e-7


    nb_pix = (im_size[0]*im_size[1])
    y_true = torch.div(y_true, nb_pix)
    kl = torch.mul(torch.div(y_pred, y_true+_EPSILON), y_pred)
    print(np.shape(kl))


    return kl

def square(x):
    return torch.mul(x, x)


# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    _EPSILON = 1e-7
    shape_c_out = 256
    shape_r_out = 256
    max_y_pred, _ = torch.max(y_pred, dim=2)
    max_y_pred, _ = torch.max(max_y_pred, dim=2)
    max_y_pred = torch.repeat_interleave(
        torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(max_y_pred, dim=-1),
                                                shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)
    y_pred /= max_y_pred

    sum_y_true = torch.repeat_interleave(
        torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(torch.sum(torch.sum(y_true, dim=2), dim=2), dim=-1),
                                                shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)
    sum_y_pred = torch.repeat_interleave(
        torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(torch.sum(torch.sum(y_pred, dim=2), dim=2), dim=-1),
                                                shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)

    y_true /= (sum_y_true + _EPSILON)
    y_pred /= (sum_y_pred + _EPSILON)

    N = shape_r_out * shape_c_out
    sum_prod = torch.sum(torch.sum(y_true * y_pred, dim=2), dim=2)
    sum_x = torch.sum(torch.sum(y_true, dim=2), dim=2)
    sum_y = torch.sum(torch.sum(y_pred, dim=2), dim=2)
    sum_x_square = torch.sum(torch.sum(torch.mul(y_true, y_true), dim=2), dim=2)
    sum_y_square = torch.sum(torch.sum(torch.mul(y_pred, y_pred), dim=2), dim=2)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = torch.sqrt((sum_x_square - torch.mul(sum_x, sum_x) / N) * (sum_y_square - torch.mul(sum_y, sum_y) / N))

    return -2 * num / den


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# Normalized Scanpath Saliency Loss
def nss_2(y_true, y_pred):
    shape_c_out = 256
    shape_r_out = 256
    _EPSILON = 1e-7

    max_y_pred, _ = torch.max(y_pred, dim=2)
    max_y_pred, _ = torch.max(max_y_pred, dim=2)
    max_y_pred = torch.unsqueeze(max_y_pred, dim=-1)

    max_y_pred = torch.repeat_interleave(
        torch.unsqueeze(torch.repeat_interleave(max_y_pred, shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)
    y_pred /= max_y_pred
    y_pred_flatten = torch.flatten(y_pred)

    y_mean = torch.mean(y_pred_flatten, dim=-1)
    y_mean = torch.repeat_interleave(
        torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(torch.unsqueeze(y_mean, dim=-1), dim=-1),
                                                shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)
    # print(np.shape(y_pred_flatten))
    y_std = torch.std(y_pred_flatten, dim=-1)
    # print(np.shape(y_std))
    y_std = torch.repeat_interleave(
        torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(torch.unsqueeze(y_std, dim=-1), dim=-1),
                                                shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)

    y_pred = (y_pred - y_mean.cuda()) / (y_std + _EPSILON)

    y_true = y_true.cuda()

    return -(torch.sum(torch.sum(y_true * y_pred, dim=2), dim=2) / torch.sum(torch.sum(y_true, dim=2), dim=2))


def plot_cm(cm, classes, model_name,
                          normalize=False,
                          title='Confusion matrix', print_cm=True,
                          cmap=plt.cm.Blues, plot_dir='../test/'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        matrix_path = os.path.join(plot_dir, 'confusion_matrix_N_' + model_name + '.png')
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        matrix_path = os.path.join(plot_dir, 'confusion_matrix_' + model_name + '.png')

    if print_cm:
        print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(matrix_path)
    plt.clf()
    plt.close()
