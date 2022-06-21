# -*- coding: utf-8 -*-
# @Time : 2022/3/8 10:38
# @Author : liuxiangyu
import cv2
import numpy as np
from PIL import Image
import h5py
import pandas as pd
import os
from xml.dom.minidom import parse
import xml.dom.minidom
os.environ['PATH'] = "D:/software/openslide/openslide-win64-20171122/bin" + ";" + os.environ['PATH']
import openslide



def get_coordinates(annotation_file):
    DOMTree = xml.dom.minidom.parse(annotation_file)
    collection = DOMTree.documentElement
    coordinatess = collection.getElementsByTagName("Coordinates")
    polygons = []
    for coordinates in coordinatess:
        coordinate = coordinates.getElementsByTagName("Coordinate")
        poly_coordinates = []
        for point in coordinate:
            x = point.getAttribute("X")
            y = point.getAttribute("Y")
            poly_coordinates.append([float(x), float(y)])
        polygons.append(np.array(poly_coordinates,dtype=int))
    return polygons


def read_attention_scores(h5file):
    file = h5py.File(h5file, 'r')
    attn_dset = file['attention_scores']
    data_unc_dset = file['data_unc']
    total_unc_dset = file['total_unc']
    model_unc_dset = file['model_unc']
    coord_dset = file['coords']


    attn = attn_dset[:]
    data_unc = data_unc_dset[:]
    total_unc = total_unc_dset[:]
    model_unc = model_unc_dset[:]

    coords = coord_dset[:]
    file.close()
    return attn, data_unc, total_unc, model_unc, coords


def normalize(data, total_data):
    m = np.mean(total_data)
    mx = max(total_data)
    mn = min(total_data)
    return np.array([(float(i) - mn) / (mx - mn) for i in data])


def read_shapes(shape_file):
    shape_dict = {}
    with open(shape_file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            line_records = line.split(',')
            shape_dict[line_records[0].split('.')[0]] = [int(line_records[1]), int(line_records[2])]
    return shape_dict


def to_percentiles(scores, uncs_type=None):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100
    if uncs_type:
        length_each = len(scores) // 3
        return scores[uncs_type*length_each:(uncs_type+1)*length_each]
    else:
        return scores

if __name__ == '__main__':
    # slide图像尺寸
    shape_file = 'heatmap_results/images_shape.txt'
    shape_dict = read_shapes(shape_file)

    # 标注文件目录
    annotation_path = 'heatmap_results/annotations'

    # attention h5格式文件，由绘图代码保存出来的
    attention_path = 'heatmap_results/BMIL/scores'
    # attention_path = 'heatmap_results/overlap_attention'

    # 下采样级别
    level = 8

    # patch尺寸
    patch_size = 256 // (2 ** level)

    # attention阈值
    thresholds = np.arange(0, 1, 0.05)

    result_dict_mean = dict()
    for threshold in thresholds:
        result_dict_mean[threshold] = {'iou_mean': [],'gt_coverage_mean': []}

    result_dict_mean_final = {'threshold': [], 'iou_mean': [],'gt_coverage_mean': []}

    # anno_mask_save_path = 'heatmap_results/anno_mask'
    # atten_mask_save_path = 'heatmap_results/block_atten_mask'
    #
    # if not os.path.exists(anno_mask_save_path):
    #     os.makedirs(anno_mask_save_path)
    #
    # if not os.path.exists(atten_mask_save_path):
    #     os.makedirs(atten_mask_save_path)


    for attention_file in os.listdir(attention_path):
        # 得到病理图片名称
        slide_name = attention_file.split('_')[0] + '_' +attention_file.split('_')[1]

        # 获取病理图片的大小：
        w, h = shape_dict[slide_name]

        # 获取对应的标注xml
        polygons = get_coordinates(os.path.join(annotation_path, slide_name + '.xml'))
        img_anno = np.zeros((h // (2 ** level), w // (2 ** level), 1), np.uint8)
        for polygon in polygons:
            polygon = polygon // (2**level)
            cv2.fillConvexPoly(img_anno, polygon, 255)
        # cv2.imwrite(os.path.join(anno_mask_save_path, slide_name + '.jpg'), img_anno)
        img_anno = img_anno.reshape((h // (2**level), w // (2**level)))
        img_anno = img_anno // 255

        # 读取attention值
        attn, data_unc, total_unc, model_unc, coords = read_attention_scores(os.path.join(attention_path, attention_file))


        uncs = []
        uncs.extend(data_unc)
        uncs.extend(total_unc)
        uncs.extend(model_unc)
        uncs = np.array(uncs)

        scores = to_percentiles(uncs, uncs_type=2)

        scores = scores.flatten()
        scores = normalize(scores, scores)
        img_atten = np.zeros((h // (2 ** level), w // (2 ** level), 1), np.uint8)
        for score, coord in zip(scores, coords):
            x = coord[0] // (2 ** level)
            y = coord[1] // (2 ** level)
            img_atten[y:y + patch_size, x: x + patch_size, :] = score * 255

        for threshold in thresholds:
            img_temp = img_atten.copy()
            ret, img_temp = cv2.threshold(img_temp, 255 * threshold, 255, cv2.THRESH_BINARY)
            img_temp_save = img_temp.copy()
            img_temp = img_temp // 255
            union = img_anno + img_temp
            union_nums = np.sum(union > 0)
            intersection_nums = np.sum(union == 2)
            iou = intersection_nums / union_nums
            gt_coverage = intersection_nums / np.sum(img_anno == 1)

            result_dict_mean[threshold]['iou_mean'].append(iou)
            result_dict_mean[threshold]['gt_coverage_mean'].append(gt_coverage)


    for k, v in result_dict_mean.items():
        result_dict_mean_final['threshold'].append(k)
        result_dict_mean_final['iou_mean'].append(np.mean(v['iou_mean']))
        result_dict_mean_final['gt_coverage_mean'].append(np.mean(v['gt_coverage_mean']))

    df = pd.DataFrame(result_dict_mean_final)

    df.to_csv("heatmap_results/BMIL/model_unc_block_mean.csv", index=False)
