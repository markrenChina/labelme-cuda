#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import random
import copy
import math
import PIL
import multiprocessing
from PIL import Image, ImageDraw

import imgviz
import numpy as np

import labelme

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


def points_to_masks(img_shape, label_file, class_name_to_id, data, image_id, scale_rate=1.0):
    # img_shape h*w
    masks = {}  # for area
    segmentations = collections.defaultdict(list)  # for segmentation
    for shape in label_file.shapes:
        points = [[int(x * scale_rate) for x in row] for row in shape["points"]]
        label = shape["label"]
        group_id = shape.get("group_id")
        shape_type = shape.get("shape_type", "polygon")
        mask = labelme.utils.shape_to_mask(
            img_shape, points, shape_type
        )

        if group_id is None:
            group_id = uuid.uuid1()

        instance = (label, group_id)

        if instance in masks:
            masks[instance] = masks[instance] | mask
        else:
            masks[instance] = mask

        if shape_type == "rectangle":
            (x1, y1), (x2, y2) = points
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            points = [x1, y1, x2, y1, x2, y2, x1, y2]
        if shape_type == "circle":
            (x1, y1), (x2, y2) = points
            r = np.linalg.norm([x2 - x1, y2 - y1])
            # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
            # x: tolerance of the gap between the arc and the line segment
            n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
            i = np.arange(n_points_circle)
            x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
            y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
            points = np.stack((x, y), axis=1).flatten().tolist()
        else:
            points = np.asarray(points).flatten().tolist()

        segmentations[instance].append(points)
    segmentations = dict(segmentations)
    for instance, mask in masks.items():
        cls_name, group_id = instance
        if cls_name not in class_name_to_id:
            continue
        cls_id = class_name_to_id[cls_name]

        mask = np.asfortranarray(mask.astype(np.uint8))
        mask = pycocotools.mask.encode(mask)
        area = float(pycocotools.mask.area(mask))
        bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

        data["annotations"].append(
            dict(
                id=len(data["annotations"]),
                image_id=image_id,
                category_id=cls_id,
                segmentation=segmentations[instance],
                area=area,
                bbox=bbox,
                iscrowd=0,
            )
        )
    return masks


def gen_coco_dateset(indices, label_files, base_dir,
                     data, class_name_to_id, out_ann_file,
                     noviz, visual_path, scale_rate=1.0):
    for image_id, index in enumerate(indices):
        filename = label_files[index]
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]
        img_filename = base + ".jpg"
        out_img_file = osp.join(base_dir, img_filename)

        img_pil = labelme.utils.img_data_to_pil(label_file.imageData)
        (width, height) = (int(img_pil.width * scale_rate), int(img_pil.height * scale_rate))
        img_pil = img_pil.resize((width, height))
        img = np.array(img_pil)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )
        masks = points_to_masks(img.shape[:2], label_file,
                                class_name_to_id, data, image_id, scale_rate)

        if not noviz:
            vizimg_to_file(class_name_to_id, img, img_filename, masks, visual_path)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


def vizimg_to_file(class_name_to_id, img, img_filename, masks, visual_path):
    viz = img
    if masks:
        labels, captions, masks = zip(
            *[
                (class_name_to_id[cnm], cnm, msk)
                for (cnm, gid), msk in masks.items()
                if cnm in class_name_to_id
            ]
        )
        viz = imgviz.instances2rgb(
            image=img,
            labels=labels,
            masks=masks,
            captions=captions,
            font_size=15,
            line_width=2,
        )
    out_viz_file = osp.join(
        visual_path, img_filename
    )
    imgviz.io.imsave(out_viz_file, viz)


def init_dirs(args):
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = osp.join(args.output_dir, "train")
    os.makedirs(train_path, exist_ok=True)
    val_path = osp.join(args.output_dir, "val")
    os.makedirs(val_path, exist_ok=True)
    train_visual_path = osp.join(args.output_dir, "Visualization-train")
    val_visual_path = osp.join(args.output_dir, "Visualization-val")
    if not args.noviz:
        os.makedirs(train_visual_path, exist_ok=True)
        os.makedirs(val_visual_path, exist_ok=True)
    print("Creating dataset:", args.output_dir)
    return train_path, val_path, train_visual_path, val_visual_path


def normal_generate(args):
    """
    正常生成
    """
    # if osp.exists(args.output_dir):
    #     print("Output directory already exists:", args.output_dir)
    #     sys.exit(1)
    train_path, val_path, train_visual_path, val_visual_path = init_dirs(args)
    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[
            dict(
                url=None,
                id=0,
                name=None,
            )
        ],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(
                supercategory=None,
                id=class_id,
                name=class_name,
            )
        )

    train_ann_file = osp.join(args.output_dir, "instances_train.json")
    val_ann_file = osp.join(args.output_dir, "instances_val.json")
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))

    # 划分 train 和 val
    split_rate = args.split_rate
    indices = list(range(len(label_files)))
    random.shuffle(indices)
    train_size = int((1.0 - split_rate) * len(label_files))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    print('启动2个进程分别执行')
    pool = multiprocessing.Pool()
    pool.apply_async(gen_coco_dateset, args=(
        train_indices, label_files,
        train_path, copy.deepcopy(data),
        class_name_to_id, train_ann_file,
        args.noviz, train_visual_path, args.scale_rate
    ))
    pool.apply_async(gen_coco_dateset, args=(
        val_indices, label_files,
        val_path, copy.deepcopy(data),
        class_name_to_id, val_ann_file,
        args.noviz, val_visual_path, args.scale_rate
    ))
    pool.close()
    pool.join()
    print('所有进程执行完毕')


def update_one(data, filename, base, img_filename, class_name_to_id, noviz, visual_path, scale_rate):
    """
    处理单个更新
    Args:
        data : 最后要输出json数据
        filename : 新的labelme文件
        base: 基础文件名
        img_filename: 对应的图片相当路径
        class_name_to_id: 标签转index
        noviz : 不验证
        visual_path: 验证目录
        scale_rate： 缩放比例
    Returns:
        处理后的data
    Raises:
        Exception 不能通过图片文件名找到image_id
    """
    images = data['images']
    image_filename = f'{base}.jpg'
    image_id = -1
    img_height = 0
    img_width = 0
    for item in images:
        if image_filename == osp.basename(item['file_name']):
            image_id = item['id']
            img_height = item['height']
            img_width = item['width']
            break
    if image_id == -1:
        raise Exception(f"can't find {image_filename} in data")
    label_file = labelme.LabelFile(filename=filename)
    # 删除所有 annotations 中关于此图的所有标记信息
    data['annotations'] = [item for item in data['annotations'] if item['image_id'] != image_id]
    masks = points_to_masks((img_height, img_width), label_file,
                            class_name_to_id, data, image_id, scale_rate)
    if not noviz:
        img = np.array(PIL.Image.open(img_filename))
        vizimg_to_file(class_name_to_id, img, image_filename, masks, visual_path)
    return data


def update_datasets(args):
    """
    更新数据集
    """
    if not osp.exists(args.output_dir):
        print("Output directory not exists:", args.output_dir)
        sys.exit(1)
    if not osp.exists(args.input_dir):
        print("Input directory not exists:", args.input_dir)
        sys.exit(1)
    if not osp.exists(args.update_dir):
        print("Update directory not exists:", args.update_dir)
        sys.exit(1)
    train_path, val_path, train_visual_path, val_visual_path = init_dirs(args)
    train_ann_file = osp.join(args.output_dir, "instances_train.json")
    with open(train_ann_file, 'r') as f:
        train_data = json.load(f)
    class_name_to_id = {}
    for item in train_data['categories']:
        class_name_to_id[item['name']] = item['id']
    val_ann_file = osp.join(args.output_dir, "instances_val.json")
    with open(val_ann_file, 'r') as f:
        val_data = json.load(f)
    # 需要更新的所有文件
    label_files = glob.glob(osp.join(args.update_dir, "*.json"))
    labeled_files = glob.glob(osp.join(args.input_dir, "*.json"))
    train_files = glob.glob(osp.join(train_path, "*.jpg"))
    val_files = glob.glob(osp.join(val_path, "*.jpg"))
    cover_jsons = []
    # 遍历所有文件，判断是否有更新
    for i, filename in enumerate(label_files):
        # 有更新的放入对应的训练集或验证集
        base_filename = osp.basename(filename)
        base = osp.splitext(base_filename)[0]
        print(f'[{i}/{len(label_files)}]: {base_filename}')
        labeled_filename = osp.join(args.input_dir, base_filename)
        if labeled_filename in labeled_files:
            # 读取第一个 JSON 文件
            with open(filename, 'r') as f1:
                new_data = json.load(f1)
            # 读取第二个 JSON 文件
            with open(labeled_filename, 'r') as f2:
                old_data = json.load(f2)
            if new_data == old_data:
                print(f"no change: {base_filename}")
                continue
            else:
                print(f"{base_filename} changed")
                # 判断在验证集还是训练集
                print("Generating dataset from:", filename)
                image_filename = osp.join(train_path, base + ".jpg")
                if image_filename in train_files:
                    # 在训练集更新
                    update_one(train_data, filename, base, image_filename,
                               class_name_to_id, args.noviz, train_visual_path, args.scale_rate)
                image_filename = osp.join(val_path, base + ".jpg")
                if image_filename in val_files:
                    # 在验证集更新
                    update_one(val_data, filename, base, image_filename,
                               class_name_to_id, args.noviz, val_visual_path, args.scale_rate)
                cover_jsons.append((labeled_filename, new_data))
        else:
            # todo 新增
            raise Exception('todo append')
    # 重新排序 data['annotations']
    for i, item in enumerate(train_data['annotations']):
        item['id'] = i
    for i, item in enumerate(val_data['annotations']):
        item['id'] = i
    with open(train_ann_file, "w") as f:
        json.dump(train_data, f)
    with open(val_ann_file, "w") as f:
        json.dump(val_data, f)
    for labeled_filename, new_data in cover_jsons:
        with open(labeled_filename, 'w') as f:
            json.dump(new_data, f)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--split_rate", help="val file rate", default=0.1)
    parser.add_argument("--scale_rate", help="val file rate", type=float, default=1.0)
    parser.add_argument("--update_dir", help="update input annotated directory,"
                                             "It will check json with input_dir. If changed generate new",
                        type=str, default=None)
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()
    print("=" * 20 + os.getcwd() + "=" * 20)
    if args.update_dir is None:
        normal_generate(args)
    else:
        update_datasets(args)


if __name__ == "__main__":
    main()
