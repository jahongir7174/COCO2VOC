import collections
import glob
import json
import os

import numpy
import tqdm
from PIL import Image
from pascal_voc_writer import Writer


def coco91_to_coco80_class():
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
         None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         None, 73, 74, 75, 76, 77, 78, 79, None]
    return x


def convert_coco_json(json_dir='../Dataset/COCO/annotations/'):
    jsons = glob.glob(json_dir + '*.json')
    coco80 = coco91_to_coco80_class()
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush']
    for json_file in sorted(jsons):
        if 'instances_' not in json_file:
            continue
        with open(json_file) as f:
            data = json.load(f)
        print(json_file)
        results = collections.defaultdict(list)
        folder_name = os.path.basename(json_file).replace('instances_', '').split('.')[0]
        images = {'%g' % x['id']: x for x in data['images']}
        for x in data['annotations']:
            if x['iscrowd']:
                continue

            file_name = images['%g' % x['image_id']]['file_name']
            box = numpy.array(x['bbox'], dtype=numpy.float64)
            x_min, y_min, x_max, y_max = box[0], box[1], box[0] + box[2], box[1] + box[3]
            results[file_name].append([coco80[x['category_id'] - 1], x_min, y_min, x_max, y_max])
        with open(f'{base_dir}/{folder_name}.txt', 'w') as f:
            for key, value in tqdm.tqdm(results.items()):
                if os.path.exists(f'{base_dir}/{folder_name}/{key}'):
                    f.write(f'{key[:-4]}\n')
                    image = Image.open(f'{base_dir}/{folder_name}/' + key)
                    image.verify()
                    w, h = image.size
                    writer = Writer(f'{key[:-4]}.xml', w, h)
                    for v in value:
                        label, x_min, y_min, x_max, y_max = v
                        writer.addObject(names[label], int(x_min), int(y_min), int(x_max), int(y_max))
                    writer.save(f'{base_dir}/labels/{key[:-4]}.xml')


if __name__ == '__main__':
    base_dir = '../Dataset/COCO'
    if not os.path.exists(f'{base_dir}/labels'):
        os.makedirs(f'{base_dir}/labels')
    convert_coco_json()
