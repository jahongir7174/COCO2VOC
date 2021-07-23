import collections
import glob
import json
import os

import numpy
import tqdm
from PIL import Image, ExifTags
from pascal_voc_writer import Writer

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def coco91_to_coco80_class():
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
         None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         None, 73, 74, 75, 76, 77, 78, 79, None]
    return x


def via2coco():
    classes = ('Bonnet/Hood',
               'Front Fog Lamp',
               'Rear Door',
               'Rear bumper',
               'Staff',
               'Rediator Grille',
               'Tail Lamp',
               'Front Fender',
               'Front Door',
               'Roof',
               'Front Wheel',
               'Rear Wheel',
               'Front bumper',
               'Rear Fog Lamp',
               'Trunk Door',
               'Head Lamp',
               'Rear Fender',
               'Side View Mirror')

    via_data = json.load(open('../Dataset/PepCar/annotation/via_v.json'))

    img_id = 0
    box_id = 0
    images = []
    categories = []
    annotations = []
    for key in via_data:
        file_name = via_data[key]["filename"]

        img_id += 1
        img = Image.open(os.path.join("../Dataset/PepCar/images/val/", file_name))
        img.verify()
        size = img.size
        try:
            rotation = dict(img.getexif().items())[orientation]
            if rotation == 6:
                size = (size[1], size[0])
            elif rotation == 8:
                size = (size[1], size[0])
        except KeyError:
            pass
        w, h = size

        images.append({'file_name': file_name, 'id': img_id, 'height': h, 'width': w})

        for region in via_data[key]["regions"]:
            box_id += 1
            points_x = region["shape_attributes"]["all_points_x"]
            points_y = region["shape_attributes"]["all_points_y"]
            x_min, x_max = min(points_x), max(points_x)
            y_min, y_max = min(points_y), max(points_y)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            mask = []
            left_lines = []
            right_lines = []
            max_x = max(points_x)
            index = points_x.index(max_x)
            for i, point_x in enumerate(points_x):
                if i != index:
                    if points_y[i] < points_y[index]:
                        angle = abs(point_x - points_x[index]) / abs(points_y[i] - points_y[index] + 1e-7)
                        left_lines.append([angle, point_x, points_y[i]])
                    else:
                        angle = abs(point_x - points_x[index]) / abs(points_y[i] - points_y[index] + 1e-7)
                        right_lines.append([angle, point_x, points_y[i]])

            left_lines.sort()
            right_lines.sort(reverse=True)

            for left_line in left_lines:
                mask.append(left_line[1])
                mask.append(left_line[2])
            for right_line in right_lines:
                mask.append(right_line[1])
                mask.append(right_line[2])
            mask.append(points_x[index])
            mask.append(points_y[index])
            annotations.append({'id': box_id,
                                'bbox': bbox,
                                'iscrowd': 0,
                                'image_id': img_id,
                                'segmentation': [mask],
                                'area': bbox[2] * bbox[3],
                                'category_id': classes.index(region["region_attributes"]["carpart"]) + 1})
    for i, key in enumerate(classes):
        categories.append({'supercategory': key, 'id': i + 1, 'name': key})

    json_data = json.dumps({'images': images, 'categories': categories, 'annotations': annotations})
    with open('coco_v.json', 'w') as f:
        f.write(json_data)


def voc2coco():
    classes = ('back_hand_break',
               'back_handle',
               'back_light',
               'back_mudguard',
               'back_pedal',
               'back_reflector',
               'back_wheel',
               'bell',
               'chain',
               'dress_guard',
               'dynamo',
               'front_handbreak',
               'front_handle',
               'front_light',
               'front_mudguard',
               'front_pedal',
               'front_wheel',
               'gear_case',
               'kickstand',
               'lock',
               'saddle',
               'steer')
    file_names = [file_name for file_name in glob.glob("../Dataset/VIP2021/images/val/*.jpg", recursive=True)]
    file_names = list(sorted(file_names))
    output_json_dict = {"images": [],
                        "type": "instances",
                        "annotations": [],
                        "categories": []}
    box_id = 1
    print('Start converting !')
    for file_name in file_names:
        image = Image.open(file_name)
        width, height = image.size
        img_id = os.path.basename(file_name).split('.')[0]
        img_info = {'file_name': os.path.basename(file_name),
                    'height': height,
                    'width': width,
                    'id': img_id}
        output_json_dict['images'].append(img_info)
        with open(file_name.replace('images', 'labels').replace('jpg', 'txt')) as f:
            for box in f.readlines():
                category_id, x_min, y_min, x_max, y_max = list(map(int, box.rstrip().split()))
                box_w = x_max - x_min
                box_h = y_max - y_min
                ann = {'area': box_w * box_h,
                       'iscrowd': 0,
                       'bbox': [x_min, y_min, box_w, box_h],
                       'category_id': category_id,
                       'ignore': 0,
                       'segmentation': []}
                ann.update({'image_id': img_id, 'id': box_id})
                output_json_dict['annotations'].append(ann)
                box_id += 1

    for label_id, label in enumerate(classes):
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open('val.json', 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def coco2voc(json_dir='../Dataset/COCO/annotations/'):
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
    coco2voc()
