from pycocotools.cocoeval import COCOeval 
import keras 
import numpy as np 
import json

from model import efficientdet

#from keras_retinanet.preprocessing.coco import CocoGenerator
from generators.coco import CocoGenerator
from utils.anchors import anchors_for_shape, anchor_targets_bbox
import cv2

def preprocess_image(image, image_size=512):
        image_height, image_width = image.shape[:2]
        if image_height > image_width:
            scale = image_size / image_height
            resized_height = image_size
            resized_width = int(image_width * scale)
        else:
            scale = image_size / image_width
            resized_height = int(image_height * scale)
            resized_width = image_size
        image = cv2.resize(image, (resized_width, resized_height))
        new_image = np.ones((image_size, image_size, 3), dtype=np.float32) * 128.
        offset_h = (image_size - resized_height) // 2
        offset_w = (image_size - resized_width) // 2
        new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image.astype(np.float32)
        new_image /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        new_image[..., 0] -= mean[0]
        new_image[..., 1] -= mean[1]
        new_image[..., 2] -= mean[2]
        new_image[..., 0] /= std[0]
        new_image[..., 1] /= std[1]
        new_image[..., 2] /= std[2]
        return new_image, scale, offset_h, offset_w

def generate_resolutions():
    return [512, 640, 768, 896, 1024, 1280, 1408]

def detect_on_frame(image, prediction_model, anchors, score_threshold=0.5, max_detections=100):
    h, w = image.shape[:2]
    image, scale, offset_h, offset_w = preprocess_image(image)

    # time to detect
    # run network
    boxes, scores, labels = prediction_model.predict_on_batch([np.expand_dims(image, axis=0),
                                                         np.expand_dims(anchors, axis=0)])
    boxes[..., [0, 2]] = boxes[..., [0, 2]] - offset_w
    boxes[..., [1, 3]] = boxes[..., [1, 3]] - offset_h
    boxes /= scale
    boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w - 1)
    boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h - 1)
    boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, w - 1)
    boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, h - 1)

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:max_detections]

    # select detections
    # (n, 4)
    image_boxes = boxes[0, indices[scores_sort], :]
    # (n, )
    image_scores = scores[scores_sort]
    # (n, )
    image_labels = labels[0, indices[scores_sort]]
    # (n, 6)
    detections = np.concatenate(
        [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
    return detections


model, prediction_model = efficientdet(phi=0, num_classes=80)
prediction_model.load_weights('efficientdet.h5', by_name=True)


gen = CocoGenerator(data_dir='/home/floris/data/COCO', set_name='val2017')
generator = gen
threshold = 0.05


phi=0
resolutions = generate_resolutions()
threshold = 0.05
max_detections = 100
anchors = anchors_for_shape((resolutions[phi], resolutions[phi]))

image_ids = [] 
results = []
for index in range(gen.size()):
    print(index)
    image = generator.load_image(index)
    
    detections = detect_on_frame(image, prediction_model, anchors, threshold, max_detections)
    #print(detections)
    #image = generator.preprocess_image(image)
    #image, scale = generator.resize_image(image)
    #boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    #boxes /= scale

    



    detections[ :, 2] -= detections[ :, 0]
    detections[ :, 3] -= detections[ :, 1]
    # compute predicted labels and scores
    for box in detections:
        score = box[4]
        label = box[5]
        box = box[:4]
        # scores are sorted, so we can break
        if score < threshold:
            break
    
        # append detection for each positively labeled class
        image_result = {
           'image_id'    : generator.image_ids[index],
           'category_id' : generator.label_to_coco_label(label),
           'score'       : float(score),
           'bbox'        : box.tolist(),
        }
        # append detection to results
        results.append(image_result)
    
        # append image to list of processed images
        image_ids.append(generator.image_ids[index])
if not len(results):
    print("NO RESULTS")
else:
    # write output
    json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
    json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)
  
    # load results in COCO evaluation tool
    coco_true = generator.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(generator.set_name))
    
    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
