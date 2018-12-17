import argparse
import sys
import time
import cv2
import pickle
import os
import numpy as np
import pytz
import datetime
from configobj import ConfigObj
#import mtcnn_detector
from rabbitmq import RabbitClass
import tensorflow as tf
from detect_face import detect_face, create_mtcnn
import logging
import redis
from rabbit_queues import RabbitQueue, QueueHandler
import random


def make_photo(path, face, session_id):
    data = str(datetime.datetime.now().date())
    if not os.path.exists(path):
        os.mkdir(path)

    if data not in os.listdir(path):
        os.mkdir(os.path.join(path, data))

    new_path = os.path.join(path, data)
    cur_time = str(datetime.datetime.now().time())
    img_name = cur_time + '.jpg'
    path_to_write = str(os.path.join(new_path, img_name))
    logger.debug("{} {}".format(path_to_write, session_id))
    cv2.imwrite(path_to_write, face)
    return path_to_write.split("photo/")[1]


def detect(channel, method_frame, header_frame, body):
    storage_path = None

    t2 = time.time()

    if 'storage' in detector_config:
        storage_path = detector_config['storage']

    data = pickle.loads(body)
    face_string = data['face']
    t0 = data['t0']
    t1 = data['t1']
    service_id = data['service_id']
    session_id = data['session_id']

    if r.get(str(session_id)) is not None:
        return

    image = np.asarray(bytearray(face_string), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # bounding_boxes = None
    # try:
    #     bounding_boxes, feature_points = detector.detect_face(image)
    # except TypeError:
    #     pass

    bounding_boxes, feature_points = detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    # tmp_time = time.time()
    # logger.debug("delay detect: {}".format(tmp_time - t2))

    if len(bounding_boxes) == 0:
    # if bounding_boxes is None:
        if r.get("detect" + str(session_id)) is None:
            r.set("detect" + str(session_id), 1)
            logger.debug("No face {}".format(session_id))
        else:
            r.incr("detect" + str(session_id))
            logger.debug("No face {}".format(session_id))

        if r.get("detect" + str(session_id)) == b'5':
            msg = {
                "service_id": service_id,
                "session_id": session_id,
                "status": "No face"
            }
            logger.info(msg)
            r.delete("detect" + str(session_id))
        return

    max_box_area = 0
    box_id = None
    x = y = w = h = p = None
    for i, (x1, y1, x2, y2, p1) in enumerate(bounding_boxes):
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < max_box_area: continue
        max_box_area = box_area
        box_id = i
        x = x1
        y = y1
        w = x2
        h = y2
        p = p1

    # do nothing if no face detected
    if box_id is None:
        return

    height = abs(h - y)
    width = abs(w - x)
    difference = abs(height - width)
    if height > width:
        face = image[int(y):int(h), int(x - difference // 2):int(w + difference // 2)]
    else:
        face = image[int(y - difference // 2):int(h + difference // 2), int(x):int(w)]

    if not face.all() and p >= 0.98:
        if np.var(cv2.Laplacian(face, cv2.CV_64F)) < 80:
            return
        face = cv2.resize(face, (160, 160))
        path = make_photo(os.path.join(storage_path, service_id), face, session_id)
        retval, face_string = cv2.imencode(".jpg", face)

        t3 = time.time()

        msg = {
            'face': face_string.tostring(),
            'path': os.path.join(path),
            't0': t0,
            't1': t1,
            't2': t2,
            't3': t3,
            'service_id': service_id,
            "session_id": session_id
        }

        queue.send_message(msg, 'ident')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Video stream capturing component for face detection and recognition service')

    parser.add_argument('--config', '-c',
                        dest='config', type=str,
                        default="/app/recognition.cfg",
                        help='Path to configuration file'
                        )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit("No such file or directory: %s" % args.config)

    config = ConfigObj(args.config)

    for config_section in ('rabbitmq', 'detector'):
        if config_section not in config:
            sys.exit("Mandatory section missing: %s" % config_section)

    # get configuration parameters
    detector_config = config['detector']
    # detector = mtcnn_detector.MtcnnDetector()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    pnet, rnet, onet = create_mtcnn(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)), model_path="/volumes/model")
    minsize = 80  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    r = redis.StrictRedis(host='10.80.0.22', port=6379, db=0)

    logger = logging.getLogger("detectorApp")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("/volumes/logs/detector.log")  # create the logging file handler
    formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add handler to logger object

    queue = RabbitQueue("/app/queues.cfg")
    queue_handler = QueueHandler()
    queue_handler.set_queue(queue)
    queue_handler.setLevel(logging.INFO)
    queue_handler.setFormatter(formatter)
    logger.addHandler(queue_handler)

    queue = RabbitClass(args.config, logger)
    queue.create_queue('detect')
    logger.debug("Start detector")
    queue.read_queue(detect)
