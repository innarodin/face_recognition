#!/usr/bin/python3.5

import argparse
import time
import sys
import cv2
import tensorflow as tf
import numpy as np
import pickle
import os
import facenet
from facenet import to_rgb, prewhiten, crop, flip
from configobj import ConfigObj
from rabbitmq import RabbitClass
import logging
from rabbit_queues import RabbitQueue, QueueHandler


def recognize(model):
    g_recognition = tf.Graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with g_recognition.as_default():
        with tf.Session(graph=g_recognition, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # Load the model
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = g_recognition.get_tensor_by_name("input:0")
            embeddings = g_recognition.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = g_recognition.get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            emb_array = np.zeros((1, embedding_size))
            images = np.zeros((1, 160, 160, 3))

            img = None

            while True:
                if img is None:
                    img = yield None

                img = crop(img, False, 160)
                img = flip(img, False)

                images[0, :, :, :] = img

                feed_dict = {images_placeholder: images,
                             phase_train_placeholder: False}
                start_index = 0
                end_index = 1
                emb_array[start_index:end_index, :] = sess.run(
                    embeddings, feed_dict=feed_dict)

                img = yield emb_array


def identification(channel, method, props, body):
    # t5 = time.time()
    data = pickle.loads(body)
    path = data['path']
    face_string = data['face']
    t0 = float(data['t0'])
    t1 = float(data['t1'])
    t2 = float(data['t2'])
    t3 = float(data['t3'])
    service_id = data['service_id']
    session_id = data['session_id']

    t4 = time.time()

    face = np.asarray(bytearray(face_string), dtype="uint8")
    face = cv2.imdecode(face, cv2.IMREAD_COLOR)
    face = (face.astype(np.float32) - 127.5) / 128.0

    emb_array = recognizer.send(face).copy()
    if emb_array is None:
        return

    t5 = time.time()

    msg = {
        'service_id': service_id,
        'embeddings': emb_array,
        'path': path,
        't0': t0,
        't1': t1,
        't2': t2,
        't3': t3,
        't4': t4,
        't5': t5,
        'session_id': session_id
    }

    queue.send_message(msg, 'classify')


if __name__ == '__main__':
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

    for config_section in ('rabbitmq', 'ident'):
        if config_section not in config:
            sys.exit("Mandatory section missing: %s" % config_section)

    recognizer = recognize(model=config['ident']['model_path'])
    next(recognizer)

    logger = logging.getLogger("identApp")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("/volumes/logs/ident.log")  # create the logging file handler
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
    queue.create_queue('ident')
    logger.debug("Start ident")
    queue.read_queue(identification)
