#!/usr/bin/env python3

import argparse
import json
import cv2
import os
import sys
import time
import numpy as np
from configobj import ConfigObj
from imutils.video import VideoStream
from rabbit_queues import RabbitQueue, QueueHandler
from rabbitmq import RabbitClass
import logging
import pika
import redis


def create_photo(ch, method, properties, body):
    try:
        msg = json.loads(body.decode('utf-8'))
    except pika.exceptions.ConnectionClosed:
        queue.create_connection()
        logger.debug("Restart photo push")
        msg = json.loads(body.decode('utf-8'))

    session_id = msg['session_id']
    service_id = msg['service_id']
    received_time = msg['time']
    logger.debug("SERVICE_ID: {}   SESSION_ID: {}".format(service_id, session_id))

    if service_id not in ['etalon6', 'etalon7', 'etalon8']:
        return

    logger.debug("Time1: {}".format(time.time() - received_time))

    for i in range(5):
        t0 = time.time()
        if r_cache.get(str(session_id)) is not None:
            return

        frame = service_dict[service_id].read()
        if frame is None:
            logger.error("Error frame")
            sys.exit(255)

        t1 = time.time()
        logger.debug("time2: {}".format(t1 - received_time))

        retval, frame_jpg = cv2.imencode(".jpg", frame)
        msg = {
            "t0": t0,
            "t1": t1,
            "face": frame_jpg.tobytes(),
            "service_id": service_id,
            "session_id": session_id
        }

        try:
            queue.send_message(msg, 'detect')
        except pika.exceptions.ConnectionClosed:
            queue.create_connection()
            logger.debug("Restart photo push")
            queue.send_message(msg, 'detect')

        time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Video stream capturing component for face detection and recognition service')
    parser.add_argument('--config', '-c',
                        dest='config', type=str,
                        default='/volumes/recognition.cfg',
                        help='Path to configuration file'
                        )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit("No such file or directory: %s" % args.config)

    r_cache = redis.StrictRedis(host='10.80.0.22', port=6379, db=0)
    config = ConfigObj(args.config)

    vs6 = VideoStream(src="rtsp://user:cVis_288@10.100.146.221/Streaming/Channels/1?tcp").start()
    vs7 = VideoStream(src="rtsp://admin:cVis_288@10.100.158.125/Streaming/Channels/101").start()
    vs8 = VideoStream(src="rtsp://admin:cVis_288@10.100.158.127/Streaming/Channels/1").start()

    service_dict = {'etalon6': vs6,
                    'etalon7': vs7,
                    'etalon8': vs8
                    }

    logger = logging.getLogger("photo_pushApp")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("/volumes/logs/push.log")  # create the logging file handler
    formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add handler to logger object

    queue_vad = RabbitQueue(args.config)
    queue_handler = QueueHandler()
    queue_handler.set_queue(queue_vad)
    queue_handler.setLevel(logging.INFO)
    queue_handler.setFormatter(formatter)
    logger.addHandler(queue_handler)

    if 'rabbitmq' not in config:
        sys.exit("Mandatory section missing: %s" % 'rabbitmq')

    queue = RabbitClass(args.config, logger)
    logger.debug("Start photo push")
    # create_photo()
    queue_vad.read_queue_with_direct_exchange(create_photo, 'kicks', 'kick_face')
