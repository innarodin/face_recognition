#!/usr/bin/python3.5

import argparse
import time
import sys
from scipy import spatial
import numpy as np
import pickle
import operator
import os
import postgresql
from configobj import ConfigObj
import redis
from rabbitmq import RabbitClass
import json
import logging
from rabbit_queues import RabbitQueue, QueueHandler
from redisworks import Root


def face_distance(face_encodings, face_to_compare):
    if not np.isnan(face_encodings).any():
        dist = spatial.distance.sqeuclidean(face_encodings, face_to_compare)
        return dist
    else:
        return np.empty(0)


def get_centroids_from_db(address_name):
    model = '20181022-081754'
    with postgresql.open('pq://postgres:postgres@10.80.0.22:5432/recognition') as db:
        result = db.query("SELECT c.name, c.embedding, c.distance FROM centroids c JOIN addresses a on "
                          "c.address_id=a.id WHERE a.name='{}' and c.model='{}' and c.creation_date=("
                          "SELECT MAX(c.creation_date) FROM centroids c JOIN addresses a ON c.address_id=a.id "
                          "WHERE a.name='{}')".format(address_name, model, address_name))
        centers = {}
        for i in result:
            emb_array = [[]]
            emb_array[0] = list(map(float, i[1]))
            emb_array = np.asarray(emb_array)
            centers[i[0]] = (emb_array, i[2])

    return centers


def get_centroids_from_bot():
    with postgresql.open('pq://postgres:postgres@10.80.0.22:5432/recognition') as db:
        result = db.query("SELECT name, embedding, distance FROM centers;")
        centers = {}
        for i in result:
            emb_array = [[]]
            emb_array[0] = list(map(float, i[1]))
            emb_array = np.asarray(emb_array)
            centers[i[0]] = (emb_array, i[2])

    return centers


def save_emb_to_db(found_cluster, embedding):
    with postgresql.open('pq://postgres:postgres@10.80.0.22:5432/recognition') as db:
        ins = db.prepare("INSERT INTO embeddings (name, embedding) VALUES ($1, $2);")
        ins(found_cluster, embedding[0])


def classification(channel, method, props, body):
    data = pickle.loads(body)
    path = data['path']
    service_id = data['service_id']
    embedding = data['embeddings']
    t0 = float(data['t0'])
    t1 = float(data['t1'])
    t2 = float(data['t2'])
    t3 = float(data['t3'])
    t4 = float(data['t4'])
    t5 = float(data['t5'])
    session_id = data['session_id']

    t6 = time.time()

    if r.get(str(session_id)) is not None:
        return

    centers = get_centroids_from_bot()

    face_distances = {}
    for name in centers:
        distance = face_distance(centers[name][0], embedding)
        face_distances[name] = distance

    min_dist_cluster = min(face_distances.items(), key=operator.itemgetter(1))[0]

    if face_distances[min_dist_cluster] <= centers[min_dist_cluster][1]:
        found_cluster = min_dist_cluster
        dist = face_distances[found_cluster]

        save_emb_to_db(found_cluster, embedding)

    else:
        found_cluster = "Unknown"
        dist = None

    msg = {
        'predicts': (found_cluster, dist),
        'service_id': service_id,
        'session_id': session_id,
        'type': 'face'
    }
    queue.send_message(msg, 'voice_face')
    logger.debug("sent")

    msg = {
        "service_id": service_id,
        "session_id": session_id,
        "status": "NAME: {}, DIST: {}".format(found_cluster, dist)
    }
    logger.info(msg)

    t7 = time.time()
    msg = {
        'name': found_cluster,
        'confidence': dist,
        'service_id': service_id,
        'path': path,
        'session_id': session_id,
        't0': t0,
        't1': t1,
        't2': t2,
        't3': t3,
        't4': t4,
        't5': t5,
        't6': t6,
        't7': t7
    }
    queue.send_message(msg, 'final')

    logger.debug("redis {} {}".format(session_id, r.get(str(session_id))))


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

    for config_section in ('rabbitmq',):
        if config_section not in config:
            sys.exit("Mandatory section missing: %s" % config_section)

    logger = logging.getLogger("classifyApp")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("/volumes/logs/classify.log")  # create the logging file handler
    formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add handler to logger object

    queue = RabbitQueue("/app/queues.cfg")
    queue_handler = QueueHandler()
    queue_handler.set_queue(queue)
    queue_handler.setLevel(logging.INFO)
    queue_handler.setFormatter(formatter)
    logger.addHandler(queue_handler)

    logger.debug("Start classify")

    people_dict = {}
    centers_dict = {}

    root = Root(host='10.80.0.22', port=6379, db=0)
    r = redis.StrictRedis(host='10.80.0.22', port=6379, db=0)

    queue = RabbitClass(args.config, logger)
    queue.create_queue('classify')
    queue.read_queue(classification)
