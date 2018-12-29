import postgresql
import numpy as np
import redis
import json
import time
import cv2
import tensorflow as tf
import facenet
from facenet import to_rgb, prewhiten, crop, flip
from scipy import spatial
import operator
import pickle


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


def face_distance(face_encodings, face_to_compare):
    if not np.isnan(face_encodings).any():
        dist = spatial.distance.sqeuclidean(face_encodings, face_to_compare)
        return dist
    else:
        return np.empty(0)


def get_centroids_from_bot():
    st = time.time()
    with postgresql.open('pq://postgres:postgres@10.100.10.6:5432/recognition') as db:
        result = db.query("SELECT name, embedding, distance FROM centers;")
        centers = {}
        for i in result:
            emb_array = [[]]
            emb_array[0] = list(map(float, i[1]))
            emb_array = np.asarray(emb_array)
            centers[i[0]] = (emb_array, i[2])

        print(time.time() - st)

    return centers


if __name__ == '__main__':
    # img = cv2.imread("/home/clara/dump/dockers/face_recognition/volumes/photo/etalon6/2018-12-26/12:05:56.103603.jpg")
    # recognizer = recognize(model="/home/clara/dump/dockers/face_recognition/volumes/20181022-081754")
    # next(recognizer)
    # face = (img.astype(np.float32) - 127.5) / 128.0
    # embedding = recognizer.send(face).copy()

    r = redis.StrictRedis(host='10.100.10.6', port=6379, db=0)

    st = time.time()
    centers = pickle.loads(r.get('centers'))
    name = "38899555"
    if name in centers:
        print("yes", name)
    # centers = {}
    # for i in result:
    #     centers[i] = (result[i][0], result[i][1])
    #     print(centers[i])


    # face_distances = {}
    # for name in centers:
    #     distance = face_distance(centers[name][0], embedding)
    #     face_distances[name] = distance
    #
    # min_dist_cluster = min(face_distances.items(), key=operator.itemgetter(1))[0]
    #
    # if face_distances[min_dist_cluster] <= centers[min_dist_cluster][1]:
    #     found_cluster = min_dist_cluster
    #     dist = face_distances[found_cluster]
    # else:
    #     found_cluster = "Unknown"
    #     dist = None
    #
    # print(found_cluster, dist)
    # print(time.time() - st)
