import json
import select
import numpy as np
import postgresql
import psycopg2
import psycopg2.extensions
from scipy import spatial
import sys
import logging


def get_embs_by_name(name):
    with postgresql.open('pq://postgres:postgres@10.80.0.22:5432/recognition') as db:
        result = db.query("SELECT embedding FROM embeddings WHERE name='{}';".format(name))

    embs = []

    for i in result:
        emb_array = [[]]
        emb_array[0] = list(map(float, i[0]))
        emb_array = np.asarray(emb_array)
        embs.append(emb_array)

    return embs


def face_distance(face_encodings, face_to_compare):
    if not np.isnan(face_encodings).any():
        dist = spatial.distance.sqeuclidean(face_encodings, face_to_compare)
        return dist
    else:
        return np.empty(0)


def update_centroid(name):
    embs = get_embs_by_name(name)
    center = np.mean(embs, axis=0)

    with postgresql.open('pq://postgres:postgres@10.80.0.22:5432/recognition') as db:
        result = db.query("SELECT name FROM centers;")
        is_updated = 0
        for res in result:
            if name in res:
                upd = db.prepare("UPDATE centers SET embedding=$1 WHERE name=$2;")
                upd(center[0], name)
                is_updated = 1

        if is_updated == 0:
            ins = db.prepare("INSERT INTO centers (name, embedding, distance) VALUES ($1, $2, $3);")

            dists = []
            for emb in embs:
                dists.append(face_distance(center, emb))

            dist = max(dists)
            if dist < 0.3:
                dist = 0.3
            elif dist > 0.8:
                dist = 0.8

            ins(name, center[0], dist)


if __name__ == '__main__':
    conn = psycopg2.connect(dbname='recognition', user='postgres', password='postgres', host='10.80.0.22')
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    curs = conn.cursor()
    curs.execute("LISTEN embeddings;")

    logger = logging.getLogger("Check_inserts")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("/volumes/logs/check_inserts.log")  # create the logging file handler
    formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add handler to logger object

    inserts = {}

    logger.info("Waiting for notifications on channel 'embeddings'")

    while True:
        if select.select([conn], [], [], 5) != ([], [], []):
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                logger.info("Got NOTIFY: {} {} {}".format(notify.pid, notify.channel, notify.payload))

                name = json.loads(notify.payload)["name"]
                try:
                    inserts[name] += 1
                except KeyError:
                    inserts[name] = 1

                if inserts[name] % 7 == 0:
                    logger.info("Update {}".format(name))
                    sys.stdout.flush()
                    update_centroid(name)
