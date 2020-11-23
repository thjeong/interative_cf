import numpy as np
import pandas as pd
import tensorflow as tf
import time, sys, datetime, copy, math, os
from threading import Thread
from multiprocessing import Process
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
log = logging.getLogger('root')

class cf_similarity(object):
    def __init__(self, mat1, mat2, sim_col: str, ftr_col='FTR', val_col='RATE', log_level=logging.WARN):

        ###
            1) input matrix 1 : contains users(items) to find similar users(items)
            2) input matrix 2 : contains users(items) to be similar users(items) in matrix 1
            3) sim_col : column name of users/items in matrix1 and 2
            4) ftr_col : column name of items/users in matrix1 and 2
            5) val_col : column name of value for ftr_col (ex. View or usage or buy count)
            * Items are usually goods / videos / coupon, etc. Users are a users to use items.

            ex) mat1 x mat2 = sim_mat
               [sim_col x ftr_col] x [ftr_col x sim_col] = [sim_col x sim_col]
               * columns names should exist on both matrices in the same manner.

            usage1) In the case of user based similarity,
                items are used as a feature set to calculate similarity for each users.
                [sim_col, ftr_col, val_col] = ['CLNN', 'FTR', 'RATE']

            usage2) In the case of item based similarity,
                users are used as a feature set to calculate similarity for each items(stores/videos).
                [sim_col, ftr_col, val_col] = ['ITEM', 'FTR', 'RATE']

            default) Item based similarity: column names are supposed to be 'CLNN', 'FTR', 'RATE'
                If it's a case of user based similarity, 'ITEM', 'FTR', 'RATE'

        ###
        log.setLevel(log_level)
        log.info('cf init data loading ...')


        self.SIM_COLNAME = sim_col

        start_time = time.time()
        self.mat1 = mat1[[sim_col, ftr_col, val_col]]
        self.mat2 = mat2[[sim_col, ftr_col, val_col]]

        self.mat1.columns = ['TGT', 'FTR', 'RATE']
        self.mat2.columns = ['TGT', 'FTR', 'RATE']

        ftrs = pd.unique(pd.concat([self.mat1, self.mat2])['FTR'])
        ftrs_map = pd.DataFrame({
            'FTR': ftrs,
            'FTR_IDX': range(len(ftrs))
        })

        mat1_targets = pd.unique(self.mat1['TGT'])
        mat1_target_map = pd.DataFrame({
            'TGT': mat1_targets,
            'MAT1_TGT_IDX': range(len(mat1_targets))
        })

        mat2_targets = pd.unique(self.mat2['TGT'])
        mat2_target_map = pd.DataFrame({
            'TGT': mat2_targets,
            'MAT2_TGT_IDX': range(len(mat2_targets))
        })

        self.ftrs_count = len(ftrs)
        self.mat1_targets_count = len(mat1_targets)
        self.mat2_targets_count = len(mat2_targets)
        log.info('# of targets in 1st matrix: %d' % self.mat1_targets_count)
        log.info('# of features: %d' % self.ftrs_count)
        log.info('# of references in 2nd matrix: %d' % self.mat2_targets_count)

        log.info('indexing 1st matrix ...')
        self.mat1 = pd.merge(self.mat1, mat1_target_map, how='left')
        self.mat1 = pd.merge(self.mat1, ftrs_map)
        self.mat1 = self.mat1[['MAT1_TGT_IDX', 'TGT', 'FTR_IDX', 'RATE']]
        self.mat1 = self.mat1.sort_values(by=['MAT1_TGT_IDX', 'FTR_IDX'])

        log.info('indexing 2nd matrix ...')
        self.mat2 = pd.merge(self.mat2, mat2_target_map, how='left')
        self.mat2 = pd.merge(self.mat2, ftrs_map)
        self.mat2 = self.mat2[['MAT2_TGT_IDX', 'TGT', 'FTR_IDX', 'RATE']]
        self.mat2 = self.mat2.sort_values(by=['MAT2_TGT_IDX', 'FTR_IDX'])

        log.info('normalizing 1st matrix ...')
        # L2-Normalize
        distances = self.mat1.assign(DISTANCE=self.mat1['RATE'] ** 2).groupby('TGT')['DISTANCE'].sum().transform(
            math.sqrt)
        distances = distances.to_frame().reset_index(level=0)
        self.mat1 = pd.merge(self.mat1, distances, how='left')
        self.mat1 = self.mat1.assign(NORMED_RATE=self.mat1['RATE'] / self.mat1['DISTANCE'])

        log.info('normalizing 2nd matrix ...')
        distances = self.mat2.assign(DISTANCE=self.mat2['RATE'] ** 2).groupby('TGT')['DISTANCE'].sum().transform(
            math.sqrt)
        distances = distances.to_frame().reset_index(level=0)
        self.mat2 = pd.merge(self.mat2, distances, how='left')
        self.mat2 = self.mat2.assign(NORMED_RATE=self.mat2['RATE'] / self.mat2['DISTANCE'])

        log.info('building indices ...')
        self.target_map1 = mat1_target_map.set_index('MAT1_TGT_IDX')['TGT'].to_dict()
        self.target_map2 = mat2_target_map.set_index('MAT2_TGT_IDX')['TGT'].to_dict()

        log.info('setup time elapsed : %.2f secs' % (time.time() - start_time))

    def calc(self, k, filename_to_save, slicing1=2000, slicing2=50000, append=False, trace=False, mode='gpu', axis=0):
        calc_time = time.time()

        # reset previous graph
        tf.reset_default_graph()

        if not append and os.path.isfile(filename_to_save):
            os.remove(filename_to_save)

        if not os.path.isfile(filename_to_save) and axis == 0:
            with open(filename_to_save, 'a') as f:
                f.write('%s,SIM_%s,SCORE' % (self.SIM_COLNAME, self.SIM_COLNAME))

        if mode == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=28,
                inter_op_parallelism_threads=28)
        elif mode == 'gpu1':
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            session_conf = tf.ConfigProto()
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            session_conf = tf.ConfigProto()


        RANK_TO_SAVE = k
        SLICE1_SIZE = slicing1
        SLICE2_SIZE = slicing2
        FEATURE_SIZE = self.ftrs_count

        FIRST_ITERATION_COUNT = math.ceil(self.mat1_targets_count / SLICE1_SIZE)
        SECOND_ITERATION_COUNT = math.ceil(self.mat2_targets_count / SLICE2_SIZE)

        log.info('# of iteration in 1st matrix : %d' % FIRST_ITERATION_COUNT)
        log.info('# of iteration in 2nd matrix : %d' % SECOND_ITERATION_COUNT)
        self.mat1['MAT1_SLICE_IDX'] = self.mat1['MAT1_TGT_IDX'].map(lambda x: x % SLICE1_SIZE)
        self.mat2['MAT2_SLICE_IDX'] = self.mat2['MAT2_TGT_IDX'].map(lambda x: x % SLICE2_SIZE)
        self.mat1['MAT1_ITERATION'] = self.mat1['MAT1_TGT_IDX'].map(lambda x: x // SLICE1_SIZE)
        self.mat2['MAT2_ITERATION'] = self.mat2['MAT2_TGT_IDX'].map(lambda x: x // SLICE2_SIZE)

        MAT1_SHAPE = (SLICE1_SIZE, FEATURE_SIZE)
        MAT2_SHAPE = (SLICE2_SIZE, FEATURE_SIZE)

        writing_thread = None
        for idx1 in range(FIRST_ITERATION_COUNT):
            start1 = time.time()
            START_TARGET_IDX1 = idx1 * SLICE1_SIZE

            mat1_sample = self.mat1[self.mat1['MAT1_ITERATION'] == idx1]

            np_mat1_indices = np.array(mat1_sample[['MAT1_SLICE_IDX', 'FTR_IDX']], dtype=np.int64)
            np_mat1_values = np.array(mat1_sample['NORMED_RATE'], dtype=np.float64)

            log.info('STAGE %d mat1 %s prepared' % (idx1 + 1, str(MAT1_SHAPE)))

            ### tf setup
            sess = tf.Session(config=session_conf)
            tf_top_k = tf.placeholder(tf.int32, name="tf_top_k")
            tf_base_idx = tf.placeholder(tf.int32, name="tf_base_idx")

            with tf.variable_scope("mat1", reuse=tf.AUTO_REUSE):
                tf_mat1_dense_shape = tf.placeholder(tf.int64, [2])
                tf_mat1_indices = tf.placeholder(tf.int64, [None, None])
                tf_mat1_values = tf.placeholder(tf.float64, [None])
                tf_mat1_sparse = tf.sparse.SparseTensor(indices=tf_mat1_indices, values=tf_mat1_values,
                                                        dense_shape=tf_mat1_dense_shape)
                tf_mat1 = tf.get_variable("tf_mat1", shape=MAT1_SHAPE, dtype=tf.float64)
                assign_op = tf_mat1.assign(tf.sparse.to_dense(tf_mat1_sparse, default_value=0))

            with tf.variable_scope("mat2"):
                tf_mat2_dense_shape = tf.placeholder(tf.int64, [2])
                tf_mat2_indices = tf.placeholder(tf.int64, [None, None])
                tf_mat2_values = tf.placeholder(tf.float64, [None])
                tf_mat2_sparse = tf.sparse.SparseTensor(indices=tf_mat2_indices, values=tf_mat2_values,
                                                        dense_shape=tf_mat2_dense_shape)
                tf_mat2 = tf.sparse.to_dense(tf_mat2_sparse, default_value=0)

            with tf.variable_scope("mat3", reuse=tf.AUTO_REUSE):
                tf_prev_indices = tf.get_variable("tf_prev_indices", shape=[SLICE1_SIZE, RANK_TO_SAVE], dtype=tf.int32)
                tf_prev_values = tf.get_variable("tf_prev_values", shape=[SLICE1_SIZE, RANK_TO_SAVE], dtype=tf.float64)

            tf_result_dense = tf.matmul(tf_mat1,
                                        tf_mat2,
                                        transpose_a=False,
                                        transpose_b=True,
                                        a_is_sparse=True,
                                        b_is_sparse=True)

            tf_loop_result = tf.nn.top_k(tf_result_dense, k=tf_top_k, sorted=True)

            # add up current loop2 index
            tf_mat3_indices = tf.concat([tf.add(tf_loop_result.indices, tf_base_idx), tf_prev_indices], 1)
            tf_mat3_values = tf.concat([tf_loop_result.values, tf_prev_values], 1)

            tf_result = tf.nn.top_k(tf_mat3_values, k=tf_top_k, sorted=True)
            # tf_result_values = tf_result.values
            _idx = tf_result.indices
            _rows = tf.broadcast_to(tf.expand_dims(tf.range(tf.shape(tf_result.values)[0]), 1), tf.shape(_idx))
            _ex_idx = tf.concat((tf.expand_dims(_rows, 2), tf.expand_dims(_idx, 2)), axis=2)

            tf_result_indices = tf_prev_indices.assign(tf.gather_nd(tf_mat3_indices, _ex_idx))
            tf_result_values = tf_prev_values.assign(tf_result.values)

            # load mat1 (reuse in loop2)
            sess.run(tf.global_variables_initializer())
            sess.run(assign_op, feed_dict={
                tf_mat1_indices: np_mat1_indices,
                tf_mat1_values: np_mat1_values,
                tf_mat1_dense_shape: MAT1_SHAPE
            })


            sampling_time, tf_time = 0, 0
            ret_values, ret_indices = [[0.0] * RANK_TO_SAVE] * SLICE1_SIZE, [[0] * RANK_TO_SAVE] * SLICE1_SIZE

            sample2_thread = None
            # slice2 최초 sampling
            self.sample_mat2(0)
            LAST_ITERATION = SECOND_ITERATION_COUNT - 1
            for idx2 in range(SECOND_ITERATION_COUNT):

                # 샘플링은 별도 thread에서 처리 : GPU가 바쁠동안 병행해서 동작
                sampling_start = time.time()
                if sample2_thread is not None:
                    sample2_thread.join()
                np_mat2_indices = self.next_mat2_indices
                np_mat2_values = self.next_mat2_values
                if idx2 < LAST_ITERATION:
                    sample2_thread = Thread(target=self.sample_mat2, args=(idx2 + 1,))
                    sample2_thread.start()
                sampling_time += time.time() - sampling_start

                tf_start = time.time()
                START_TARGET_IDX2 = idx2 * SLICE2_SIZE
                dict_to_feed = {tf_mat2_indices: np_mat2_indices,
                                tf_mat2_values: np_mat2_values,
                                tf_mat2_dense_shape: MAT2_SHAPE,
                                tf_base_idx: START_TARGET_IDX2,
                                tf_top_k: RANK_TO_SAVE if RANK_TO_SAVE < SLICE2_SIZE else SLICE2_SIZE
                                }
                if idx2 == LAST_ITERATION:
                    ret_values, ret_indices = sess.run([tf_result_values, tf_result_indices],
                                                       feed_dict=dict_to_feed)
                else:
                    sess.run([tf_result_values.op, tf_result_indices.op],
                             feed_dict=dict_to_feed)
                tf_time += time.time() - tf_start

            sess.close()
            log.info('STAGE %d LOOP %d mat2 %s processed (%.2f / %.2f secs elapsed)' % (
            idx1 + 1, idx2 + 1, str(MAT2_SHAPE), sampling_time, tf_time))


            # cut off the dummy values in the last loop
            if idx1 == (FIRST_ITERATION_COUNT - 1):
                # SLICE1_SIZE = self.mat1_targets_count % SLICE1_SIZE
                SLICE1_SIZE = self.mat1_targets_count % SLICE1_SIZE if self.mat1_targets_count != SLICE1_SIZE else self.mat1_targets_count
                ret_values = ret_values[:SLICE1_SIZE, :]
                ret_indices = ret_indices[:SLICE1_SIZE, :]

            if trace:
                self.ret_values, self.ret_indices = ret_values, ret_indices
                print(ret_values[:3, :100])
                print(ret_indices[:3, :100])
                break

            if writing_thread is not None:
                writing_thread.join()

            # writing_thread = Process(target=self.write_result, args=(idx1+1, ret_values, ret_indices, START_TARGET_IDX1, RANK_TO_SAVE, filename_to_save, axis))
            writing_thread = Process(target=self.write_result, args=(
            idx1 + 1, copy.deepcopy(ret_values), copy.deepcopy(ret_indices), START_TARGET_IDX1, RANK_TO_SAVE,
            filename_to_save, axis))
            writing_thread.start()
            log.info('STAGE %d total elapsed time %.2f sec' % (idx1 + 1, time.time() - start1))
        if writing_thread is not None:
            writing_thread.join()

        log.info('similarity iteration done : %.2f secs elapsed' % (time.time() - calc_time))

    def sample_mat2(self, idx2):
        mat2_sample = self.mat2[self.mat2['MAT2_ITERATION'] == idx2]
        self.next_mat2_indices = np.array(mat2_sample[['MAT2_SLICE_IDX', 'FTR_IDX']], dtype=np.int64)
        self.next_mat2_values = np.array(mat2_sample['NORMED_RATE'], dtype=np.float64)

    def write_result(self, stage, ret_values, ret_indices, start_target_idx1, rank_to_save, filename_to_save, axis=0):
        writing_time = time.time()
        with open(filename_to_save, 'a') as fw:
            if axis == 1:
                for user_idx in range(len(ret_values)):
                    tgt_id = self.target_map1.get(start_target_idx1 + user_idx)
                    sim_users, sim_scores = [], []
                    for loop_idx in range(0, len(ret_values[user_idx])):
                        sim_users.append(self.target_map2.get(ret_indices[user_idx][loop_idx]))
                        sim_scores.append(ret_values[user_idx][loop_idx])
                        # fw.write('%s,%s,%.6f' % (tgt_id, self.target_map2.get(ret_indices[user_idx][loop_idx]), ret_values[user_idx][loop_idx]))
                    fw.write('%s,%s,%s' % (tgt_id, ','.join(sim_users), ','.join(['%.6f' % x for x in sim_scores])))
            else:
                for user_idx in range(len(ret_values)):
                    tgt_id = self.target_map1.get(start_target_idx1 + user_idx)
                    for loop_idx in range(0, len(ret_values[user_idx])):
                        fw.write('%s,%s,%.6f' % (tgt_id, self.target_map2.get(ret_indices[user_idx][loop_idx]), ret_values[user_idx][loop_idx]))

        log.info('STAGE %d result mat %s has been written (%.2f elapsed)' % (
        stage, str(ret_values.shape), time.time() - writing_time))

