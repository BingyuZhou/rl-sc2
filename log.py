import tensorflow as tf
import datetime

""" log info"""
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/" + current_time + "/train"
test_log_dir = "logs/" + current_time + "/test"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
