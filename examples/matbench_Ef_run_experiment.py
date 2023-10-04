#	Author: Zachary Humphreys
# 
#	This is a sample script demonstrating the training process
#	used for the matbench formation energy dataset, for fold 0.
#	This will not run as-is and requires pre-building of both
#	the model, and the dataset. 

import os
import sys
import numpy as np
import pickle
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import callbacks
from tensorflow_addons.activations import mish

# Load untrained model
model = tf.keras.models.load_model(r"model")

def print_test_results():
	y_preds = []
	y_tests = []
	for i in range(len(test_generator)):
		y_tests.append(sc.inverse_transform(test_generator[i][1].reshape(-1,1)))
		y_preds.append(sc.inverse_transform(model.predict(test_generator[i][0]).reshape(-1,1)))

	y_preds = np.vstack(y_preds)
	y_tests = np.vstack(y_tests)
	aes = np.abs(y_tests-y_preds)
	mae = np.mean(aes) 
	print(mae)
	print((np.abs(y_tests-y_tests.mean())).mean()/mae)
	print(aes.std())
	print(aes.max())
	ewt_001 = 100*np.sum(aes<=0.01)/aes.shape[0]
	ewt_002 = 100*np.sum(aes<=0.02)/aes.shape[0]
	print(f"EwT 10meV: {ewt_001}")
	print(f"EwT 20meV: {ewt_002}")

args = sys.argv
args = [None,"fold_0"]
os.chdir(args[1])
print(os.listdir(os.getcwd()))

with open(r"data.pkl", 'rb') as f:
	exp_dict = pickle.load(f)

def get_data_generators(X_train, X_val, X_test, y_train, y_val, y_test, batch_size = 96):
	from megnet.data.graph import GraphBatchGenerator
	train_generator = GraphBatchGenerator(*X_train, targets = y_train, batch_size=batch_size)
	val_generator	= GraphBatchGenerator(*X_val, targets = y_val, batch_size=batch_size)
	test_generator  = GraphBatchGenerator(*X_test, targets = y_test, batch_size=batch_size)
	return train_generator, val_generator, test_generator

# Load preconstructed dataset
X_train = exp_dict["X_train"]
X_val	= exp_dict["X_val"]
X_test	= exp_dict["X_test"]
y_train = exp_dict["y_train"]
y_val	= exp_dict["y_val"]
y_test	= exp_dict["y_test"]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
y_train = sc.fit_transform(y_train.reshape(-1,1))
y_val   = sc.transform(y_val.reshape(-1,1))
y_test  = sc.transform(y_test.reshape(-1,1))

train_generator, val_generator, test_generator = get_data_generators(X_train, X_val, X_test, y_train, y_val, y_test, 96)

def get_callbacks(attempt_name):
	patience = 15
	root_filepath = attempt_name
	checkpoint_filepath = root_filepath + r"/checkpoint"
	checkpoint_EMA_filepath = root_filepath + r"/EMA_checkpoint"
	cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath, monitor = "val_mae", save_best_only = True)
	cb_LRPlateau  = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience = patience, min_lr = 1.25e-4)
	cb_EarlyStop  = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience= int(patience*3)+3)
	cb_checkpointEMA = tfa.callbacks.AverageModelCheckpoint(filepath = checkpoint_EMA_filepath, monitor="val_mae", update_weights=True, save_best_only = True)
	cb_TerminateOnNan = tf.keras.callbacks.TerminateOnNaN()
	return [cb_checkpoint,cb_LRPlateau,cb_EarlyStop,cb_checkpointEMA,cb_TerminateOnNan]

callbacks = get_callbacks(experiment_name)
model.summary()
lr = 5e-4
opt = tf.keras.optimizers.Adam(lr)
opt = tfa.optimizers.MovingAverage(opt)
model.compile(opt,"mse",metrics=["mae"])
#print("All success!")
#exit()
# This will early stop because of large errors
model.fit(train_generator,
		  steps_per_epoch=len(train_generator),
		  validation_data=val_generator,
		  validation_steps=len(val_generator),
		  epochs=1000,
		  batch_size=1,
		  verbose = 2,
		  callbacks=callbacks)

def get_callbacks(attempt_name):
        patience = 15
        root_filepath = attempt_name
        checkpoint_filepath = root_filepath + r"/checkpoint"
        checkpoint_EMA_filepath = root_filepath + r"/EMA_checkpoint"
        cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath, monitor = "val_mae", save_best_only = True)
        cb_LRPlateau  = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience = patience, min_lr = 1.00e-4)
        cb_EarlyStop  = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience= int(patience*2)+3)
        cb_checkpointEMA = tfa.callbacks.AverageModelCheckpoint(filepath = checkpoint_EMA_filepath, monitor="val_mae", update_weights=True, save_best_only = True)
        cb_TerminateOnNan = tf.keras.callbacks.TerminateOnNaN()
        return [cb_checkpoint,cb_LRPlateau,cb_EarlyStop,cb_checkpointEMA,cb_TerminateOnNan]

callbacks = get_callbacks(experiment_name)
model.fit(train_generator,
                  steps_per_epoch=len(train_generator),
                  validation_data=val_generator,
                  validation_steps=len(val_generator),
                  epochs=1000,
                  batch_size=1,
                  verbose = 2,
                  callbacks=callbacks)

def get_callbacks(attempt_name):
        patience = 20
        root_filepath = attempt_name
        checkpoint_filepath = root_filepath + r"/checkpoint"
        checkpoint_EMA_filepath = root_filepath + r"/EMA_checkpoint"
        cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath, monitor = "val_mae", save_best_only = True)
        cb_LRPlateau  = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience = patience, min_lr = 1e-12)
        cb_EarlyStop  = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience= int(patience*4)+3)
        cb_checkpointEMA = tfa.callbacks.AverageModelCheckpoint(filepath = checkpoint_EMA_filepath, monitor="val_mae", update_weights=True, save_best_only = True)
        cb_TerminateOnNan = tf.keras.callbacks.TerminateOnNaN()
        return [cb_checkpoint,cb_LRPlateau,cb_EarlyStop,cb_checkpointEMA,cb_TerminateOnNan]

lr = 2.5e-4
opt = tf.keras.optimizers.Adam(lr)
opt = tfa.optimizers.MovingAverage(opt)
model.compile(opt,"mae",metrics=["mae"])
#print("All success!")
#exit()
callbacks = get_callbacks(experiment_name)
model.fit(train_generator,
                  steps_per_epoch=len(train_generator),
                  validation_data=val_generator,
                  validation_steps=len(val_generator),
                  epochs=1000,
                  batch_size=1,
                  verbose = 2,
                  callbacks=callbacks)

model.save(r"final")
