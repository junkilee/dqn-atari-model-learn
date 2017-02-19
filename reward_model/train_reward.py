# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
import sys
from datetime import datetime
from time import time
import os
import numpy as np
import gym
from keras.layers import Dense, Input, Convolution2D, Activation, Flatten, merge
from keras.models import Model
from keras.optimizers import Adam
import pandas as pd
from tqdm import tqdm
from jun_batch_helper import generate_data, extract_all_episodes 

IMAGE_SHAPE = (4, 84, 84)
NUM_FRAMES = 4
NUM_ACTIONS = 3

def get_reward_model():
  input1 = Input(shape=IMAGE_SHAPE)
  input2 = Input(shape=(NUM_ACTIONS, ))
  conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(input1)
  conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(conv1)
  conv3 = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')(conv2)
  flat = Flatten()(conv3)
  merged = merge([flat, input2], mode='concat', concat_axis = 1)
  dense = Dense(512, activation='relu')(merged)
  output = Dense(1)(dense)
  model = Model(input=[input1, input2], output = output)

  adam = Adam(lr=1e-6)
  model.compile(loss='mse',optimizer=adam, metrics=['accuracy', 'mse'])

  print(model.summary())

  return model

if __name__ == "__main__":
  if len(sys.argv) > 1:
    train_id = sys.argv[1]
  else:
    train_id = '99'

  # setting the directory and filename for train and test data files
  train_dir = "freeway/train"
  test_dir = "freeway/test"
  model_dir = "model/"
  postfix = train_id
  model_filename = os.path.join(model_dir, 'model_' + postfix)

  # default training parameters
  epochs = 1000000
  batch_size = 32

  # initialize numpy
  #seed = 7
  #np.random.seed(seed)

  model = get_reward_model()
  
  #train_frames, train_actions, train_rewards = extract_all_episodes(test_dir, NUM_FRAMES)
  train_frames, train_actions, train_rewards = extract_all_episodes(train_dir, NUM_FRAMES)
  test_frames, test_actions, test_rewards = extract_all_episodes(test_dir, NUM_FRAMES)

  start_time = time()
  model.fit_generator(generate_data(train_frames, train_actions, train_rewards, NUM_FRAMES), samples_per_epoch = 500000, nb_epoch = 50, 
    validation_data = generate_data(test_frames, test_actions, test_rewards, NUM_FRAMES), nb_val_samples = 50000)
  end_time = time()
  print ('elasped time = {} secs'.format(end_time - start_time))
  #score = model.evaluate(test_data[0], test_data[1], batch_size = batch_size)
  #print('\n')
  #print ('score = ', score)

  # serialize model to JSON
  model.save(model_filename + '.h5')
