# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
import cv2, numpy as np, os, tqdm

num_actions = 3

def modify_image(image):
  image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_LINEAR)
  image = np.rint(image[:, :, 0]*0.2989 + image[:, :, 1]*0.5870 + image[:, :, 2]*0.1140).astype(np.uint8)
  return image

def extract_episode(episode_dir, num_frames):
  with open(os.path.join(episode_dir, 'act.log')) as f:
    actions = np.array([int(line) for line in f.readlines()])
    actions_onehot = np.zeros((len(actions), num_actions), dtype=np.uint8)
    actions_onehot[range(len(actions)), actions] = 1
  with open(os.path.join(episode_dir, 'reward.log')) as f:
    rewards = np.array([int(line) for line in f.readlines()], dtype=np.uint8)
  frame_files = [os.path.join(episode_dir, x) for x in os.listdir(episode_dir)]
  frame_files = [f for f in frame_files if f.endswith('.png')]
  #for f in frame_files:
  #    cv2.imshow('f', cv2.resize(modify_image(cv2.imread(f)) / 255.0, (400, 400)))
  #    cv2.waitKey()
  #frames = np.zeros((len(frame_files) - num_frames + 1, num_frames, 84, 84), dtype=np.uint8)
  #terms = np.zeros((len(frame_files) - num_frames + 1, ), dtype=np.uint8)
  frames = []
  actions = []
  terms = []
  previous = np.zeros((num_frames, 84, 84), dtype=np.uint8)
  for i, f in enumerate(frame_files):
    image = modify_image(cv2.imread(f))
    previous[0:num_frames - 1] = previous[1:]
    previous[num_frames - 1] = image
    if i >= num_frames - 1: # when it reaches 3
      frames.append(previous)
      actions.append(actions_onehot[i])
      terms.append(0)
      if i >= (len(frame_files) - 1):
        terms.append(1)
      
  return frames, actions, terms

def extract_all_episodes(episode_dir, num_frames):
  episodes = [os.path.join(episode_dir, x) for x in os.listdir(episode_dir)]
  episodes = [e for e in episodes if os.path.isdir(e)]
  all_frames, all_actions, all_terms = [], [], []
  for episode in tqdm.tqdm(episodes):
    frames, actions, terms = extract_episode(episode, num_frames)
    all_frames += frames
    all_actions += actions
    all_terms += terms
  return all_frames, all_actions, all_terms

def generate_data(all_frames, all_actions, all_terms, num_frames, batch_size = 32):
  
  while 1:
    x = np.zeros((batch_size, num_frames, 84, 84))
    a = np.zeros((batch_size, num_actions))
    t = np.zeros((batch_size,))

    indexes = np.random.choice(len(all_frames), batch_size, replace='False')
    for i in range(batch_size):
      x[i] = all_frames[indexes[i]]
      a[i] = all_actions[indexes[i]]
      t[i] = all_terms[indexes[i]]

    yield ([x,a], t)
    

