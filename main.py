import os  # 用于操作文件和目录
import shutil  # 用于复制文件和目录树
import subprocess  # 用于执行外部命令
import pickle  # 用于序列化和反序列化Python对象结构
from collections import deque  # 用于创建一个双端队列，支持从两端添加或删除元素

import gym  # gym lib
import torch  
import numpy as np 
import matplotlib.pyplot as plt  
from PIL import Image  

from agent import DQNAgent  

# def train sub function
def train(n_episodes=100,  
          max_t=10000,  
          eps_start=1.0,  
          eps_end=0.01,  
          eps_decay=0.996):  
  scores = []  # to save score
  done_timesteps = []  
  scores_window = deque(maxlen=100) 
  eps = eps_start  
  for i_episode in range(1, n_episodes + 1):  
    state = env.reset()  
    score = 0  
    for timestep in range(max_t): 
      action = agent.act(state, eps) 
      next_state, reward, done, _ = env.step(action)  
      agent.step(state, action, reward, next_state, done)  
      state = next_state  
      score += reward  
      if done:  
        print(
            'Episode {} done in {} timesteps.\n'.format(
                i_episode,
                timestep))  
        done_timesteps.append(timestep)  
        break  # break loop 
      scores_window.append(score)
      scores.append(score)
      eps = max(eps * eps_decay, eps_end)

      if timestep % SAVE_EVERY == 0:  
        print('Episode {}\tTimestep {}\tAverage Score {:.2f}\n'.format(
            i_episode, timestep, np.mean(scores_window)), end="")  
        torch.save(
            agent.qnetwork_local.state_dict(),
            SAVE_DIR + 'model.pth')  # 保存模型
        with open(SAVE_DIR + 'scores', 'wb') as fp:  
          pickle.dump(scores, fp)
        with open(SAVE_DIR + 'dones', 'wb') as fp: 
          pickle.dump(done_timesteps, fp)

 
  torch.save(agent.qnetwork_local.state_dict(), SAVE_DIR + 'model.pth')
  with open(SAVE_DIR + 'scores', 'wb') as fp:
    pickle.dump(scores, fp)
  with open(SAVE_DIR + 'dones', 'wb') as fp:
    pickle.dump(done_timesteps, fp)

  return scores 

# test subfun
def test(env, trained_agent, n_games=1, n_steps_per_game=10000):
  for game in range(n_games):  
    frame_folder = 'tmp'  
    if not os.path.exists(frame_folder):  
      os.makedirs(frame_folder) 
    frame_files = []  
    observation = env.reset()  
    score = 0  
    for step in range(n_steps_per_game): 
      action = trained_agent.act(observation)  
      observation, reward, done, info = env.step(
          action) 
      frame = env.render(mode='rgb_array') 
      img = Image.fromarray(frame) 
      frame_file = os.path.join(
          frame_folder,
          f'frame_{step:04d}.png') 
      img.save(frame_file)  
      frame_files.append(frame_file) 
      score += reward  
      if done:  
        print('GAME-{} OVER! score={}'.format(game, score))  # 打印信息
        break  
    env.close()  

    # ffmpeg save gif
    ffmpeg_cmd = [
        r'D:\ffmpeg\bin\ffmpeg',
        '-framerate', '20',  
        '-i', os.path.join(frame_folder, 'frame_%04d.png'),  
        '-c:v', 'libx264',  
        '-pix_fmt', 'yuv420p',  
        'output_game_{}.mp4'.format(game) 
    ]
    subprocess.run(ffmpeg_cmd)  
    shutil.rmtree(frame_folder)  


# main func
if __name__ == '__main__':
  TRAIN = False  # train or test
  BUFFER_SIZE = int(1e5)  
  BATCH_SIZE = 64  
  GAMMA = 0.99  
  TAU = 1e-3  
  LR = 5e-4  
  UPDATE_EVERY = 100 
  SAVE_EVERY = 100  
  MAX_TIMESTEPS = 10  
  N_EPISODES = 10  
  SAVE_DIR = "./train/" 

  # get env from gym
  env = gym.make('SpaceInvaders-v0')  

  # train func
  if TRAIN:
    agent = DQNAgent(state_size=4, 
                     action_size=env.action_space.n, 
                     seed=0,  
                     lr=LR,  
                     gamma=GAMMA,  
                     tau=TAU,  
                     buffer_size=BUFFER_SIZE,  
                     batch_size=BATCH_SIZE,  
                     update_every=UPDATE_EVERY)  
    scores = train(n_episodes=N_EPISODES)  

    # plot
    N = 100
    fig = plt.figure() 
    ax = fig.add_subplot(111)  
    plt.plot(
        np.convolve(
            np.array(scores), np.ones(
                (N, )) / N, mode='valid'))  
    plt.ylabel('Score') 
    plt.xlabel('Timestep #') 
    plt.show() 
  else:
    N_GAMES = 5  
    N_STEPS_PER_GAME = 10000 

    test_agent = DQNAgent(state_size=4,
                          action_size=env.action_space.n,
                          seed=0)

    # load mod
    test_agent.qnetwork_local.load_state_dict(
        torch.load(SAVE_DIR + 'model.pth'))

    test_agent.qnetwork_local.eval()

    test(env, test_agent)
