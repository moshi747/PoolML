import pooltool as pt
import numpy as np
import math
import gym
from copy import deepcopy
import torch
from pool import generate_balls, balls_to_obs, shoot

#variables associated to pool
labels = ['cue', '1', '2', '3', '4', '5', '6', '7', '8', '9']
table = pt.PocketTable(model_name="7_foot")
pockets = table.get_pockets()

#model for predicting how easy a shot is
reg_net = torch.load('reg_net.pt')

n_balls = 3

class Pool_random(gym.Env):

    def __init__(self):

        # Observation space corresponding to x,y coordiinate of each ball
        self.observation_space = gym.spaces.Box(low=np.tile(np.array([0],dtype=np.float32),n_balls*2),
                                            high=np.tile(np.array([1,2],dtype=np.float32),n_balls),
                                            dtype=np.float32)

        # 3 dimensions corresponding to speed, vspin, hspin
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1],dtype=np.float32),
                                       high=np.array([1, 1, 1],dtype=np.float32), dtype=np.float32)
        
        self.balls = None
        self.n = n_balls
        self.observation = None
    
    
    def reset(self):
            
        # Generate starting layout of balls
        balls = generate_balls(self.n)
        self.balls = balls

        observation = balls_to_obs(balls)

        return observation
    
    def step(self, action):
        
        balls = self.balls
        terminated = False

        action = action*np.array([3,0.4,0.3])+np.array([3,0,0])
                
        on_table = [key for key, value in balls.items() if value.rvw[0,2] > 0]
        target = on_table[1]
        
        c_balls = shoot(balls, target, action[0], action[1], action[2])
        on_table = [key for key, value in c_balls.items() if value.rvw[0,2] > 0]
        
        if 'cue' not in on_table:
            reward = 0
            terminated = True
        elif target in on_table:
            if np.array([(balls[key].rvw[0] == balls[key].rvw[0]).all() for key in balls.keys() if key != 'cue']).all():
                reward = -10
            else:
                reward = -0.1
        elif len(on_table) > 2:
            next_ball = on_table[1]
            pred = reg_net(torch.as_tensor(np.hstack((balls['cue'].rvw[0,:2],
                                                      balls[next_ball].rvw[0,:2])), dtype=torch.float32))
            reward = math.exp(4*pred.item())
        elif len(on_table) == 2:
            next_ball = on_table[1]
            pred = reg_net(torch.as_tensor(np.hstack((balls['cue'].rvw[0,:2],
                                                      balls[next_ball].rvw[0,:2])), dtype=torch.float32))
            reward = math.exp(4*pred.item())
            terminated = True
        else:
            reward = 0
            terminated = True
            
        self.balls = c_balls
        observation = balls_to_obs(c_balls)
        
        return observation, reward, terminated, False
    
#Environment for starting with a fixed rack

class Pool_static(gym.Env):

    def __init__(self):

        # Observation space corresponding to x,y coordiinate of each ball
        self.observation_space = gym.spaces.Box(low=np.tile(np.array([0],dtype=np.float32),20),
                                            high=np.tile(np.array([1, 2],dtype=np.float32),10),
                                            dtype=np.float32)

        # 3 dimensions corresponding to speed, vspin, hspin
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1],dtype=np.float32),
                                       high=np.array([1, 1, 1],dtype=np.float32), dtype=np.float32)
        
        while True:
            try:
                balls = pt.get_nine_ball_rack(table, ordered = True)
                cue = pt.Cue(cueing_ball=balls['cue'])

                cue.aim_at_ball(balls['1'])
                cue.strike(V0=10, b=0, a=0)

                shot = pt.System(cue=cue, table=table, balls=balls)

                shot.simulate(continuize=False)
                break
            except:
                pass
        
        start_balls = deepcopy(balls)
        self.balls = start_balls
        self.observation = balls_to_obs(start_balls)
    
    def reset(self):
        
        self.balls = deepcopy(start_balls)
        return self.observation
    
    def step(self, action):
        
        balls = self.balls
        terminated = False

        action = action*np.array([3,0.4,0.3])+np.array([3,0,0])
                
        on_table = [key for key, value in balls.items() if value.rvw[0,2] > 0]
        target = on_table[1]
        
        balls = shoot(balls, target, action[0], action[1], action[2])
        on_table = [key for key, value in balls.items() if value.rvw[0,2] > 0]
        
        self.balls = balls
        
        if 'cue' not in on_table:
            reward = 0
            terminated = True
        elif target in on_table:
            if np.array([(balls[key].rvw[0] == balls[key].rvw[0]).all() for key in balls.keys() if key != 'cue']).all():
                reward = -10
            else:
                reward = -0.1
        elif len(on_table) > 2:
            next_ball = on_table[1]
            pred = reg_net(torch.as_tensor(np.hstack((balls['cue'].rvw[0,:2],
                                                      balls[next_ball].rvw[0,:2])), dtype=torch.float32))
            reward = math.exp(4*pred.item())
        elif len(on_table) == 2:
            next_ball = on_table[1]
            pred = reg_net(torch.as_tensor(np.hstack((balls['cue'].rvw[0,:2],
                                                      balls[next_ball].rvw[0,:2])), dtype=torch.float32))
            reward = math.exp(4*pred.item())
            terminated = True
        else:
            reward = 0
            terminated = True

        observation = balls_to_obs(balls)
        
        return observation, reward, terminated, False


#Environment for starting with a 9 ball rack
    
class Pool_break(gym.Env):

    def __init__(self):

        # Observation space corresponding to x,y coordiinate of each ball
        self.observation_space = gym.spaces.Box(low=np.tile(np.array([-90],dtype=np.float32),21),
                                            high=np.tile(np.array([90],dtype=np.float32),21),
                                            dtype=np.float32)

        # 3 dimensions corresponding to speed, vspin, hspin
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1],dtype=np.float32),
                                       high=np.array([1, 1, 1],dtype=np.float32), dtype=np.float32)
        
        self.balls = None
        self.observation = None
    
    def reset(self):
        
        while True:
            try:
                balls = pt.get_nine_ball_rack(table, ordered = True)
                cue = pt.Cue(cueing_ball=balls['cue'])

                cue.aim_at_ball(balls['1'])
                cue.strike(V0=10, b=0, a=0)

                shot = pt.System(cue=cue, table=table, balls=balls)

                shot.simulate(continuize=False)
                break
            except:
                pass
        
        self.balls = balls
        self.observation = balls_to_obs(balls)
        return self.observation
    
    def step(self, action):
        
        balls = self.balls
        terminated = False

        action = action*np.array([3,0.4,0.3])+np.array([3,0,0])
                
        on_table = [key for key, value in balls.items() if value.rvw[0,2] > 0]
        target = on_table[0]
        
        c_balls = shoot(balls, target, action[0], action[1], action[2])
        on_table = [key for key, value in c_balls.items() if value.rvw[0,2] > 0]
        
        """
        Various reward stages:
        (1) If scratch then -2 reward and episode is terminated
        (2) If miss then small penalty (-0.1)
        (3) If make then reward proportional to how well the position is for the next target according to reg_net
        (4) Episode terminates if the number of ball is <= 2
        """
        
        if 'cue' not in on_table:
            reward = -2
            terminated = True
        elif target in on_table:
            if np.array([(balls[key].rvw[0] == balls[key].rvw[0]).all() for key in labels if key != 'cue']).all():
                reward = -4
            else:
                reward = -0.1
        elif len(on_table) > 2:
            next_ball = on_table[0]
            pred = reg_net(torch.as_tensor(np.hstack((balls['cue'].rvw[0,:2],
                                                      balls[next_ball].rvw[0,:2])), dtype=torch.float32))
            reward = math.exp(pred.item()*4)
        elif len(on_table) == 2:
            next_ball = on_table[0]
            pred = reg_net(torch.as_tensor(np.hstack((balls['cue'].rvw[0,:2],
                                                      balls[next_ball].rvw[0,:2])), dtype=torch.float32))
            reward = pred.item()*2
            terminated = True
        else:
            reward = 0
            terminated = True
            
        self.balls = c_balls
        observation = balls_to_obs(c_balls)
        
        return observation, reward, terminated, False
