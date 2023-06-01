import pooltool as pt
import pandas as pd
import numpy as np
import torch
from pool import create_system, shoot, balls_to_obs
from copy import deepcopy
import itertools
import time

reg_net = torch.load('reg_net.pt')

table = pt.Table.pocket_table()

#Generates data for shot difficulty (used to train reg_net)
def generate_data(n_iter, tests=100):

    df = pd.DataFrame(columns = ['cue_x', 'cue_y', 'one_x', 'one_y', 'success'])
    delta = 1/tests

    #We see as we vary the angle from (ideal - 0.5, ideal + 0.5) how often we make the shot. This measures the difficulty
    for i in range(n_iter):
        balls = generate_balls(2)
        make = 0
        obs = balls_to_obs(balls)[:4]

        #Add a small offset to the angle
        theta = angle(balls) - 0.5
        for j in range(tests):
            theta += delta
            shoot(balls, '1', 2, 0, 0, theta)
            if balls['1'].rvw[0,2] < 0:
                make += 1

        row = np.hstack([obs, [make]])
        df.loc[len(df.index)] = row

    return df


#Determins the best hit on the cue ball for the best position according to reg_net
def generate_hit_data(n_iter, reg_net):
    
    df = pd.DataFrame(columns = ['cue_x', 'cue_y', 'one_x', 'one_y', 'two_x', 'two_y', 'hit'])


    for i in range(n_iter):
        balls = generate_balls(3)
        target = list(balls.keys())[1]
        cue = pt.Cue(cueing_ball=balls['cue'])
        cue.aim_for_best_pocket(balls[target], pockets.values())
        phi = cue.phi
        
        high_score = 0
        success = True
        hit = 0
        
        #Loop over all the different ways we can hit the cue ball
        for j in range(891):
            
            c_balls = deepcopy(balls)
            speed, vspin, hspin = act_to_aim2(j)
            
            cue = pt.Cue(cueing_ball=c_balls['cue'])
            cue.phi = phi
            cue.strike(V0=speed, b=vspin, a=hspin)

            shot = pt.System(cue=cue, table=table, balls=c_balls)
            
            try:
                shot.simulate()
                on_table = [key for key, value in c_balls.items() if value.rvw[0,2] > 0]

                #Don't want to count if cue ball hits the next object ball since in real life that is not desired
                if (c_balls['2'].rvw[0,:2] != balls['2'].rvw[0,:2]).any():
                    continue

                #See how well the position is if we successfully potted the 1 ball and did not scratch
                if 'cue' in on_table and '1' not in on_table and '2' in on_table:

                    pred = reg_net(torch.as_tensor(np.hstack((c_balls['cue'].rvw[0,:2],
                                                              c_balls['2'].rvw[0,:2])), dtype=torch.float32))
                    score = pred.item()
                    if score > high_score:
                        hit = j
                        high_score = score
            except:
                success = False
                break

        if high_score > 0 and success:
            obs = balls_to_obs(balls)
            obs = np.append(obs, hit)   
            df.loc[len(df.index)] = obs
            
# Generates data for how good the position is for a hit in the two ball situation.

def get_value_data(steps):
    
    df = pd.DataFrame(columns = ['cue_x', 'cue_y', 'one_x', 'one_y', 'two_x', 'two_y', 'speed', 'vspin', 'hspin', 'value'])

    while True:
        balls = generate_balls(3, symmetry=True)
        obs = balls_to_obs(balls)
        target = list(balls.keys())[1]
        cue = pt.Cue(cueing_ball=balls['cue'])
        cue.aim_for_best_pocket(balls[target], pockets.values())

        #randomly generate hit
        a = np.random.rand(3)
        a = np.array([3, 1, 0.6])*a + np.array([0.5,-0.5,-0.3])

        #execute shot
        cue.strike(V0=a[0], b=a[1], a=a[2])
        shot = pt.System(cue=cue, table=table, balls=balls)

        try:
            shot.simulate()
            on_table = [key for key, value in balls.items() if value.rvw[0,2] > 0]

            if (balls['2'].rvw[0,:2] != balls['2'].rvw[0,:2]).any():
                score = 0

            elif 'cue' in on_table and '1' not in on_table and '2' in on_table:
                pred = reg_net(torch.as_tensor(np.hstack((balls['cue'].rvw[0,:2],
                                                          balls['2'].rvw[0,:2])), dtype=torch.float32))
                score = 10*(math.exp(pred.item())-1)

            else:
                score = 0

        except:
            continue

        row = np.hstack((obs, a, [score]))
        df.loc[len(df.index)] = row
        if len(df) == steps:
            break

    return df

def get_pos_data(steps):

    df = pd.DataFrame(columns = ['cue_x', 'cue_y', 'one_x', 'one_y', 'speed', 'vspin', 'hspin',
                                 'phi', 'lb', 'lc', 'lt', 'rb', 'rc', 'rt', 'end_x', 'end_y'])

    while True:
        balls = generate_balls(2, symmetry=True)
        obs = balls_to_obs(balls)
        cue = pt.Cue(cueing_ball=balls['cue'])
        cue.aim_for_best_pocket(balls['1'], pockets.values())
        phi = cue.phi
        pocket = pt.potting.pick_best_pot(cue, balls['1'], pockets.values())
        if pocket is None:
            continue
        
        if pocket.id == 'lb':
            p_a = [1,0,0,0,0,0]
        elif pocket.id == 'lc':
            p_a = [0,1,0,0,0,0]
        elif pocket.id == 'lt':
            p_a = [0,0,1,0,0,0]
        elif pocket.id == 'rb':
            p_a = [0,0,0,1,0,0]
        elif pocket.id == 'rc':
            p_a = [0,0,0,0,1,0]
        else:
            p_a = [0,0,0,0,0,1]

        #randomly generate hit
        a = np.random.rand(3)
        a = np.array([3, 1, 0.6])*a + np.array([0.5,-0.5,-0.3])

        #execute shot
        cue.strike(V0=a[0], b=a[1], a=a[2])
        shot = pt.System(cue=cue, table=table, balls=balls)

        balls, success = shoot(balls, '1', a[0], a[1], a[2], phi)
        
        if not success:
            continue
        
        end = balls_to_obs(balls)[:2]

        row = np.hstack((obs, a, [phi], p_a, end))
        df.loc[len(df.index)] = row
        if len(df) == steps:
            break

    return df


def get_hit_data(steps, model=None):
    
    df = pd.DataFrame(columns = ['cue_x', 'cue_y', 'one_x', 'one_y', 'two_x', 'two_y', 'speed', 'vspin', 'hspin'])
    step = 0
    
    S_ARRAY = np.linspace(0.5,3.5,21)
    V_ARRAY = np.linspace(-0.5,0.5,11)
    H_ARRAY = np.linspace(-0.3,0.3,7)
    
    def get_score(speed, vspin, hspin):
        adj = 0
        _, success = shoot(shot, '1', speed, vspin, hspin, phi)
        
        if np.linalg.norm(shot.balls['cue'].xyz[:2]-np.array([two_x,two_y])) < 0.25:
            adj += 0.025/max(np.linalg.norm(shot.balls['cue'].xyz[:2]-np.array([two_x,two_y])),0.025)
        
        if not success:
            score = 0.5
        elif shot.balls['cue'].xyz[2] < 0 or shot.balls['1'].xyz[2] > 0 or shot.balls['2'].xyz[2] < 0:
            score = 0
        elif shot.balls['2'].xyz[0] != two_x:
            adj += 0.4
            score = reg_net(torch.as_tensor(np.array(balls_to_obs(shot.balls)[:4]), dtype=torch.float32)).item()
        else:
            score = reg_net(torch.as_tensor(np.array(balls_to_obs(shot.balls)[:4]), dtype=torch.float32)).item()
        
        shot.reset_balls()
        shot.reset_history()
        return score-adj
    
    def hit_cube():
        cube = np.zeros((21,11,7))
        for i in range(21):
            for j in range(11):
                for k in range(7):
                    speed = S_ARRAY[i]
                    vspin = V_ARRAY[j]
                    hspin = H_ARRAY[k]
                    score = get_score(speed, vspin, hspin)
                    score -= np.abs(np.array([i,j,k])-np.array([7, 5, 3])).sum()/50
                    score = max(score, 0)
                    cube[i,j,k] = score
        return cube
    
    def find_max(cube):
        w, l, h = cube.shape
        max_tot = 0
        for i in range(w-2):
            for j in range(l-1):
                for k in range(h-1):
                    total = 0
                    for x, y, z in itertools.product([0,1,2], [0,1], [0,1]):
                        total += cube[i+x,j+y,k+z]
                    if total > max_tot:
                        max_tot = total
                        max_ind = [i,j,k]
        if max_tot == 0:
            return 0
        max_ent = 0
        i, j, k = max_ind[0], max_ind[1], max_ind[2]
        for x, y, z in itertools.product([0,1,2], [0,1], [0,1]):
            if cube[i+x, j+y, k+z] > max_ent:
                max_ent = cube[i+x, j+y, k+z]
                max_ind = [i+x, j+y, k+z]
        return max_ind

    while step < steps:
        shot = create_system(3, symmetry=True)
        obs = balls_to_obs(shot.balls)
        two_x = shot.balls['2'].xyz[0]
        two_y = shot.balls['2'].xyz[1]
        
        shot, success = shoot(shot, '1', 1.5, 0, 0)
        
        if not success or shot.balls['1'].xyz[2] > 0:
            continue
        shot.reset_balls()

        shot.aim_for_best_pocket('1')
        phi = shot.cue.phi
        
        if model is not None:
            pocket = pt.potting.pick_best_pot(cue, balls['1'], pockets.values())
            if pocket is None:
                continue
            if pocket.id == 'lb':
                p_a = [1,0,0,0,0,0]
            elif pocket.id == 'lc':
                p_a = [0,1,0,0,0,0]
            elif pocket.id == 'lt':
                p_a = [0,0,1,0,0,0]
            elif pocket.id == 'rb':
                p_a = [0,0,0,1,0,0]
            elif pocket.id == 'rc':
                p_a = [0,0,0,0,1,0]
            else:
                p_a = [0,0,0,0,0,1]
        
        cube = hit_cube()
        max_ind = find_max(cube)
        if max_ind == 0:
            continue

        speed = S_ARRAY[max_ind[0]]
        vspin = V_ARRAY[max_ind[1]]
        hspin = H_ARRAY[max_ind[2]]
        
        hit = [speed, vspin, hspin]
        row = np.hstack((obs, hit))
        df.loc[len(df.index)] = row
        step += 1
        if step%10 == 0:
            print(f'{step} steps done')
            
    return df