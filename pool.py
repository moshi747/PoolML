import pooltool as pt
import numpy as np
import torch
from scipy.optimize import differential_evolution
from copy import deepcopy
import functools
from collections import defaultdict

#variables associated to pool
labels = ['cue', '1', '2', '3', '4', '5', '6', '7', '8', '9']
table = pt.Table.pocket_table()

R = 0.028575
W = 1.9812 / 2
L = 1.9812

reg_net = torch.load('reg_net.pt')

class PoolAI:
    
    """
    
    A pool playing AI that tells you what kind of hit to put on the cue ball
    given the current layout of the balls
    
    Args:
        model: the base ML model
        features: 'list' the input features required for the model. Each item should be a function
            which acts on the ball dictionary
        output: 'str' the output type of the model. Either 'value' or 'position' or 'hit'
    
    """
    
    def __init__(self, model=None, features=[], output='position', n_balls=3, hit_scale='std'):
        
        self.model = model
        self.features = features
        self.output = output
        self.n_balls = n_balls
        self.scale = hit_scale
        
    def action(self, balls):
        
        feat_batch = []

        for func in self.features:
            feat = func(balls)
            feat_batch.append(feat)
        
        if len(feat_batch) > 1:
            feat_batch = tuple(feat_batch)
            obs = np.hstack(feat_batch)
        else:
            obs = np.array(feat_batch)
        
        if self.model == None:
            hit = np.array([1.5, 0, 0])

        elif self.output == 'hit':
            hit = self.model.predict(obs.reshape(1,-1))[0]
            
        else:
            hit = get_hit(balls, obs, self.model, output=self.output)
        
        if self.scale == 'std':
            hit = np.array([3, 1, 0.6])*hit + np.array([0,-0.5,-0.3])
        
        return hit

def create_system(n, coords=[], divided=None, symmetry=False):
    
    """
    Args:
        n = number of balls
        coord = coordinate (x,y) points of starting balls
        divided = section of the table to generate balls from
        symmetry = whether to account for symmetry of the table
    """

    points = []
    for coord in coords:
        points.append(coord)
        
    if symmetry and len(points) == 0:
        point = random_ball(location=[0,0])
        points.append(point)
    
    for _ in range(n-len(points)):
        close = True
        while close:
            test = random_ball(location=divided)
            close = False
            for point in points:
                if np.square(point - test).sum() <= 2*R:
                    close = True
                    break
        points.append(test)
    
    balls = {}
    
    for i in range(n):
        balls[labels[i]] = pt.Ball.create(labels[i], xy=points[i])

    shot = pt.System(
                table=table,
                cue = pt.Cue(),
                balls=balls
                )
    return shot
                

def random_ball(location = None):
    a = np.random.rand(2)
    if not location is None:
        a[0] = (W/2-2*R)*a[0]+location[0]*W/2+R
        a[1] = (L/2-2*R)*a[1]+location[1]*L/2+R
    else:
        a[0] = (W-2*R)*a[0]+R
        a[1] = (L-2*R)*a[1]+R
    return a


""" Start of feature functions """

def balls_to_obs(balls, ball_keys=labels):
    
    length = len([key for key in balls.keys() if key in ball_keys])
    on_table = [key for key, value in balls.items() if value.xyz[2] > 0]
    obs = [balls[key].xyz[:2] for key in ball_keys if key in on_table] 
    
    if length-len(obs)>0:
        extra = np.zeros(2*(length-len(obs)))
        obs.append(extra)
        
    return np.hstack(obs)

    

""" End of feature functions """


def shoot(shot, target, speed, vspin, hspin, phi=None):
    
    shot.cue.set_state(
        cue_ball_id='cue',
        V0=speed,
        b=vspin,
        a=hspin,
        theta=0,
    )

    if phi == None:
        shot.aim_for_best_pocket(target)
    else:
        shot.cue.phi = phi

    try:
        shot.strike()
        pt.simulate(shot)
        success = True
    except:
        success = False
                
    return shot, success


def rescale(coords, table_size):
    w, l = table_size[0], table_size[1]
    for center in coords:
        center[0] = center[0]*W/w
        center[1] = center[1]*L/l
    cue = coords[0]

    if cue[0]>W/2:
        for center in coords:
            center[0] = W-center[0]
    if cue[1]>L/2:
        for center in coords:
            center[1] = L-center[1]
    return coords
    
    
def view_shots(poolai, coords=None, n_shots=10):
    
    interface = pt.ShotViewer()
    n_balls = poolai.n_balls
    
    if coords==None:
        for _ in range(n_shots):

            shot = create_system(n_balls, coords)

            target = '1'
            balls = shot.balls
            hit = poolai.action(balls)

            _, success = shoot(shot, target, hit[0], hit[1], hit[2])
            if not success:
                continue

            # show actual shot
            interface.show(shot)
            
    else:
        shot = create_system(len(coords), coords)

        target = '1'
        balls = shot.balls

        c_balls = {}
        for label in labels[:n_balls]:
            c_balls[label] = balls[label]

        hit = poolai.action(c_balls)

        _, success = shoot(shot, target, hit[0], hit[1], hit[2])
        if not success:
            print('failed simulation')

        # show actual shot
        interface.show(shot)
    
    interface.exitfunc()
        
def evaluate(models, n_shots=100):
    
    scores = defaultdict(int)
    n_balls = models[0].n_balls
    
    for _ in range(n_shots):
        
        shot = create_system(len(coords), symmetry=True)
        
        for poolai in models:
            balls = shot.balls
            target = '1'

            hit = poolai.action(balls)

            _, success = shoot(shot, target, hit[0], hit[1], hit[2])
            if not success:
                break
            
            on_table = [key for key, value in balls.items() if value.xyz[2] > 0]

            if 'cue' in on_table and '1' not in on_table and '2' in on_table:
                pred = reg_net(torch.as_tensor(np.hstack((balls['cue'].xyz[:2],
                                                          balls['2'].xyz[:2])), dtype=torch.float32))
                score = pred.item()
            else:
                score = 0

            shot.reset_balls()
            shot.reset_history()
            
            scores[poolai] += score
    
    for poolai in models:
        scores[poolai] /= n_shots
        
    return scores
        
        
def view_data(df, interface):
    
    def obs_to_sys(obs):
        coords = []
        for i in range(len(obs)//2):
            coords.append([obs[2*i], obs[2*i+1]])
        shot = create_system(len(obs)//2, coords)
        return shot
    
    for i in range(len(df)):
        row = df.iloc[i,:]
        shot = obs_to_sys(row[:6])
        hit = row[6:]
        
        shot = shoot(shot, '1', hit[0], hit[1], hit[2])
        interface.show(shot)
        
    interface.stop()
        
            