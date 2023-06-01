import pooltool as pt
import numpy as np
import torch
from scipy.optimize import differential_evolution
from copy import deepcopy
import functools

#variables associated to pool
labels = ['cue', '1', '2', '3', '4', '5', '6', '7', '8', '9']
table = pt.PocketTable(model_name="7_foot")
pockets = table.get_pockets()

R = pt.constants.R
W = pt.constants.table_width
L = pt.constants.table_length

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
    
    def __init__(self, model, features=[], output='position', n_balls=3, hit_scale='std'):
        
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
        
        if self.output == 'hit':
            hit = self.model.predict(obs.reshape(1,-1))[0]
            
        else:
            hit = get_hit(balls, obs, self.model, output=self.output)
        
        if self.scale == 'std':
            hit = np.array([3, 1, 0.6])*hit + np.array([0,-0.5,-0.3])
        
        return hit
        

def random_ball(location = None):
    a = np.random.rand(2)
    if not location is None:
        a[0] = (W/2-2*R)*a[0]+location[0]*W/2+R
        a[1] = (L/2-2*R)*a[1]+location[1]*L/2+R
    else:
        a[0] = (W-2*R)*a[0]+R
        a[1] = (L-2*R)*a[1]+R
    return a

def generate_balls(n, coords=[], divided=None, symmetry=False):
    
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
        balls[labels[i]] = pt.Ball(labels[i], xyz=points[i])
        
    return balls


""" Start of feature functions """

def balls_to_obs(balls, ball_keys=labels):
    
    length = len([key for key in balls.keys() if key in ball_keys])
    on_table = [key for key, value in balls.items() if value.rvw[0,2] > 0]
    obs = [balls[key].rvw[0,:2] for key in ball_keys if key in on_table] 
    
    if length-len(obs)>0:
        extra = np.zeros(2*(length-len(obs)))
        obs.append(extra)
        
    return np.hstack(obs)

# def balls_to_fullobs(balls, ball_keys=labels):
    
#     length = len([key for key in balls.keys() if key in ball_keys])
#     on_table = [key for key, value in balls.items() if value.rvw[0,2] > 0]
#     obs = [balls[key].rvw[0,:2] for key in ball_keys if key in on_table] 
    
#     if length-len(obs)>0:
#         extra = np.zeros(2*(length-len(obs)))
#         obs.append(extra)
        
#     cue = pt.Cue(cueing_ball=balls['cue'])
#     cue.aim_for_best_pocket(balls[target], pockets.values())
#     phi = cue.phi
    
#     pocket = pt.potting.pick_best_pot(cue, balls['1'], pockets.values())
#     if pocket is None:
#         continue
#     if pocket.id == 'lb':
#         p_a = [1,0,0,0,0,0]
#     elif pocket.id == 'lc':
#         p_a = [0,1,0,0,0,0]
#     elif pocket.id == 'lt':
#         p_a = [0,0,1,0,0,0]
#     elif pocket.id == 'rb':
#         p_a = [0,0,0,1,0,0]
#     elif pocket.id == 'rc':
#         p_a = [0,0,0,0,1,0]
#     else:
#         p_a = [0,0,0,0,0,1]
        
#     return np.hstack(obs)


def positions(ball_keys):
    return functools.partial(balls_to_obs, ball_keys=ball_keys)


def angle(balls):
    cue = pt.Cue(cueing_ball=balls['cue'])
    target = list(balls.keys())[1]
    cue.aim_for_best_pocket(balls[target], pockets.values())
    return [cue.phi]


def pocket(balls):
    target = list(balls.keys())[1]
    cue = pt.Cue(cueing_ball=balls['cue'])
    pocket = pt.potting.pick_best_pot(cue, balls[target], pockets.values())
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
    return p_a
    

""" End of feature functions """


def shoot(balls, target, speed, vspin, hspin, phi=None):
    
    cue = pt.Cue(cueing_ball=balls['cue'])

    if phi == None:
        cue.aim_for_best_pocket(balls[target], pockets.values())
    else:
        cue.phi = phi

    cue.strike(V0=speed, b=vspin, a=hspin)
    shot = pt.System(cue=cue, table=table, balls=balls)

    try:
        shot.simulate(continuize=False)
        success = True
    except:
        success = False
                
    return balls, success


def get_hit(balls, obs, model, output='position', maximize=True):
    
    """
    Args:
        balls = current layout of the balls
        obs = encoded features of the balls relative to the model.
            The first 6 entries must be the coordinatinates of the cue ball followed by the next two object balls.
        model = either a single model or a list of models
        output = type of value outputted by the model. Either 'value' or 'location'
        maximize = whether the output is already optimal or we need to optimize
    """
    
    ball_obs = balls_to_obs(balls)
    next_ball = ball_obs[4:6]
    
    def get_loss(hit, obs, model):
        
        input_array = np.hstack((obs,hit))
        
        if type(model) is list:
            pred = 0
            for tree in model:
                pred += tree.predict(input_array.reshape(1,3))
            pred = pred/len(model)
        else:
            pred = model.predict(input_array.reshape(1,3))
        
        pred = pred.reshape(-1)
        
        if output == 'position':
            ball_pos = np.hstack((pred, next_ball))
            pred = reg_net(torch.as_tensor(ball_pos, dtype=torch.float32)).detach().numpy()
        
        return -pred

    args = (obs, model)

    bounds = [(0,1), (0,1), (0,1)]

    if maximize:
        result = differential_evolution(get_loss, bounds, args, maxiter=100, tol=1e-7)
        a = result.x
    else:
        a = model.act(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
        
    hit = np.array([3, 1, 0.6])*a + np.array([0,-0.5,-0.3])
    return hit


def view_shots(poolai, n_shots=10):
    
    interface = pt.ShotViewer()
    
    for _ in range(n_shots):
        
        n_balls = poolai.n_balls
        balls = generate_balls(n_balls, coords=[[0.5,1],[0.05,0.2]])
        target = list(balls.keys())[1]

        hit = poolai.action(balls)
        
        cue = pt.Cue(cueing_ball=balls['cue'])
        cue.aim_for_best_pocket(balls[target], pockets.values())
        
        # show starting point
        shot = pt.System(cue=cue, table=table, balls=balls)
        interface.show(shot)

        # show actual shot
        cue.strike(V0=hit[0], b=hit[1], a=hit[2])
        shot.simulate()

        interface.show(shot)
    
    interface.stop()
        
def evaluate(models, n_shot=100):
    
    scores = defaultdict(int)
    
    for _ in range(n_shots):
        
        n_balls = models[0].n_balls
        c_balls = generate_balls(n_balls, symmetry=True)
        
        for poolai in models:
            balls = deepcopy(c_balls)
            target = '1'

            hit = poolai.action(balls)

            balls, success = shoot(balls, target, hit[0], hit[1], hit[2])
            if not success:
                break
            
            on_table = [key for key, value in balls.items() if value.rvw[0,2] > 0]
            if 'cue' in on_table and '1' not in on_table and '2' in on_table:
                pred = reg_net(torch.as_tensor(np.hstack((balls['cue'].rvw[0,:2],
                                                          balls['2'].rvw[0,:2])), dtype=torch.float32))
                score = pred.item()
            else:
                score = 0
            
            scores[poolai] += score
        
    return scores
        
        
def view_data(df, interface):
    
    def obs_to_balls(obs):
        balls = {}
        for i in range(len(obs)//2):
            balls[labels[i]] = pt.Ball(labels[i], xyz=[obs[2*i],obs[2*i+1]])
        return balls
    
    for i in range(len(df)):
        row = df.iloc[i,:]
        balls = obs_to_balls(row[:6])
        hit = row[6:]
        
        target = '1'     
        cue = pt.Cue(cueing_ball=balls['cue'])
        cue.aim_for_best_pocket(balls[target], pockets.values())
        
        # show starting point
        shot = pt.System(cue=cue, table=table, balls=balls)
        interface.show(shot)

        # show actual shot
        cue.strike(V0=hit[0], b=hit[1], a=hit[2])
        shot.simulate()

        interface.show(shot)
        
    interface.stop()
        
            