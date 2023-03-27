import pooltool as pt
import numpy as np
from copy import deepcopy

#variables associated to pool
labels = ['cue', '1', '2', '3', '4', '5', '6', '7', '8', '9']
table = pt.PocketTable(model_name="7_foot")
pockets = table.get_pockets()


def random_ball():
    a = np.random.rand(2)
    a[0] = (0.93)*a[0]+0.03
    a[1] = (1.92)*a[1]+0.03
    return a

def generate_balls(n_balls):
    points = []
    points.append(random_ball())
    
    for _ in range(n_balls-1):
        close = True
        while close:
            test = random_ball()
            close = False
            for point in points:
                if np.square(point - test).sum() < 0.06:
                    close = True
                    break
        points.append(test)

    balls = {}
    
    for i in range(n_balls):
        balls[labels[i]] = pt.Ball(labels[i], xyz=points[i])

    return balls

def balls_to_obs(balls):
    on_table = [key for key, value in balls.items() if value.rvw[0,2] > 0]
    obs = [balls[key].rvw[0,:2] for key in labels if key in on_table]
    
    cue = pt.Cue(cueing_ball=balls['cue'])
    
    extra = np.zeros(2*(len(balls)-len(obs)))
    obs.append(extra)
    return np.hstack(obs)

def get_angle(balls, target):
    cue = pt.Cue(cueing_ball=balls['cue'])
    cue.aim_for_best_pocket(balls[target], pockets.values())
    return cue.phi


def shoot(balls, target, speed, vspin, hspin, angle=None):

    cue = pt.Cue(cueing_ball=balls['cue'])

    if angle == None:
        cue.aim_for_best_pocket(balls[target], pockets.values())
    else:
        cue.phi = angle

    cue.strike(V0=speed, b=vspin, a=hspin)
    shot = pt.System(cue=cue, table=table, balls=balls)

    try:
        shot.simulate(continuize=False)
    except:
        pass
                
    return balls

def act_to_aim2(act):
    speed = act%9*0.3+0.3
    vspin = act%11*0.1-0.5
    hspin = act%9*0.1-0.4
    return speed, vspin, hspin
