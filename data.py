import pooltool as pt
import pandas as pd
import numpy as np
from pool import generate_balls, get_angle, shoot, balls_to_obs

table=pt.PocketTable(model_name="7_foot")
pockets = table.get_pockets()

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
        angle = get_angle(balls, '1') - 0.5
        for j in range(tests):
            angle = angle + delta
            shoot(balls, '1', 2, 0, 0, angle)
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

    df = pd.DataFrame(columns = ['cue_x', 'cue_y', 'one_x', 'one_y', 'two_x', 'two_y', 'speed', 'vspin', 'hspin', 'value'])

    while True:
        balls = generate_balls(3)
        obs = balls_to_obs(balls)
        target = list(balls.keys())[1]
        cue = pt.Cue(cueing_ball=balls['cue'])
        cue.aim_for_best_pocket(balls[target], pockets.values())

        #randomly generate hit
        a = np.random.rand(3)
        a = np.array([3, 1, 0.6])*a + np.array([0,-0.5,-0.3])

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