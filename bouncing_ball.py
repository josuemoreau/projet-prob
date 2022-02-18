from inference import ImportanceSampling, MetropolisHastings
from distribution import uniform
from test_inference import test
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, pi, log

#On suppose ci dessous que col_vec et ort_vec sont de norme 1.
Obstacle = namedtuple('Obstacle', ['center', 'col_vec', 'width'])
#Ma balle n'a pas de rayon. Inchallah !

def bouncing_ball(prob, data):
    obstacles = []
    to_return = []

    ball_pos, dt, ground, bucket, nb_platform, sample_uniform_range_x, sample_uniform_range_y, platform_width = data

    for _ in range(nb_platform):
        angle = prob.sample(uniform(0, pi))
        obs_vec = np.array([cos(angle), sin(angle)])
        center = np.array([prob.sample(uniform(*sample_uniform_range_x)),\
            prob.sample(uniform(*sample_uniform_range_y))])
        obs = Obstacle(center = center,\
            col_vec = obs_vec,
            width = platform_width)
        obstacles.append(obs)
        to_return.extend([angle, center[0], center[1]])
    
    
    inter, _ = deterministic_bouncing_ball(ball_pos, dt, ground, bucket, obstacles)
    distance_to_bucket = calc_distance(inter, bucket)
    prob.factor(-log(1 + distance_to_bucket))
    return tuple(to_return)

def calc_distance(inter, bucket):
    inter = inter[0]
    if inter is None:
        distance_to_bucket = float('inf')
    else:
        to_bucket = inter - bucket.center
        distance_to_bucket = max(0, sqrt(to_bucket @ to_bucket) - bucket.width / 2)
    return distance_to_bucket

def deterministic_bouncing_ball(ball_pos, dt, ground, bucket, obstacles):
    pt = np.array([ball_pos, np.array([0, 0])])
    trajectory = [pt]
    ground_hit = False
    t = 0
    while not ground_hit:
        t += dt
        pt, ground_hit = step(pt, dt, ground, obstacles)
        trajectory.append(pt)
        if t > 1000:
            return None, trajectory
    return trajectory[-1], trajectory

def step(old_pt, dt, ground, obstacles):
    new_pt = move(old_pt, dt)
    for obs in obstacles:
        inter = intersect(old_pt, new_pt, obs)
        if inter is not None:
            new_pt = push(new_pt, obs)
    inter = intersect(old_pt, new_pt, ground)
    if inter is not None:
        return np.array([inter, np.array([0, 0])]), True
    return new_pt, False

def push(pt, obs):
    new_pt = pt.copy()
    #new_pt[0] = obs.center + ((pt[0] - obs.center) @ obs.col_vec)
    new_pt[0] = obs.center + vec_symmetric(new_pt[0] - obs.center, obs.col_vec)
    new_pt[1] = vec_symmetric(new_pt[1], obs.col_vec)
    return new_pt

def move(pt, dt):
    g = np.array([0, -9.8])
    gamma = np.array([0.1, 0.1])
    #Sans frottement
    #a = g
    #Frottements linéraires
    #a = g - gamma * pt[1]
    #Frottement quadratique
    a = g - gamma * pt[1] * normalize(pt[1])
    #D'abord compute v, puis x avec le nouveau v ?
    return np.array([pt[0] + pt[1] * dt, pt[1] + a * dt])

def intersect(old_pt, new_pt, obs) -> bool:
    #Traverse-t-on la droite qui porte l'obstacle ?
    obs_ort_vec = ort_vector(obs.col_vec)
    if ((old_pt[0] - obs.center) @ obs_ort_vec >= 0) ==\
        ((new_pt[0] - obs.center) @ obs_ort_vec >= 0) :
        return None
    #Si oui, le traverse-t-on au niveau de l'obstacle ?
    #On calcule le point d'intersection
    V1 = obs_ort_vec
    A1 = obs.center
    V2 = ort_vector(new_pt[0] - old_pt[0])
    A2 = old_pt[0]
    inter_x = (V2[1]*(V1 @ A1) - V1[1]*(V2 @ A2)) / (V1[0]*V2[1] - V1[1]*V2[0])
    inter_y = (V2[0]*(V1 @ A1) - V1[0]*(V2 @ A2)) / (V1[1]*V2[0] - V1[0]*V2[1])
    inter = np.array([inter_x, inter_y])
    if (inter - obs.center) @ (inter - obs.center) <= (obs.width/2)**2:
        return inter
    return None

def vec_symmetric(u, v):
    '''Symmétrique du vecteur u par rapport au vecteur v'''
    return 2 * (u @ v) / (v @ v) * v - u

def plot_traj(traj, ground, bucket, obstacles):
    plt.axline(ground.center, ground.center + ground.col_vec, color='brown')
    bucket1 = bucket.center - bucket.col_vec * bucket.width / 2
    bucket2 = bucket.center + bucket.col_vec * bucket.width / 2
    plt.plot([bucket1[0], (bucket1[0] + bucket2[0]) / 2, bucket2[0], bucket1[0]],\
        [bucket1[1], (bucket1[1] + bucket2[1]) / 2 -1, bucket2[1], bucket1[1]],
        color='black')
    for obs in obstacles:
        obs1 = obs.center - obs.col_vec * obs.width / 2
        obs2 = obs.center + obs.col_vec * obs.width / 2
        plt.plot([obs1[0], obs2[0]], [obs1[1], obs2[1]],
        color='green')
    x = [elem[0][0] for elem in traj]
    y = [elem[0][1] for elem in traj]
    plt.plot(x, y, color='red')
    plt.axis('scaled')
    plt.show()

def normalize(vector):
    norm = sqrt(vector @ vector)
    if norm == 0:
        return 0
    return vector / norm

def ort_vector(vec):
    return np.array([-vec[1], vec[0]])


def plot_prob(dist, data):
    ball_pos, dt, ground, bucket, nb_platform, sample_uniform_range_x, sample_uniform_range_y, platform_width = data
    supp = dist._support
    best = supp.values[supp.probs.index(max(supp.probs))]
    obstacles = []
    for i in range(0, len(best), 3):
        angle = best[i]
        obs_vec = np.array([cos(angle), sin(angle)])
        center = np.array([best[i+1], best[i+2]])
        obs = Obstacle(center = center,\
            col_vec = obs_vec,
            width = 4)
        obstacles.append(obs)
    inter, traj = deterministic_bouncing_ball(ball_pos, dt, ground, bucket, obstacles)
    distance = calc_distance(inter, bucket)
    print("Meilleure distance : ", distance)
    plot_traj(traj, ground, bucket, obstacles)


if __name__ == '__main__':

    ball_pos = np.array([0, 8])
    ground = Obstacle(center = np.array([19, 0]),\
        col_vec= np.array([1, 0]),
        width=float('inf'))
    bucket = Obstacle(center = ground.center,\
        col_vec= ground.col_vec,
        width=0.5)
    dt = 0.01
    sample_uniform_range_x = (0, bucket.center[0]+1)
    sample_uniform_range_y = (0, ball_pos[1])
    platform_width = 4
    nb_platform = 3

    model = bouncing_ball
    data = [ball_pos, dt, ground, bucket, nb_platform, sample_uniform_range_x,\
        sample_uniform_range_y, platform_width]
    name = "Bouncing Ball"
    options = {
        'shrink': False,
        'plot_with_support': True,
        'plot_style': 'line+scatter'
    }
    
    method = ImportanceSampling
    print("-- Bouncing Ball, {} --".format(method.name()))
    mh = method(model, data)
    dist = mh.infer(n=1000)
    plot_prob(dist, data)
    
    method = MetropolisHastings
    print("-- Bouncing Ball, {} --".format(method.name()))
    mh = method(model, data)
    dist = mh.infer(n=1000)
    plot_prob(dist, data)

    


    '''ball_pos = np.array([0.5, 10])
    ground = Obstacle(center = np.array([0, 0]),\
        col_vec= np.array([1, 0]),
        width=float('inf'))
    bucket = Obstacle(center = ground.center,\
        col_vec= ground.col_vec,
        width=0.5)
    pl1_vec = np.array([cos(-pi/8), sin(-pi/8)])
    pl1 = Obstacle(center = np.array([0.5, 3]),\
        col_vec= pl1_vec,
        width=4)
    pl2_vec = np.array([cos(-pi/2), sin(-pi/2)])
    pl2 = Obstacle(center = np.array([7, 6]),\
        col_vec= pl2_vec,
        width=2)
    obstacles = [pl1, pl2]
    dt = 0.01
    in_bucket, traj = deterministic_bouncing_ball(ball_pos, dt, ground, bucket, obstacles)
    print(in_bucket)
    plot_traj(traj, ground, bucket, obstacles)'''

