from inference import ImportanceSampling, MetropolisHastings
from distribution import uniform
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, pi, log

#On suppose ci dessous que vec est de norme 1.
Obstacle = namedtuple('Obstacle', ['center', 'vec', 'width'])
#Attention, dans la suite, le point, ou la balle, est une matrice 2*2.
#La première ligne est son vecteur position, la deuxième son vecteur vitesse.

#Modèle probabiliste.
def bouncing_ball(prob, data):
    obstacles = []
    to_return = []

    ball_pos, dt, ground, bucket, nb_platform, sample_uniform_range_x, sample_uniform_range_y, platform_width = data

    #Génération des plateformes.
    for _ in range(nb_platform):
        angle = prob.sample(uniform(0, pi))
        obs_vec = np.array([cos(angle), sin(angle)])
        center = np.array([prob.sample(uniform(*sample_uniform_range_x)),\
            prob.sample(uniform(*sample_uniform_range_y))])
        obs = Obstacle(center = center,\
            vec = obs_vec,
            width = platform_width)
        obstacles.append(obs)
        to_return.extend([angle, center[0], center[1]])
    
    #Execution du modèle, attribution du score.
    inter, _ = deterministic_bouncing_ball(ball_pos, dt, ground, bucket, obstacles)
    distance_to_bucket = calc_distance(inter, bucket)
    prob.factor(-log(1 + distance_to_bucket))
    return tuple(to_return)

#Calcule la distance entre le point d'impact de la balle sur le sol, et le seau.
def calc_distance(inter, bucket):
    inter = inter[0]
    if inter is None:
        distance_to_bucket = float('inf')
    else:
        to_bucket = inter - bucket.center
        distance_to_bucket = max(0, sqrt(to_bucket @ to_bucket) - bucket.width / 2)
    return distance_to_bucket

#Simule la balle.
def deterministic_bouncing_ball(ball_pos, dt, ground, bucket, obstacles):
    pt = np.array([ball_pos, np.array([0, 0])])
    trajectory = [pt]
    ground_hit = False
    t = 0
    while not ground_hit:
        t += dt
        pt, ground_hit = step(pt, dt, ground, obstacles)
        trajectory.append(pt)
        #Au cas où la simulation ne peut pas aboutir.
        if t > 1000:
            return None, trajectory
    return trajectory[-1], trajectory

#Avance d'un pas de temps dans la simulation.
def step(old_pt, dt, ground, obstacles):
    new_pt = move(old_pt, dt)
    for obs in obstacles:
        inter = intersect(old_pt, new_pt, obs)
        if inter is not None:
            new_pt = push(new_pt, obs)
    inter = intersect(old_pt, new_pt, ground)
    #Si la balle a touché le sol, le point que l'on renvoit est
    #le point d'intersection.
    if inter is not None:
        return np.array([inter, np.array([0, 0])]), True
    return new_pt, False

#Pousse un point qui a traversé une plateforme
def push(pt, obs):
    new_pt = pt.copy()
    #new_pt[0] = obs.center + ((pt[0] - obs.center) @ obs.vec)
    new_pt[0] = obs.center + vec_symmetric(new_pt[0] - obs.center, obs.vec)
    new_pt[1] = vec_symmetric(new_pt[1], obs.vec)
    return new_pt

#Applique les forces à la balle.
def move(pt, dt):
    g = np.array([0, -9.8])
    gamma = np.array([0.1, 0.1])
    #Sans frottement
    #a = g
    #Frottements linéraires
    #a = g - gamma * pt[1]
    #Frottement quadratique
    a = g - gamma * pt[1] * normalize(pt[1])
    return np.array([pt[0] + pt[1] * dt, pt[1] + a * dt])

#Vérifir si la balle, entre deux pas de temps, a intersecté un obstacle.
def intersect(old_pt, new_pt, obs) -> bool:
    #On vérifie d'abord si elle a intersecté la droite portant l'obstacle.
    obs_ort_vec = ort_vector(obs.vec)
    if ((old_pt[0] - obs.center) @ obs_ort_vec >= 0) ==\
        ((new_pt[0] - obs.center) @ obs_ort_vec >= 0) :
        return None
    #Si oui, le traverse-t-on au niveau de l'obstacle ?
    #On calcule le point d'intersection, et on vérifie qu'il est bien sur l'obstacle.
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

#renvoie le symmétrique du vecteur u par rapport au vecteur v.
def vec_symmetric(u, v):
    return 2 * (u @ v) / (v @ v) * v - u

#Affiche la trajectoire de la balle.
def plot_traj(traj, ground, bucket, obstacles):
    plt.axline(ground.center, ground.center + ground.vec, color='brown')
    bucket1 = bucket.center - bucket.vec * bucket.width / 2
    bucket2 = bucket.center + bucket.vec * bucket.width / 2
    plt.plot([bucket1[0], (bucket1[0] + bucket2[0]) / 2, bucket2[0], bucket1[0]],\
        [bucket1[1], (bucket1[1] + bucket2[1]) / 2 -1, bucket2[1], bucket1[1]],
        color='black')
    for obs in obstacles:
        obs1 = obs.center - obs.vec * obs.width / 2
        obs2 = obs.center + obs.vec * obs.width / 2
        plt.plot([obs1[0], obs2[0]], [obs1[1], obs2[1]],
        color='green')
    x = [elem[0][0] for elem in traj]
    y = [elem[0][1] for elem in traj]
    plt.plot(x, y, color='red')
    plt.axis('scaled')
    plt.show()

#Renvoie le vecteur normal de vec.
def normalize(vec):
    norm = sqrt(vec @ vec)
    if norm == 0:
        return 0
    return vec / norm

#Renvoie le vecteur orthogonal à vec.
#Si vec est unitaire, l'orthogonal aussi.
def ort_vector(vec):
    return np.array([-vec[1], vec[0]])

#Affiche la meilleur trajectoire d'une distribution
def plot_best_prob(dist, data):
    #On reconsitue les conditions de la simulation du meilleur score.
    ball_pos, dt, ground, bucket, *_ = data
    supp = dist._support
    best = supp.values[supp.probs.index(max(supp.probs))]
    obstacles = []
    for i in range(0, len(best), 3):
        angle = best[i]
        obs_vec = np.array([cos(angle), sin(angle)])
        center = np.array([best[i+1], best[i+2]])
        obs = Obstacle(center = center,\
            vec = obs_vec,
            width = 4)
        obstacles.append(obs)
    #On resimule, puis on affiche.
    inter, traj = deterministic_bouncing_ball(ball_pos, dt, ground, bucket, obstacles)
    distance = calc_distance(inter, bucket)
    print("Meilleure distance : ", distance)
    plot_traj(traj, ground, bucket, obstacles)


if __name__ == '__main__':

    #Création des données.
    ball_pos = np.array([0, 8])
    ground = Obstacle(center = np.array([19, 0]),\
        vec= np.array([1, 0]),
        width=float('inf'))
    bucket = Obstacle(center = ground.center,\
        vec= ground.vec,
        width=0.5)
    dt = 0.01
    sample_uniform_range_x = (0, bucket.center[0]+1)
    sample_uniform_range_y = (0, ball_pos[1])
    platform_width = 4
    nb_platform = 3

    #Application du modèle, affichage.
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
    plot_best_prob(dist, data)
    
    method = MetropolisHastings
    print("-- Bouncing Ball, {} --".format(method.name()))
    mh = method(model, data)
    dist = mh.infer(n=1000)
    plot_best_prob(dist, data)
