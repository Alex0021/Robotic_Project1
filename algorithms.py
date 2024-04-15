import numpy as np


def get_viewing_dir(drone, neighbors, algo: str):
    """
    Compute the viewing direction of the drone based on the neighbors
    with the selected algorithm.

    Args:
        drone (Drone): The selected drone
        neighbors (list[DroneNeighbor]): The neighbors of the drone
        algo (str): The algorithm to use to compute the viewing direction
    """
    assert algo in ["average", "outter", "tangent_plane"], "Algorithm {0} not supported".format(algo)
    return eval(algo)(drone, neighbors)

def average(drone, neighbors):
    """
    Compute the desired viewing direction based on the centroid of the neighbors.

    1. Find the centroid (mean) of the neighbors
    2. Set the viewing direction to the opposite of the centroid
    3. Don't forget to normalize the vector

    Args:
        drone (Drone): The selected drone
        neighbors (list[DroneNeighbor]): The neighbors of the drone
    """
    centroid = np.mean([n.get_abs_pos() for n in neighbors], axis=0)
    vec_to_centroid = centroid - drone.pos
    viewing_dir = -vec_to_centroid / np.linalg.norm(vec_to_centroid)
    return viewing_dir

def outter(drone, neighbors):
    """
    Compute the desired viewing direction based on the outter product of the neighbors.

    1. Find the outter product of the neighbors
    2. Set the viewing direction to the opposite of the outter product
    3. Don't forget to normalize the vector

    Args:
        drone (Drone): The selected drone
        neighbors (list[DroneNeighbor]): The neighbors of the drone
    """
    # Testing first with 2D case
    neighbors_pos = np.array([n.get_abs_pos() for n in neighbors])
    dists = neighbors_pos - drone.pos
    # Compute dot product between all combination of neighbors
    smaller = np.inf
    smaller_indices = (0, 0)
    for i in range(len(neighbors)):
        for j in range(i+1, len(neighbors)):
            d = np.dot(dists[i], dists[j])
            if d < smaller:
                smaller = d
                smaller_indices = (i, j)
    viewing_dir = (dists[smaller_indices[0]] + dists[smaller_indices[1]]) / 2
    viewing_dir = -viewing_dir / np.linalg.norm(viewing_dir)
    return viewing_dir

def tangent_plane(drone, neighbors):
    """
    Compute the desired viewing direction based on the tangent plane of the neighbors.

    1. Find the centroid (mean) of the neighbors
    2. Compute covariance matrix of the neighbors
    3. Apply PCA and keep the last eigenvector (with smallest eigenvalue)
    4. This vector gives the normal to the tangent plane
    5. Set the viewing direction to the opposite of the normal

    Args:
        drone (Drone): The selected drone
        neighbors (list[DroneNeighbor]): The neighbors of the drone
    """
    neighbors_pos = np.array([n.get_abs_pos() for n in neighbors])
    in_2d = np.std(neighbors_pos[:, 2]) < 0.01
    if in_2d:
        neighbors_pos = neighbors_pos[:, :2]
    centroid = np.mean(neighbors_pos, axis=0)
    neighbors_centered = neighbors_pos - centroid
    cov = np.sum([np.outer(n, n) for n in neighbors_centered], axis=0)
    eig_val, eig_vectors = np.linalg.eig(cov)
    eig_val, eig_vectors = np.real(eig_val), np.real(eig_vectors)
    normal = eig_vectors[np.argmin(eig_val)]
    if in_2d:
        viewing_dir = np.hstack((-np.dot(normal, centroid-drone.pos[:2])*normal, 0))
    else:
        viewing_dir = -np.dot(normal, centroid-drone.pos)*normal
    return viewing_dir