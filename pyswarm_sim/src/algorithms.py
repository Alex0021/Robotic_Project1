import numpy as np
from scipy.spatial import ConvexHull
from typing import TYPE_CHECKING
import alphashape
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon

if TYPE_CHECKING:
    from pyswarm_sim.src.drone import Drone, DroneNeighbor


def get_viewing_dir(drone, neighbors: list['DroneNeighbor'], algo: str, **params):
    """
    Compute the viewing direction of the drone based on the neighbors
    with the selected algorithm.

    Args:
        drone: The selected drone
        neighbors: The neighbors of the drone
        algo: The algorithm to use to compute the viewing direction
    """
    if len(neighbors) == 0:
        return drone.get_heading()
    algo = algo.lower()
    # If algo is NONE, return the current heading
    if algo.upper() == "NONE":
        return drone.get_heading()
    assert algo.upper() in ["AVERAGE", "OUTER", "TANGENT_PLANE", "CONVEX_HULL", "ALPHA_SHAPE"], "Algorithm {0} not supported".format(algo)
    vd = eval(algo)(drone, neighbors, params)
    # Check valid vd and normalize
    norm = np.linalg.norm(vd)
    if norm > 0:
        return vd / norm
    else:
        return drone.get_heading()

def average(drone: 'Drone', neighbors: list['DroneNeighbor'], params: dict):
    """
    Compute the desired viewing direction based on the centroid of the neighbors.

    1. Find the centroid (mean) of the neighbors
    2. Set the viewing direction to the opposite of the centroid

    Args:
        drone: The selected drone
        neighbors: The neighbors of the drone
        params: {'in_2d': True or False}
    """
    centroid = np.mean([n.get_abs_pos() for n in neighbors], axis=0)
    vec_to_centroid = centroid - drone.pos
    viewing_dir = -vec_to_centroid
    # If 2D, set z component to zero
    if params.get('in_2d', False):
        viewing_dir[2] = 0
    return viewing_dir

def outer(drone: 'Drone', neighbors: list['DroneNeighbor'], params: dict):
    """
    Compute the desired viewing direction based on the outer product of the neighbors.

    1. Find combinations of neighbors depending on n_points
    2. Calculate pair that gives highest angle/area/volume
    3. Set the viewing direction to the opposite of the average of the pair

    Args:
        drone: The selected drone
        neighbors: The neighbors of the drone
        params: {'n_points': int}
    """
    # Get number of points to compute estimate
    n_points = params.get('n_points', 2)
    in_2d = params.get('in_2d', False)
    if n_points < 2:
        raise ValueError("n_points must be greater than 1")
    else:
        if n_points > len(neighbors)+1:
            print("WARNING :: n_points must be less or equal to the number of neighbors+1")
            return drone.get_heading()
        
    match n_points:
        case 2: # 2D case: Max angle
            neighbors_pos = np.array([n.get_abs_pos() for n in neighbors])
            dists = neighbors_pos - drone.pos
            # Normalize distances
            dists = dists / np.linalg.norm(dists, axis=1)[:, np.newaxis]
            if in_2d:
                dists = dists[:, :2]
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
        case 3: # 3D case w/ triangles: Find max area
            neighbors_pos = np.array([n.get_abs_pos() for n in neighbors])
            dists = neighbors_pos - drone.pos
            if in_2d:
                dists = dists[:, :2]
            # Calculate area of each triangle using the norm of the cross product
            highest = -np.inf
            highest_indices = (0, 0)
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    # Check area
                    a = np.linalg.norm(np.cross(dists[i], dists[j]))/2
                    if a > highest:
                        highest = a
                        highest_indices = (i, j)
            viewing_dir = (dists[highest_indices[0]] + dists[highest_indices[1]]) / 2
        case _: # 3D case w/ convex hull: Find max volume
            indices = combinations(len(neighbors), n_points-1)
            points = np.array([n.get_abs_pos() for n in neighbors])
            drone_pos = drone.pos.copy()
            if in_2d:
                points = points[:, :2]
                drone_pos = drone_pos[:2]
            highest_volume = 0
            highest_indices = None
            for idx in indices:
                hull = ConvexHull(np.concatenate((points[idx], [drone_pos])))
                if in_2d:
                    if hull.area > highest_volume:
                        highest_volume = hull.area
                        highest_indices = idx
                else:
                    if hull.volume > highest_volume:
                        highest_volume = hull.volume
                        highest_indices = idx
            # Compute viewing direction
            centroid = np.mean(np.concatenate((points[highest_indices], [drone_pos])), axis=0)
            viewing_dir = centroid - drone_pos

    viewing_dir = -viewing_dir
    if in_2d:
        viewing_dir = np.hstack((viewing_dir, 0))
    return viewing_dir


def tangent_plane(drone: 'Drone', neighbors: list['DroneNeighbor'], params: dict):
    """
    Compute the desired viewing direction based on the normal 
    of a tangent plane minimizing the distance to every neighbors.

    1. Find the centroid (mean) of the neighbors
    2. Compute covariance matrix of the neighbors
    3. Apply PCA and keep the last eigenvector (with smallest eigenvalue)
    4. This vector gives the normal to the tangent plane of the surface
    5. Set the viewing direction to the opposite of the normal

    Args:
        drone: The selected drone
        neighbors: The neighbors of the drone
        params: {'in_2d': True or False}
    """
    neighbors_pos = np.array([n.get_abs_pos() for n in neighbors])
    in_2d = params.get('in_2d', False)
    if in_2d:
        neighbors_pos = neighbors_pos[:, :2]
    centroid = np.mean(neighbors_pos, axis=0)
    neighbors_centered = neighbors_pos - centroid
    cov = np.sum([np.outer(n, n) for n in neighbors_centered], axis=0)
    eig_val, eig_vectors = np.linalg.eig(cov)
    eig_val, eig_vectors = np.real(eig_val), np.real(eig_vectors)
    sorted_indices = np.argsort(eig_val)
    normal = eig_vectors[:,sorted_indices[0]]
    # Verify if eigenvalues are close to each other (2 directions are good)
    if abs(eig_val[sorted_indices[1]] - eig_val[sorted_indices[0]]) < 0.01:
        print("WARNING :: Eigenvalues are too close to each other, averaging the two smallest")
        normal = np.mean(eig_vectors[:,sorted_indices[:2]], axis=0)
    if in_2d:
        viewing_dir = np.hstack((-np.sign(np.dot(normal, centroid-drone.pos[:2]))*normal, 0))
    else:
        viewing_dir = -np.sign(np.dot(normal, centroid-drone.pos))*normal
    return viewing_dir

def convex_hull(drone: 'Drone', neighbors: list['DroneNeighbor'], params: dict):
    """
    Compute the desired viewing direction based on the convex hull of the neighbors.

    1. Find the convex hull of the neighbors + the drone
    2. Set the viewing direction either to the normal of the adjacent faces
       or to the one of the visible faces

    Args:
        drone: The selected drone
        neighbors: The neighbors of the drone
        params: {'faces': 'adjacent' or 'visible', 'in_2d': True or False}
    """
    # Check if # neighbors is enough
    in_2d = params.get('in_2d', False)
    if len(neighbors) < 2:
        print("WARNING :: Not enough neighbors to compute convex hull")
        return drone.get_heading()
    elif len(neighbors) == 2:
        # Do outer2 metric (max angle)
        neighbors_pos = np.array([n.get_abs_pos() for n in neighbors])
        dists = neighbors_pos - drone.pos
        if in_2d:
            dists = dists[:, :2]
        # Only 2 neighbors, so take average of the two
        viewing_dir = -(dists[0] + dists[1]) / 2
    else:
        # Convex hull
        points = np.array([n.get_abs_pos() for n in neighbors])
        idx_drone = len(neighbors)
        # Prepare standard options for qhull solver
        qhull_options = '' if in_2d else 'tJ'
        ndim = 2 if in_2d else 3
        if in_2d:
            points = points[:, :2]
        if params.get('faces', 'adjacent') == 'adjacent':
            points = np.concatenate((points, [drone.pos[:ndim]]))
            hull = ConvexHull(points, qhull_options=f'Q{qhull_options}')
            # Check if drone is in the convex hull
            if idx_drone not in hull.vertices:
                return drone.get_heading()
            # Compute the normal of the adjacent faces
            adj_idx = np.where(hull.simplices == idx_drone)[0]
            normals = hull.equations[adj_idx, :ndim]
            # Compute the viewing direction
            viewing_dir = np.mean(normals, axis=0)
        elif params.get('faces', 'adjacent') == 'visible':
            # Edge case with 3 neighbors
            if len(neighbors) == 3 and not in_2d:
                centroid = np.mean(points[:3], axis=0)
                normal = np.cross(points[1] - points[0], points[2] - points[0])
                viewing_dir = -np.sign(np.dot(normal, centroid - points[-1]))*normal
            else:
                hull = ConvexHull(points, qhull_options=f'Q{qhull_options}')
                # Compute the centroid of each face
                hull_centroids = np.mean(points[hull.simplices], axis=1)
                # Find closest to the drone
                closest_idx = np.argmin(np.linalg.norm(hull_centroids - drone.pos[:ndim], axis=1))
                closest_normal = hull.equations[closest_idx, :ndim]
                visible_normals = []
                for i in range(len(hull.equations)):
                    if np.dot(hull.equations[i, :ndim], closest_normal) > 0:
                        visible_normals.append(hull.equations[i, :ndim])
                # Compute the viewing direction
                viewing_dir = np.mean(visible_normals, axis=0)
        else:
            raise ValueError("Invalid value for 'faces' in params")
    
    # Add z component to zero if in 2D
    if in_2d:
        viewing_dir = np.hstack((viewing_dir, 0))
    return viewing_dir

def alpha_shape(drone: 'Drone', neighbors: list['DroneNeighbor'], params: dict):
    """
    Compute the desired viewing direction based on the alpha shape of the neighbors.

    1. Find the alpha shape of the neighbors + the drone
    2. Set the viewing direction either to the normal of the adjacent edges
       or to the one of the visible edges

    Args:
        drone: The selected drone
        neighbors: The neighbors of the drone
        params: {'faces': 'adjacent' or 'visible', 'in_2d': True or False}
    """
    d_ref = 1.5
    in_2d = params.get('in_2d', False)
    if len(neighbors) < 2:
        print("WARNING :: Not enough neighbors to compute alpha shape")
        return drone.get_heading()
    elif len(neighbors) == 2:
        # Do outer2 metric (max angle)
        neighbors_pos = np.array([n.get_abs_pos() for n in neighbors])
        dists = neighbors_pos - drone.pos
        if in_2d:
            dists = dists[:, :2]
        # Only 2 neighbors, so take average of the two
        viewing_dir = -(dists[0] + dists[1]) / 2
    else:
        # Alpha shape
        points = np.array([n.get_abs_pos() for n in neighbors])
        if in_2d:
            points = points[:, :2]
            ndim = 2
        else:
            raise NotImplementedError("Not tested for 3D")
        # Add the drone's position
        points = np.vstack((points, drone.pos[:ndim]))

        # Compute the alpha shape
        alpha_shape = alphashape.alphashape(points, alpha=d_ref)
        
        # Get the drone's coordinates
        drone_coords = drone.pos[:ndim]
        
        if alpha_shape is None:
            print("WARNING :: Alpha shape is None")
            return drone.get_heading()
        elif isinstance(alpha_shape, (Polygon, MultiPolygon)):
            # Get boundary coordinates
            if isinstance(alpha_shape, Polygon):
                boundaries = [alpha_shape.exterior]
            else:  # MultiPolygon
                boundaries = [poly.exterior for poly in alpha_shape.geoms]
            # Collect all boundary coordinates
            boundary_coords_list = [np.array(boundary.coords) for boundary in boundaries]
        elif isinstance(alpha_shape, (LineString, MultiLineString)):
            if isinstance(alpha_shape, LineString):
                lines = [alpha_shape]
            else:  # MultiLineString
                lines = list(alpha_shape.geoms)
            # Collect all line coordinates
            boundary_coords_list = [np.array(line.coords) for line in lines]
        else:
            # Alpha shape is not a Polygon or LineString, cannot proceed
            print("WARNING :: Alpha shape is not a Polygon or LineString")
            # Print the type of the alpha shape
            print(type(alpha_shape))
            return drone.get_heading()

        # Find indices where the drone's position matches boundary coordinates
        indices = []
        for boundary_coords in boundary_coords_list:
            idx = np.where(np.all(boundary_coords == drone_coords, axis=1))[0]
            indices.extend(idx)
        num_coords = len(boundary_coords)

        if len(indices) == 0:
            # Drone is not on the boundary
            # Print the position of the drone
            print("Drone not on boundary, position:", drone_coords)
            drone.set_boundary_estimate(False)
            return drone.get_heading()
        else:
            # Drone is on the boundary
            drone.set_boundary_estimate(True)
        
        # Compute the normals of the adjacent edges
        normals = []
        for idx in indices:
            prev_idx = (idx - 1) % num_coords
            next_idx = (idx + 1) % num_coords
            vec_prev = boundary_coords[idx] - boundary_coords[prev_idx]
            vec_next = boundary_coords[next_idx] - boundary_coords[idx]
            normal_prev = np.array([-vec_prev[1], vec_prev[0]])
            normal_next = np.array([-vec_next[1], vec_next[0]])
            normals.append(normal_prev)
            normals.append(normal_next)
            
        # Compute the viewing direction
        viewing_dir = np.mean(normals, axis=0)
        viewing_dir /= np.linalg.norm(viewing_dir)

    # Add z component as zero if in 2D
    if in_2d:
        viewing_dir = np.hstack((viewing_dir, 0))
    return viewing_dir





def combinations(n, k):
   result = []

   def backtrack(start, current_combination):
      if len(current_combination) == k:
         result.append(current_combination)
         return

      for i in range(start, n):
         backtrack(i + 1, current_combination + [i])

   backtrack(0, [])
   return result