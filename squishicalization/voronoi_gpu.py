
import numpy as np
from matplotlib import pyplot as plt
from itertools import permutations
from scipy.signal import argrelextrema
from numba import cuda
from sympy.utilities.iterables import multiset_permutations
import math




@cuda.jit
def euclid_cuda_fast(array_in, mins, regs, seeds):
    
    pos = cuda.grid(1)

    x = array_in[pos][0]
    y = array_in[pos][1]
    z = array_in[pos][2]

    region = 0
    for seed in seeds:
        dist = math.sqrt((seed[0] - x)**2 + (seed[1] -y)**2 + (seed[2] -z)**2)
        if(mins[pos] > dist):
            mins[pos] = dist
            regs[pos] = region
        region +=1

    

@cuda.jit()
def multi_seed_min_polyhedral_dist_3point(indices, distances, regions, seeds, planes):
    eps = 0.00001
# def multi_seed_min_polyhedral_dist_3point(indices, distances, regions, seeds, planes, intpoints):
    pos = cuda.grid(1)
    # if(pos > indices.shape[0]):return
    region = 0
    for region in range(1,seeds.shape[0]+1):
        
        line_start=indices[pos]
        line_end=seeds[region]
        
        la_x = line_start[0]
        la_y = line_start[1]
        la_z = line_start[2]
        
        lb_x = line_end[0]
        lb_y = line_end[1]
        lb_z = line_end[2]

        lab_x = lb_x - la_x
        lab_y = lb_y - la_y
        lab_z = lb_z - la_z

        
        min_planar_dist = np.inf
        # if(la_x == lb_x and la_y == lb_y and la_z == lb_z):
        #     distances[pos]=0
        #     regions[pos]=region
        #     continue

        planeind = 0
        t_min = np.inf
        for plane in planes:  

            plane_0= plane[1]
            plane_1= plane[2]
            plane_2= plane[0]
            # plane_0= plane[2]
            # plane_1= plane[1]
            # plane_2= plane[0]

            p0_x = plane_0[0]+ la_x
            p0_y = plane_0[1]+ la_y
            p0_z = plane_0[2]+ la_z

            p1_x = plane_1[0]+ la_x
            p1_y = plane_1[1]+ la_y
            p1_z = plane_1[2]+ la_z
            
            p2_x = plane_2[0]+ la_x
            p2_y = plane_2[1]+ la_y
            p2_z = plane_2[2]+ la_z

            p02_x = p0_x - p2_x
            p02_y = p0_y - p2_y
            p02_z = p0_z - p2_z

            p01_x = p0_x - p1_x
            p01_y = p0_y - p1_y
            p01_z = p0_z - p1_z

            cross_01_02_x = p01_y * p02_z - p02_y * p01_z
            cross_01_02_y = p01_z * p02_x - p01_x * p02_z
            cross_01_02_z = p01_x * p02_y - p02_x * p01_y

            num = (cross_01_02_x * (la_x - p0_x)) + (cross_01_02_y * (la_y - p0_y)) + (cross_01_02_z * (la_z - p0_z))
            denom = (- lab_x * cross_01_02_x) + (- lab_y * cross_01_02_y) + (- lab_z * cross_01_02_z) 
            if(denom != 0):t = num/denom
            else: t = np.inf

            

            if(t>=0 and t <= 1):      
                intersect_x = la_x + t * lab_x 
                intersect_y = la_y + t * lab_y 
                intersect_z = la_z + t * lab_z 



                dist = math.sqrt((intersect_x -la_x)**2 + (intersect_y- la_y)**2 + (intersect_z-la_z)**2 )

                if(min_planar_dist > dist):
                    min_planar_dist=dist                    
                    best_int_x = intersect_x            
                    best_int_y = intersect_y            
                    best_int_z = intersect_z

                # intpoints[pos][region][planeind][0] = intersect_x
                # intpoints[pos][region][planeind][1] = intersect_y
                # intpoints[pos][region][planeind][2] = intersect_z
                # intpoints[pos][region][planeind][3] = math.sqrt(lab_x**2+lab_y**2+lab_z**2)/math.sqrt((intersect_x -la_x)**2 + (intersect_y- la_y)**2 + (intersect_z-la_z)**2 )
            planeind += 1
        
        if(math.sqrt(lab_x**2+lab_y**2+lab_z**2) != 0):
            dist = math.sqrt(lab_x**2+lab_y**2+lab_z**2)/math.sqrt((best_int_x -la_x)**2 + (best_int_y- la_y)**2 + (best_int_z-la_z)**2 )
        else:
            dist = 0

        if(distances[pos] > dist + eps):
            
            distances[pos]=dist
            regions[pos]=region

def get_surf(points):
    ac = points[2] - points[0]
    ab = points[1] - points[0]

    n = np.cross(ac,ab)
    n = n/np.linalg.norm(n)
    print(np.vstack((points,n)).shape)
    return np.vstack((points,n))

def get_btch():
    points = np.empty([0,3])

    points1 = np.array(list(permutations([math.sqrt(2), math.sqrt(2)/2,0])))
    points2 = np.array(list(permutations([-math.sqrt(2), math.sqrt(2)/2,0])))
    points3 = np.array(list(permutations([math.sqrt(2), -math.sqrt(2)/2,0])))
    points4 = np.array(list(permutations([-math.sqrt(2), -math.sqrt(2)/2,0])))

    # points = points1


    points = np.vstack((points1,points2,points3,points4))
    
    # plot points labelled
    # labels = [i for i in range(0,points.shape[0])]
    # plt = pv.Plotter()
    # plt.add_point_labels(points, labels)
    # plt.show()

    planes = []

    # 8 hexes
    planes.append(points[[0,2,4]])
    planes.append(points[[0,13,9]])
    planes.append(points[[1,3,17]])    
    planes.append(points[[8,22,23]])
    planes.append(points[[18,19,21]])
    planes.append(points[[6,14,16]])
    planes.append(points[[7,18,20]])
    planes.append(points[[4,14,6]])
    
    # 6 sqares
    planes.append(points[[2,16,14]])
    planes.append(points[[0,1,12]])
    planes.append(points[[8,10,20]])
    planes.append(points[[7,6,19]])
    planes.append(points[[3,5,15]])
    planes.append(points[[9,11,21]])


    # planes.append(points[[0,13,12]])
    # planes.append(points[[9,11,21]])
    # planes.append(points[[2,4,14]])
    # planes.append(points[[3,17,15]])
    # planes.append(points[[0,2,4]])
    # planes.append(points[[2,0,13]])
    # planes.append(points[[8,22,23]])
    # planes.append(points[[6,19,21]])
    # planes.append(points[[18,20,22]])
    # planes.append(points[[7,15,17]])


    # print(np.array(planes))
    return np.array(planes)

def get_test_poly():
    #cube 8 points
    p = np.array([[-1,1,-1],[1,1,-1],[1,1,1],[-1,1,1],[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]])
    p = p*1

    #cube 6 surfaces
    return np.array([[p[0],p[1],p[2]],[p[1],p[5],p[6]],[p[5],p[6],p[7]],[p[0],p[4],p[7]],[p[0],p[1],p[5]],[p[3],p[2],p[6]]])



def points_to_normals(plane_poits):
    return [ [np.cross(plane_poits[i,2] - plane_poits[i,0],plane_poits[i,1] - plane_poits[i,0]), plane_poits[i,0]] for i in range(0,plane_poits.shape[0])]

