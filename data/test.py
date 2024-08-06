from dataloader import PointCloudDataset, StaticDataset
import numpy as np
from transforms import *
from create_open3d_obj import *
import copy

def norm_test(cube):
    norm = Normalize()
    invnorm = DeNormalize()
    
    cube1 = copy.deepcopy(cube) #original
    cube2 = norm(copy.deepcopy(cube1))#normalized
    cube3 = invnorm(copy.deepcopy(cube2))#denormalized

    obj1 = open_cube(cube1)
    obj2 = open_cube(cube2)
    obj3 = open_cube(cube3)

    lb1, ub1 = obj1.get_min_bound(), obj1.get_max_bound()
    lb2, ub2 = obj2.get_min_bound(), obj2.get_max_bound()
    lb3, ub3 = obj3.get_min_bound(), obj3.get_max_bound()
    
    error1 = obj1.compute_point_cloud_distance(obj3)
    error2 = obj3.compute_point_cloud_distance(obj1)
    
    merror1 = np.mean(error1)
    merror2 = np.mean(error2)
    #check if denorm worked
    if merror1 > 0 or merror2 > 0:
        print(merror1, merror2, "PCD is > 0 for a cube after Normalization and Denormalization")
    #check if bounds not changed
    if any(lb1!=lb3):
        print(ub1,ub2, "Lower-Boundingbox-Error")
    if any(ub1!=ub3):
        print(ub1,ub2, "Upper-Boundingbox-Error")
    #check if normalization worked (bounds changed?)    
    if any(lb2<-1) or any(ub2>1):
        print(lb2,ub2, "cube not normalized")


def shift_test(cube):
    randomvec = np.random.rand(3)
    randomvecback = -1*randomvec
    move = Shift(randomvec)
    moveback = Shift(randomvecback)

    cube1 = copy.deepcopy(cube)
    cube2 = move(copy.deepcopy(cube1))
    cube3 = moveback(copy.deepcopy(cube2))

    obj1 = open_cube(cube1)
    obj2 = open_cube(cube2)
    obj3 = open_cube(cube3)

    error1 = obj1.compute_point_cloud_distance(obj3)
    error2 = obj3.compute_point_cloud_distance(obj1)

    merror1 = np.mean(error1)
    merror2 = np.mean(error2)
    #check if shift worked
    if merror1 > 0.001 or merror2 > 0.001:#floatingpoint error
        print(merror1, merror2, "shift screwed up")
        raise ValueError("Check here")
    
    ###checks if shift does something at all

    error1 = obj1.compute_point_cloud_distance(obj2)
    error2 = obj2.compute_point_cloud_distance(obj1)

    merror1 = np.mean(error1)
    merror2 = np.mean(error2)

    if merror1 <= 0 or merror2 <= 0:
        print(merror1, merror2, "shift does nothing at all")
        raise ValueError("Check here")


def scl_test(cube):
    randomvec = np.random.rand(3)
    randomvecback = [1/randomvec[0], 1/randomvec[1], 1/randomvec[2]]
    scale = Scale(randomvec)
    scaleback = Scale(randomvecback)

    cube1 = copy.deepcopy(cube)
    cube2 = scale(copy.deepcopy(cube1))
    cube3 = scaleback(copy.deepcopy(cube2))

    obj1 = open_cube(cube1)
    obj2 = open_cube(cube2)
    obj3 = open_cube(cube3)

    error1 = obj1.compute_point_cloud_distance(obj3)
    error2 = obj3.compute_point_cloud_distance(obj1)


    merror1 = np.mean(error1)
    merror2 = np.mean(error2)
    #check if scale and scaleback worked
    if merror1 > 0.0001 or merror2 > 0.0001:
        print(merror1, merror2, "Scale screwed up")
        raise ValueError("Check here")
    
    error1 = obj1.compute_point_cloud_distance(obj2)
    error2 = obj2.compute_point_cloud_distance(obj1)


    merror1 = np.mean(error1)
    merror2 = np.mean(error2)
    #check if scale changed anything
    if merror1 <= 0 or merror2 <= 0:
        print(merror1, merror2, "scale did nothing at all")
        raise ValueError("Check here")


def voxelise_test(cube):
    tovoxel = Voxelise(64)
    toPC = DeVoxelise()
    cube1 = copy.deepcopy(cube)
    cube2 = tovoxel(copy.deepcopy(cube1))
    cube3 = toPC(copy.deepcopy(cube2))

    obj1 = open_cube(cube1)
    obj3 = open_cube(cube3)

    error1 = obj1.compute_point_cloud_distance(obj3)
    error2 = obj3.compute_point_cloud_distance(obj1)

    merror1 = np.mean(error1)
    merror2 = np.mean(error2)
    #check if denorm worked
    if merror1 > 0 or merror2 > 0:
        print(merror1, merror2, "voxelise screwed up")
        raise ValueError("Check here")
    print(merror1, merror2)


def rotate_test(cube):
    randomvec = 90
    rot = Rotate('z',randomvec)
    rotback = Rotate('z',-randomvec)

    cube1 = copy.deepcopy(cube)
    cube2 = rot(copy.deepcopy(cube1))
    cube3 = rotback(copy.deepcopy(cube2))

    obj1 = open_cube(cube1)
    obj2 = open_cube(cube2)
    obj3 = open_cube(cube3)

    error1 = obj1.compute_point_cloud_distance(obj3)
    error2 = obj3.compute_point_cloud_distance(obj1)


    merror1 = np.mean(error1)
    merror2 = np.mean(error2)
    #check if rotate and rotateback worked
    if merror1 > 0.0001 or merror2 > 0.0001:
        print(merror1, merror2, "rotate screwed up")
        raise ValueError("Check here")
    
    error1 = obj1.compute_point_cloud_distance(obj2)
    error2 = obj2.compute_point_cloud_distance(obj1)


    merror1 = np.mean(error1)
    merror2 = np.mean(error2)
    #check if rotate changed anything
    if merror1 <= 0 or merror2 <= 0:
        print(merror1, merror2, "rotate did nothing at all")
        raise ValueError("Check here")


def crop_test(cube):
    crop = Crop([0,48,0,48,0,48], random=True, randomSize=False, inverse=True)
    
    cube1 = copy.deepcopy(cube)
    cube2 = crop(copy.deepcopy(cube1))
    
    obj1 = open_cube(cube1)
    if len(cube2['points'])==0:
        return print("cube ist nach dem crop leer")
    obj2 = open_cube(cube2)

    error1 = obj1.compute_point_cloud_distance(obj2)

    merror1 = np.mean(error1)
    #check if crop changed anything
    if merror1 == 0:
        print("crop didnt crop anything")
        print(crop.dimensions)
        #raise ValueError("Check here")
    
def universal_test(cube):
    randomvec = np.random.randint([1, 1, 1],10)
    randomvec = np.random.rand(3)
    randomvecinv = [1/randomvec[0], 1/randomvec[1], 1/randomvec[2]]
    
    norm  = Normalize()
    denorm = DeNormalize()

    move = Shift(randomvec)
    moveback = Shift(-randomvec)
    
    scl = Scale(randomvec)
    sclback = Scale(randomvecinv)
    
    rot = Rotate('y',randomvec[0])
    rotback = Rotate('y', -randomvec[0])

    cube1 = copy.deepcopy(cube)
    cube2 = norm(scl(move(rot(copy.deepcopy(cube1)))))
    cube3 = rotback(moveback(sclback(denorm(copy.deepcopy(cube2)))))
                    

    obj1 = open_cube(cube1)
    obj3 = open_cube(cube3)

    error1 = obj1.compute_point_cloud_distance(obj3)
    error2 = obj3.compute_point_cloud_distance(obj1)


    merror1 = np.mean(error1)
    merror2 = np.mean(error2)
    if merror1>0.001 or merror2>0.001:
        print("something went wrong, check each transform itself")
        raise ValueError("Check here")

if __name__ == "__main__":
    trainset2 = PointCloudDataset("./data/dataset_dev", split="train")
    testset2 = PointCloudDataset("./data/dataset_dev", split="test")
    for i in range(100):
        print(i,'/',100)
        universal_test(testset2[2]['cube{:05d}'.format(100)])