import pyvista as pv
import numpy as np
import cv2
from    IPython import embed
import time
import matplotlib.pyplot as plt

from pyvista import examples

# import vtk
import transforms3d as t3d
import yaml
 

modelTransform = np.array([
                          [1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]
                        ], dtype=np.float32)



# plotter.camera.SetModelTransformMatrix(trans_to_matrix(modelTransform))
def rvec_tvec2mat(rvec,tvec):
    
    rvec = rvec.flatten()
    tvec = tvec.flatten()
    T = np.eye(4)
    R = cv2.Rodrigues(rvec)[0]
    T[0:3,0:3] = R
    T[0:3, -1] = tvec

    return T

class ArucoAR(object):
    """docstring for ArucoAR"""
    def __init__(self, ):
        super(ArucoAR, self).__init__()
        self.debug = False

    def set_mesh(self, filename=None, levitate=True, scale = 10):

        if filename is None:
            filename = examples.planefile

        mesh = pv.read(filename)

        # mesh.points = mesh.points - mesh.center
        np_arr = np.array(mesh.points)
        centroid = np_arr.mean(axis=0)
        lowest_z = np_arr[np.argmin(mesh.points[:,-1]), :]

        print(centroid,"centroid")
        print(lowest_z,"lowest_z")
        # np_arr -= centroid 
        np_arr -= lowest_z 

   
        max_dist = np.linalg.norm(np_arr, axis=1).max()

        if levitate:
            mesh.points = scale*np_arr/max_dist + np.array([0,0,scale/2])
        else:
            mesh.points = scale*np_arr/max_dist 

        # embed()
        self.mesh = mesh

        return mesh 

    def set_bg_image(self, img_bg):
        
        self.img_bg = img_bg   
        self.window_size = (img_bg.shape[1], img_bg.shape[0])

    def combine_images(self, img_fg, mask_fg, img_bg):
        bg_mask = 1 - mask_fg
        img_comb = np.multiply(img_fg, mask_fg) + \
                    np.multiply(img_bg, bg_mask)

        return img_comb

    def get_rendered_img(self, p, T=None):
 
        if T is not None: 
            p.camera.SetFocalPoint(0,0,1)
            p.camera.SetPosition(0, 0,0 )
            p.camera.SetViewUp(0,-1,0)
            p.camera_set = True
            mat_y_rot = np.eye(4)
            # R_y_rot = t3d.axangles.axangle2mat([1,0,0],np.pi/2)
            R_y_rot = t3d.axangles.axangle2mat([1,0,0],0*np.pi/2)
            mat_y_rot[0:3,0:3]=R_y_rot

            p.camera.SetModelTransformMatrix( pv.vtkmatrix_from_array(T@mat_y_rot))


        tic = time.time()
        _,img = p.show(auto_close=False,screenshot=True, window_size=self.window_size)
        toc = time.time()
        print('render time : {}\n'.format(toc-tic))
        
        background_255 = p.background_color[0]*255
        background_bool = np.isclose(img[:,:,0], background_255 ,atol=1.0, rtol=0.0)
        foreground_bool = 1 - background_bool
        

        foreground_mask = foreground_bool.astype(np.uint8)

        foreground_mask_col = np.dstack((foreground_mask,foreground_mask,foreground_mask))

        if self.debug:

            cv2.namedWindow("rendered_image", cv2.WINDOW_NORMAL)  
            cv2.imshow("rendered_image",img)

            cv2.namedWindow("foreground_mask_col", cv2.WINDOW_NORMAL)  
            cv2.imshow("foreground_mask_col",foreground_mask_col*255)
            cv2.waitKey(0)

        return img, foreground_mask_col


def modify_tranf_mat(transf_mat, count):
    transf_mat = transf_mat.copy()
    
    # pert_R = t3d.axangles.axangle2mat([0,1,0],np.pi/4*(count%10 - 5)/10)
    pert_R = t3d.axangles.axangle2mat([0,1,0],np.pi/4*count/10)
    pert_mat = np.eye(4)
    pert_mat[0:3,0:3] = pert_R

    # transf_mat = pert_mat@transf_mat
    transf_mat = transf_mat@pert_mat
    # transf_mat [2,-1] = -5 + np.sin(count/10)

    return transf_mat

def main():     

    # device = 'webcam'
    device = 'android'

    if device == 'webcam':
        calib_file = './data/hd310.yaml'
    elif device == 'android':
        calib_file = './data/android/calib_android.yaml'
    else:
        print('no calib data')


    with open(calib_file) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    
    # img_bg = cv2.imread('./data/sample.png')

    mtx = np.array(params['camera_matrix'])
    dist = np.array(params['dist_coefs'])

    ar_ar = ArucoAR()
    # ar_ar.set_mesh('./bunny.ply')
    # ar_ar.set_mesh('./data/bun_zipper.ply', levitate=True)
    # ar_ar.set_mesh('./data/airplane/F-22 Raptor v1.stl', levitate=True)
    ar_ar.set_mesh('./data/airplane/rafale.stl', levitate=True)
    # ar_ar.set_mesh('./data/star_wars/falcon.stl')
    # ar_ar.set_mesh('./data/uke_obj/ukelele.obj')
    # tex = examples.download_masonry_texture()
    tex = examples.download_sky_box_nz_texture()

    # tex = cv2.imread('./data/uke_obj/some.jpg')
    # ar_ar.set_mesh()


    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('./data/android/android_auto_focus_off.mp4')
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
    p = pv.Plotter(off_screen=True) 

    try:
        p.add_mesh(ar_ar.mesh, show_edges=False, texture=tex)
    except:
        p.add_mesh(ar_ar.mesh, show_edges=False)

    # light = pv.Light(color='white', light_type='headlight')
    # light.SetLightTypeToSceneLight()
    # p.add_light(light)
    light = pv.Light(position=(10, 0, -0), light_type='camera light')
    p.add_light(light)

    count = 0
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    para = cv2.aruco.DetectorParameters_create()
    para.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    video_imgs = [] 
    cv2.namedWindow('comb_img', cv2.WINDOW_NORMAL)
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        try:
            # Display the resulting comb_img
            if ret:
                cv2.imshow('comb_img',comb_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:   
                break
        except:
            print('issue in displaying')


        ar_ar.set_bg_image(frame)

        # cam_transf_mat = modify_tranf_mat (cam_transf_mat_init,count)
        # cam_transf_mat = cam_transf_mat_init
        
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame,dictionary, parameters=para)
        comb_img = frame
        if np.all(ids != None):
            rvec, tvec ,_ = cv2.aruco.estimatePoseSingleMarkers(corners, 5, mtx, dist)
            # print(rvec,"rvec")
            for i in range(0, ids.size):

                T = rvec_tvec2mat(rvec[i], tvec[i])
                img_fg, mask_fg = ar_ar.get_rendered_img(p, T)
                comb_img = ar_ar.combine_images(img_fg, mask_fg, ar_ar.img_bg)

                video_imgs.append(comb_img)
                # draw axis for the aruco markers
                # cv2.aruco.drawAxis(comb_img, mtx, dist, rvec[i], tvec[i], 10)
                # cv2.aruco.drawDetectedMarkers(comb_img, corners)


        count += 1
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    out = cv2.VideoWriter('./video_temp.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, (frame.shape[1],frame.shape[0]))
 
    for i in range(len(video_imgs)):
        out.write(video_imgs[i])
    out.release()


if __name__ == '__main__':
    main()