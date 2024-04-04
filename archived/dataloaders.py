#!/usr/bin/env python3
from torch.utils.data import Dataset, DataLoader
from derivative import dxdt
import os
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import utilities as utils

class ScrewModel_Data_Preparer(Dataset):

    def __init__(self, path_to_directory, timehorizon = 200):
        """_summary_

        Args:
            path_to_directory (str): path to the directory where the dataset is present
        """
        super().__init__()
        
        self.X = []
        self.y = []
        self.path_to_directory = path_to_directory
        self.time_horizon = timehorizon
        self.robot_data_filename = "robot_data.csv"
        self.feature_data_filname = "feature_data.csv"


        self.feature_normalization = [1280,720]
        # self.wrench_normalization = [20,20,60,6,6,6]
        self.wrench_normalization = [3000,3000,3000,300,300,300]


        self.preprare_data()

    def preprare_data(self):
        subdirs = [x[1] for x in os.walk(self.path_to_directory)]
        subdirs = subdirs[0]
        for current_directory in subdirs:
            robot_filename  = self.path_to_directory + "/" + str(current_directory) + "/" +self.robot_data_filename
            feature_filename  = self.path_to_directory + "/" + str(current_directory) + "/" + self.feature_data_filname
            
            robot_data = pd.read_csv(robot_filename).to_dict('list')
            feature_data = pd.read_csv(feature_filename).to_dict('list')

            if(len(feature_data['timestamp'])< self.time_horizon):
                print("Data has less number of points that chosen time horizon, skipping: ", current_directory)
                continue

            for ids in range(len(feature_data['timestamp'])//self.time_horizon):
                start_index = ids * self.time_horizon
                end_index = start_index + self.time_horizon

                timestamps = np.array(feature_data['timestamp'][start_index:end_index])
                # pixel_x = np.array(feature_data['u'][start_index:end_index], dtype=np.float32)/self.feature_normalization[0]
                # pixel_y = np.array(feature_data['v'][start_index:end_index], dtype=np.float32)/self.feature_normalization[0]

                r = np.array(feature_data['r'][start_index:end_index], dtype=np.float32)
                theta = np.array(feature_data['theta'][start_index:end_index], dtype=np.float32)/np.pi

                dx = dxdt(r,timestamps, kind='finite_difference',k=5)     #Here k is window size
                dy = dxdt(theta,timestamps, kind='finite_difference',k=5)     #Here k is window size

                # dx = dxdt(pixel_x,timestamps, kind='kalman',alpha=5)
                # dy = dxdt(pixel_y,timestamps, kind='kalman',alpha=5)

                # dx = dxdt(pixel_x,timestamps, kind='finite_difference',k=5)     #Here k is window size
                # dy = dxdt(pixel_y,timestamps, kind='finite_difference',k=5)     #Here k is window size

                # dx = dxdt(pixel_x,timestamps, kind='trend_filtered',order=2,alpha=1)
                # dy = dxdt(pixel_y,timestamps, kind='trend_filtered',order=2,alpha=1)


                ###Collecting robot data

                pose_data = np.column_stack((np.asarray(robot_data['X'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Y'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Z'][start_index:end_index], dtype=np.float32),
                                                        np.asarray(robot_data['A'][start_index:end_index], dtype=np.float32)/np.pi,np.asarray(robot_data['B'][start_index:end_index], dtype=np.float32)/np.pi,np.asarray(robot_data['C'][start_index:end_index], dtype=np.float32)/np.pi))


                # wrench_data = np.column_stack((np.asarray(robot_data['Fx'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[0],np.asarray(robot_data['Fy'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[1],np.asarray(robot_data['Fz'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[2],
                #                                         np.asarray(robot_data['Tx'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[3],np.asarray(robot_data['Ty'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[4],np.asarray(robot_data['Tz'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[5]))


                wrench_data = np.column_stack((np.asarray(robot_data['Kx'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[0],np.asarray(robot_data['Ky'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[1],np.asarray(robot_data['Kz'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[2],
                                                        np.asarray(robot_data['Rot_Kx'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[3],np.asarray(robot_data['Rot_Ky'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[4],np.asarray(robot_data['Rot_Kz'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[5]))

                                    


                # Current_X = np.column_stack((pixel_x,pixel_y,pose_data,wrench_data)).astype(np.float32)
                Current_X = np.column_stack((r,theta,pose_data,wrench_data)).astype(np.float32)
                Current_y = np.column_stack((dx,dy)).astype(np.float32)

                self.X.append(Current_X)
                self.y.append(Current_y)
            
        
            
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        X = self.X[index]
        y = self.y[index]

        return [X,y]



class ScrewModel_Data_PreparerV2(Dataset):

    def __init__(self, path_to_directory, timehorizon = 200):
        """_summary_

        Args:
            path_to_directory (str): path to the directory where the dataset is present
        """
        super().__init__()
        
        self.X = []
        self.y = []
        self.path_to_directory = path_to_directory
        self.time_horizon = timehorizon
        self.robot_data_filename = "robot_data.csv"
        self.feature_data_filname = "feature_data.csv"


        self.stiffness_normalization = [3000,3000,3000,300,300,300]
        self.wrench_normalization = [50,50,50,10,10,10]        

        self.preprare_data()

    def preprare_data(self):
        subdirs = [x[1] for x in os.walk(self.path_to_directory)]
        subdirs = subdirs[0]
        for current_directory in subdirs:
            robot_filename  = self.path_to_directory + "/" + str(current_directory) + "/" +self.robot_data_filename
            feature_filename  = self.path_to_directory + "/" + str(current_directory) + "/" + self.feature_data_filname
            
            robot_data = pd.read_csv(robot_filename).to_dict('list')
            feature_data = pd.read_csv(feature_filename).to_dict('list')

            if(len(feature_data['timestamp'])< self.time_horizon):
                print("Data has less number of points that chosen time horizon, skipping: ", current_directory)
                continue

            for ids in range(len(feature_data['timestamp'])//self.time_horizon):
                start_index = ids * self.time_horizon
                end_index = start_index + self.time_horizon

                timestamps = np.array(feature_data['timestamp'][start_index:end_index])
                r = np.array(feature_data['r'][start_index:end_index], dtype=np.float32)
                r = r/np.max(np.abs(r))

                theta = np.array(feature_data['theta'][start_index:end_index], dtype=np.float32)/np.pi

                dr_dt = dxdt(r,timestamps, kind='finite_difference',k=1)     #Here k is window size
                dtheta_dt = dxdt(theta,timestamps, kind='finite_difference',k=1)     #Here k is window size
                
                # dr_dt = dxdt(r,timestamps, kind='kalman',alpha=2)
                # dtheta_dt = dxdt(theta,timestamps, kind='kalman',alpha=2)

                # dr_dt = dxdt(r,timestamps, kind='finite_difference',k=5)     #Here k is window size
                # dtheta_dt = dxdt(theta,timestamps, kind='finite_difference',k=5)     #Here k is window size

                # dr_dt = dxdt(r,timestamps, kind='trend_filtered',order=2,alpha=1)
                # dtheta_dt = dxdt(theta,timestamps, kind='trend_filtered',order=2,alpha=1)

                stiffness = np.column_stack((np.asarray(robot_data['Kx'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[0],np.asarray(robot_data['Ky'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[1],np.asarray(robot_data['Kz'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[2],
                                    np.asarray(robot_data['Rot_Kx'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[3],np.asarray(robot_data['Rot_Ky'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[4],np.asarray(robot_data['Rot_Kz'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[5]))

                damping = np.column_stack((np.asarray(robot_data['Cx'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Cy'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Cz'][start_index:end_index], dtype=np.float32),
                                    np.asarray(robot_data['Rot_Cx'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Rot_Cy'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Rot_Cz'][start_index:end_index], dtype=np.float32)))

                ###Collecting robot data
                orientation_data = np.column_stack((np.asarray(robot_data['A'][start_index:end_index], dtype=np.float32)/np.pi,np.asarray(robot_data['B'][start_index:end_index], dtype=np.float32)/np.pi,np.asarray(robot_data['C'][start_index:end_index], dtype=np.float32)/np.pi))


                wrench_data = np.column_stack((np.asarray(robot_data['Fx'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[0],np.asarray(robot_data['Fy'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[1],np.asarray(robot_data['Fz'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[2],
                                                        np.asarray(robot_data['Tx'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[3],np.asarray(robot_data['Ty'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[4],np.asarray(robot_data['Tz'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[5]))                    


                # Current_X = np.column_stack((pixel_x,pixel_y,pose_data,wrench_data)).astype(np.float32)
                Current_X = np.column_stack((r,theta,orientation_data,wrench_data,stiffness,damping)).astype(np.float32)
                Current_y = np.column_stack((dr_dt,dtheta_dt)).astype(np.float32)

                self.X.append(Current_X)
                self.y.append(Current_y)
            
        
            
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        X = self.X[index]
        y = self.y[index]

        return [X,y]




class ScrewRange_Data_Preparer(Dataset):

    def __init__(self, path_to_directory, timehorizon = 200):
        """_summary_

        Args:
            path_to_directory (str): path to the directory where the dataset is present
        """
        super().__init__()
        
        self.X = []
        self.y = []
        self.path_to_directory = path_to_directory
        self.time_horizon = timehorizon
        self.robot_data_filename = "robot_data.csv"
        self.feature_data_filname = "feature_data.csv"


        self.stiffness_normalization = [3000,3000,3000,300,300,300]
        self.wrench_normalization = [50,50,50,10,10,10]
        self.feature_normalization = [1280,720]

        self.r_norm = 50        

        self.preprare_data()

    def preprare_data(self):
        subdirs = [x[1] for x in os.walk(self.path_to_directory)]
        subdirs = subdirs[0]
        for current_directory in subdirs:
            robot_filename  = self.path_to_directory + "/" + str(current_directory) + "/" +self.robot_data_filename
            feature_filename  = self.path_to_directory + "/" + str(current_directory) + "/" + self.feature_data_filname
            
            robot_data = pd.read_csv(robot_filename).to_dict('list')
            feature_data = pd.read_csv(feature_filename).to_dict('list')

            if(len(feature_data['timestamp'])< self.time_horizon):
                print("Data has less number of points that chosen time horizon, skipping: ", current_directory)
                continue

            start_index = 0
            end_index = start_index + self.time_horizon

            r = np.array(feature_data['r'][start_index:end_index], dtype=np.float32)
            
            r = r/self.r_norm

            [lower, upper] = utils.find_mean_std_dev(r)
            pixel_x = np.array(feature_data['u'][start_index:end_index], dtype=np.int32)/self.feature_normalization[0]
            pixel_y = np.array(feature_data['v'][start_index:end_index], dtype=np.int32)/self.feature_normalization[1]

            xc,yc,a,b,theta = utils.fit_ellipse_skimage(pixel_x,pixel_y)

            stiffness = np.column_stack((np.asarray(robot_data['Kx'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[0],np.asarray(robot_data['Ky'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[1],np.asarray(robot_data['Kz'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[2],
                                np.asarray(robot_data['Rot_Kx'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[3],np.asarray(robot_data['Rot_Ky'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[4],np.asarray(robot_data['Rot_Kz'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[5]))

            damping = np.column_stack((np.asarray(robot_data['Cx'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Cy'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Cz'][start_index:end_index], dtype=np.float32),
                                np.asarray(robot_data['Rot_Cx'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Rot_Cy'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Rot_Cz'][start_index:end_index], dtype=np.float32)))

            ###Collecting robot data
            orientation_data = np.column_stack((np.asarray(robot_data['A'][start_index:end_index], dtype=np.float32)/np.pi,np.asarray(robot_data['B'][start_index:end_index], dtype=np.float32)/np.pi,np.asarray(robot_data['C'][start_index:end_index], dtype=np.float32)/np.pi))


            Current_X = np.column_stack((orientation_data,stiffness,damping)).astype(np.float32)
            Current_y = np.column_stack((a,b,theta)).astype(np.float32)
            
            self.X.append(Current_X)
            self.y.append(Current_y)
            
        
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        X = self.X[index]
        y = self.y[index]

        return [X,y]



class ScrewMeanModel_Data_Preparer(Dataset):

    def __init__(self, path_to_directory):
        """_summary_

        Args:
            path_to_directory (str): path to the directory where the dataset is present
        """
        super().__init__()
        
        self.X = []
        self.y = []
        self.path_to_directory = path_to_directory
        self.robot_data_filename = "robot_data.csv"
        self.feature_data_filname = "feature_data.csv"


        self.stiffness_normalization = [3000,3000,3000,300,300,300]

        

        self.preprare_data()

    def preprare_data(self):
        subdirs = [x[1] for x in os.walk(self.path_to_directory)]
        subdirs = subdirs[0]
        data_count = 1
        for current_directory in subdirs:
            robot_filename  = self.path_to_directory + "/" + str(current_directory) + "/" +self.robot_data_filename
            feature_filename  = self.path_to_directory + "/" + str(current_directory) + "/" + self.feature_data_filname
            
            robot_data = pd.read_csv(robot_filename).to_dict('list')
            feature_data = pd.read_csv(feature_filename).to_dict('list')

            start_index = 0
            end_index = start_index + len(feature_data['timestamp'])

            timestamps = np.array(feature_data['timestamp'][start_index:end_index])
            
            r = np.array(feature_data['r'][start_index:end_index], dtype=np.float32)
            r_mean = np.mean(r)
            theta = np.array(feature_data['theta'][start_index:end_index], dtype=np.float32)/np.pi

            dr_dt = dxdt(r,timestamps, kind='finite_difference',k=1)     #Here k is window size
            dtheta_dt = dxdt(theta,timestamps, kind='finite_difference',k=1)     #Here k is window size

            # print(dtheta_dt)
            # fig = plt.figure()
            # ax = plt.subplot(111)
            # ax.plot(timestamps,dtheta_dt,'bo')

            # # plt.show()
            # fig.savefig(self.path_to_directory+"/r_"+str(data_count)+".jpg")
            # fig = plt.figure()
            # ax = plt.subplot(111)
            # ax.plot(timestamps,r,'bo')
            # # plt.show()
            # fig.savefig(self.path_to_directory+"/theta_"+str(data_count)+".jpg")

            # fig = plt.figure()
            # ax = plt.subplot(111)
            # ax.plot(timestamps,dtheta_dt,'bo')
            # # plt.show()
            # fig.savefig(self.path_to_directory+"/dtheta_dt_"+str(data_count)+".jpg")

            # fig = plt.figure()
            # ax = plt.subplot(111)
            # ax.plot(timestamps,dr_dt,'bo')
            # # plt.show()
            # fig.savefig(self.path_to_directory+"/drdt_"+str(data_count)+".jpg")
            
            ###Collecting robot data

            data = np.column_stack((np.asarray(robot_data['Kx'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[0],np.asarray(robot_data['Ky'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[1],np.asarray(robot_data['Kz'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[2],
                                    np.asarray(robot_data['Rot_Kx'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[3],np.asarray(robot_data['Rot_Ky'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[4],np.asarray(robot_data['Rot_Kz'][start_index:end_index], dtype=np.float32)/self.stiffness_normalization[5],
                                    np.asarray(robot_data['Cx'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Cy'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Cz'][start_index:end_index], dtype=np.float32),
                                    np.asarray(robot_data['Rot_Kx'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Rot_Ky'][start_index:end_index], dtype=np.float32),np.asarray(robot_data['Rot_Cz'][start_index:end_index], dtype=np.float32)))

                                

            orientation_data = np.column_stack((np.asarray(robot_data['A'][start_index:end_index], dtype=np.float32)/np.pi,np.asarray(robot_data['B'][start_index:end_index], dtype=np.float32)/np.pi,np.asarray(robot_data['C'][start_index:end_index], dtype=np.float32)/np.pi))
            
            Current_y = r_mean

            self.X.append(data)
            self.y.append(Current_y)
            data_count += 1
            
        
            
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        X = self.X[index]
        y = self.y[index]

        return [X,y]


