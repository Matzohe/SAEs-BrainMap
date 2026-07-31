import torch
import numpy as np
import pickle
import nibabel as nib
from typing import List, Optional, Union
from tqdm import tqdm
import pandas as pd
import os

from .NSD_utils import zscore_by_run, ev
from ...util import check_path, check_tensor


class NSDDataset:
    def __init__(self, args):
        # Loading Config from config path or dataclass
        
        # list the parameters that have been used in the code
        # all the parameters is comming from the config dataclass
        self.nsd_root = args.NSD.nsd_root
        self.repo = int(args.NSD.repo)
        self.stimuli_info = os.path.join(self.nsd_root, args.NSD.stimuli_info)
        self.general_mask_root = os.path.join(self.nsd_root, args.NSD.general_mask_root)
        self.roi_mask_root = os.path.join(self.nsd_root, args.NSD.roi_mask_root)
        self.session_num = int(args.NSD.session_num)
        self.beta_value_root = os.path.join(self.nsd_root, args.NSD.beta_value_root)
        self.from_coco_split = eval(args.NSD.from_coco_split)

        self.image_index_save_root = args.NSD.image_index_save_root
        self.image_root_save_root = args.NSD.image_root_save_root
        self.image_trail_save_path = args.NSD.image_trail_save_root

        self.individual_image_bool_save_root = args.NSD.individual_image_bool_save_root
        self.same_image_bool_save_root = args.NSD.same_image_bool_save_root

        self.roi_mask_save_root = args.NSD.roi_mask_save_root
        self.general_mask_save_root = args.NSD.general_mask_save_root
        self.voxal_zscore_response_save_root = args.NSD.voxal_zscore_response_save_root
        self.voxal_nonzscore_response_save_root = args.NSD.voxal_nonzscore_response_save_root
        self.nonzero_mask_save_root = args.NSD.nonzero_mask_save_root
        self.zscore_avg_activation_save_root = args.NSD.zscore_avg_activation_save_root
        self.zscore_activation_ev_save_root = args.NSD.zscore_activation_ev_save_root
        self.nonzscore_avg_activation_save_root = args.NSD.nonzscore_avg_activation_save_root
        self.nonzscore_activation_ev_save_root = args.NSD.nonzscore_activation_ev_save_root
        self.mask_coordinate_save_root = args.NSD.mask_coordinate_save_root
        self.unflattened_mask_save_root = args.NSD.unflattened_mask_save_root

        self.pure_response_save_root = args.NSD.pure_response_save_root
    
    # Get the NSD dataset stimulate infomation, the image index for each subject
    def extract_image_index(self, 
                            subj: int, 
                            save = False, 
                            ) -> List[int]:

        stim_info = pd.read_pickle(self.stimuli_info)
        key = "subject{}_rep0".format(subj)
        image_index_list = list(stim_info.cocoId[stim_info[key] != 0])

        if save:
            save_path = self.image_index_save_root.format(subj)
            check_path(save_path)
            torch.save(image_index_list, save_path)

        return image_index_list

    
    def load_image_index(self,
                         subj: int
                         ) -> List[int]:
        """_summary_

        Args:
            subj (int): which target subject, if haven't extracted image index, it will extract

        Returns:
            List[int]: target subj's image index list
        """
        save_path = self.image_index_save_root.format(subj)
        try:
            image_index_list = torch.load(save_path)
        except:
            print("extracting image index")
            image_index_list = self.extract_image_index(subj=subj, save=True)
        return image_index_list

    def extract_image_root(self,
                           subj: int,
                           save = False,
                           ) -> List[str]:
        
        from_coco_split = self.from_coco_split
        stim_info = pd.read_pickle(self.stimuli_info)
        key = "subject{}_rep0".format(subj)
        image_root_list = list(stim_info.cocoSplit[stim_info[key] != 0])
        image_index_list = list(stim_info.cocoId[stim_info[key] != 0])
        if from_coco_split:
            image_root_list = ["/".join([i, "{:012}.jpg".format(j)]) for i, j in zip(image_root_list, image_index_list)]
        else:
            image_root_list = ["{}.jpg".format(j) for j in image_index_list]

        if save:
            save_path = self.image_root_save_root.format(subj)
            check_path(save_path)
            torch.save(image_root_list, save_path)
        
        return image_root_list

    def load_image_root(self,
                        subj: int,
                        ) -> List[str]:
        save_path = self.image_root_save_root.format(subj)
        try:
            image_root_list = torch.load(save_path)
        except:
            print("extracting image root")
            image_root_list = self.extract_image_root(subj=subj, save=True)
        return image_root_list

    # Get the NSD dataset stimulate infomation, the trail index (10000 * 3) for each subject, each row is in commone with the image index list
    def extract_trail_index(self, 
                            subj: int, 
                            save = False, 
                            ) -> np.ndarray:
        # Get the fmri signal index for each image in test for a certain subj
        # each image has 3 trails, finally we will get 10000 * 3 trails
        # However, there exists a condition, that some subj didn't have 750 sessions
        # So we need to perform our code for special condition
        # In the code, we choose to sellect the first repo to select image index
        # then append the rest repo information based on the image index

        stim_info = pd.read_pickle(self.stimuli_info)
        key = "subject%d_rep%01d"
        trail_index = []
        for i in range(self.repo):
            select_info = key % (subj, i)
            trail_index.append(list(stim_info[select_info][stim_info[key % (subj, 0)] != 0]))

        # Turn to shape [10000, 3], and change from 1 based to 0 based
        # If subj didn't have 750 sessions, the trail_index would be -1
        trail_index = np.array(trail_index).T - 1  


        if save:
            save_path = self.image_trail_save_path.format(subj)
            check_path(save_path)
            torch.save(trail_index, save_path)

        return trail_index

    def load_trail_index(self,
                         subj: int
                         ) -> np.ndarray:
        save_path = self.image_trail_save_path.format(subj)
        try:
            trail_index = torch.load(save_path)
        except:
            print("extracting trail index")
            trail_index = self.extract_trail_index(subj=subj, save=True)
        return trail_index
    
    # Get the ROI mask for a certain subject, if roi is none, get the general mask for the subject
    def extract_roi_mask(self, 
                         subj: int, 
                         roi_name: str = "",
                         save=False
                         ) -> np.ndarray:
        # Get the ROI mask for a certain subject
        if roi_name == "": # cortical
            general_mask_root = self.general_mask_root
            general_mask_root = general_mask_root.format(subj)
            general_mask = nib.load(general_mask_root).get_fdata()
            new_general_mask = general_mask > -1

            if save:
                save_path = self.general_mask_save_root.format(subj, roi_name)
                unflattened_save_path = self.unflattened_mask_save_root.format(subj, roi_name)
                check_path(save_path)
                check_path(unflattened_save_path)
                torch.save(new_general_mask, save_path)
                torch.save(general_mask, unflattened_save_path)

            return new_general_mask, general_mask
        
        else:
            general_mask_root = self.general_mask_root.format(subj)
            general_mask = nib.load(general_mask_root).get_fdata()

            roi_mask_root = self.roi_mask_root.format(subj, roi_name)
            roi_mask = nib.load(roi_mask_root).get_fdata()

            new_roi_mask = roi_mask > 0
            if save:
                save_path = self.roi_mask_save_root.format(subj, roi_name)
                check_path(save_path)
                torch.save(new_roi_mask, save_path)

            return new_roi_mask, roi_mask

    def load_roi_mask(self,
                      subj: int,
                      roi_name: str = ""
                      ) -> np.ndarray:
        save_path = self.roi_mask_save_root.format(subj, roi_name)
        try:
            mask, _ = torch.load(save_path)
        except:
            print("extracting roi mask")
            mask, _ = self.extract_roi_mask(subj=subj, roi_name=roi_name, save=True)
        return mask

    def load_unflattened_mask(self,
                              subj: int,
                              roi_name: str = ""
                              ) -> np.ndarray:
        save_path = self.unflattened_mask_save_root.format(subj, roi_name)
        try:
            mask = torch.load(save_path)
        except:
            print("extracting roi mask")
            _, mask = self.extract_roi_mask(subj=subj, roi_name=roi_name, save=True)
        return mask

    # for each selected voxel, we need to extract it's actual coordinate
    def extract_mask_coordinate(self,
                                subj: int,
                                roi_name: str = "",
                                save=True
                                ) -> torch.Tensor:
        try:
            mask = torch.load(self.unflattened_mask_save_root.format(subj, roi_name))
        except:
            _, mask = self.extract_roi_mask(subj=subj, roi_name=roi_name, save=True)
        
        data = torch.ones_like(torch.from_numpy(mask))
        coodinate = torch.zeros(size=((mask > 0).sum(), 3))
        num = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    if mask[i, j, k] > 0:
                        coodinate[num, 0] = i
                        coodinate[num, 1] = j
                        coodinate[num, 2] = k
                        num += 1
        coodinate = coodinate.int().tolist()
        if save:
            save_path = self.mask_coordinate_save_root.format(subj, roi_name)
            check_path(save_path)
            torch.save(coodinate, save_path)

        return coodinate

    def load_mask_coordinate(self,
                             subj: int,
                             roi_name: str = ""
                             ) -> torch.Tensor:
        save_path = self.mask_coordinate_save_root.format(subj, roi_name)
        try:
            coodinate = torch.load(save_path)
        except:
            print("extracting mask coordinate")
            coodinate = self.extract_mask_coordinate(subj=subj, roi_name=roi_name, save=True)
        return coodinate

    # Get the voxal response for a certain subject, get voxal responses for the subj and the target roi, for all 40 sessions
    # meanwhile, choose to process zscore process
    def extract_voxal_activation(self, 
                                 subj: int, 
                                 roi_name: str = "",
                                 save=True,
                                 zscore=False, 
                                 ):
        try:
            mask = torch.load(self.roi_mask_save_root.format(subj, roi_name))
        except:
            mask, _ = self.extract_roi_mask(subj=subj, roi_name=roi_name, save=True)
        
        mask = check_tensor(mask)

        # for each person, there is 40 session, and each session has 750 figs
        # We will target subject's 30000 fMRI activation
        # The output will process zscore by default
        
        cortical_beta_mat = None

        for session in range(self.session_num):
            session_num = session + 1
            beta_value_root = self.beta_value_root.format(subj, session_num)
            # The default processing for several subjects whose session number is less than 40
            try:
                fmri_data = nib.load(beta_value_root)
                
                beta = fmri_data.get_fdata()
                cortical_beta = (beta[mask]).T  # verify the mask with array
                if cortical_beta_mat is None:
                    cortical_beta_mat = cortical_beta / 300
                else:
                    cortical_beta_mat = np.vstack((cortical_beta_mat, cortical_beta / 300))

            except:
                break

        if zscore:
            print("Zscoring...")
            cortical_beta_mat = zscore_by_run(cortical_beta_mat)
            finite_flag = np.all(np.isfinite(cortical_beta_mat))
            print("Is finite:" + str(finite_flag))

            if finite_flag == False:
                nonzero_mask = (
                    np.sum(np.isfinite(cortical_beta_mat), axis=0)
                    == cortical_beta_mat.shape[0]
                )
                nonzero_mask_save_root = self.nonzero_mask_save_root.format(subj, roi_name)
                torch.save(nonzero_mask, nonzero_mask_save_root)
            
        if save:
            if zscore:
                save_path = self.voxal_zscore_response_save_root.format(subj, roi_name)
            else:
                save_path = self.voxal_nonzscore_response_save_root.format(subj, roi_name)
            check_path(save_path)
            torch.save(cortical_beta_mat, save_path, pickle_protocol=4)

        return cortical_beta_mat

    def load_voxal_activation(self,
                              subj: int,
                              roi_name: str = "",
                              zscored: bool = True,
                              ) -> torch.Tensor:
        if zscored:
            response_root = self.voxal_zscore_response_save_root.format(subj, roi_name)
        else:
            response_root = self.voxal_nonzscore_response_save_root.format(subj, roi_name)
        try:
            response_data = torch.load(response_root)
        except:
            print("extracting voxal activation")
            response_data = self.extract_voxal_activation(subj=subj, roi_name=roi_name, save=True, zscore=zscored)
        return response_data

# compute an average voxal activation for three time repeate images
    def compute_ev(self,
                   subj: int,
                   roi_name: str = "",
                   zscored: bool = False, 
                   biascorr: bool = False,
                   save: bool = True,
                   ) -> torch.Tensor:
        trail_root = self.image_trail_save_path.format(subj)
        if zscored:
            response_root = self.voxal_zscore_response_save_root.format(subj, roi_name)
        else:
            response_root = self.voxal_nonzscore_response_save_root.format(subj, roi_name)

        try:
            response_data = torch.load(response_root)
        except:
            response_data = self.extract_voxal_activation(subj=subj, roi_name=roi_name, save=True, zscore=zscored)

        try:            
            trail_data = torch.load(trail_root)
        except:
            trail_data = self.extract_trail_index(subj=subj, save=True)
        
        repeat_num = trail_data.shape[0]

        ev_list = []
        average_voxal_activation = torch.zeros(size=(repeat_num, response_data.shape[1]))
        if response_data.shape[0] < 30000:
            new_response_data = np.zeros(shape=(30000, response_data.shape[1]))
            new_response_data[:] = np.nan
            new_response_data[:response_data.shape[0], :] = response_data.copy()
            response_data = new_response_data

        for v in tqdm(range(response_data.shape[1]), desc="computing everage voxal activations"):
            repo_data_list = []
            for r in range(self.repo):
                repo_data_list.append(response_data[trail_data[:, r], v])

            repo_data_list = torch.from_numpy(np.array(repo_data_list).T)
            mean_response = torch.nanmean(repo_data_list, dim=1)
            average_voxal_activation[:, v] = mean_response
            
            # TODO: edit the ev function to face the condition when some subjs have less than 40 sessions
            ev_list.append(ev(repo_data_list, biascorr=biascorr))

        ev_list = torch.Tensor(ev_list)

        if save:
            if zscored:
                avg_avtivation_save_path = self.zscore_avg_activation_save_root.format(subj, roi_name)
                ev_avtivation_save_path = self.zscore_activation_ev_save_root.format(subj, roi_name)
            else:
                avg_avtivation_save_path = self.nonzscore_avg_activation_save_root.format(subj, roi_name)
                ev_avtivation_save_path = self.nonzscore_activation_ev_save_root.format(subj, roi_name)
            check_path(avg_avtivation_save_path)
            check_path(ev_avtivation_save_path)
            torch.save(average_voxal_activation, avg_avtivation_save_path)
            torch.save(ev_list, ev_avtivation_save_path)
        
        return ev_list, average_voxal_activation
    
    def load_ev_value(self,
                      subj: int,
                      roi_name: str = "",
                      zscored: bool = True
                      ) -> torch.Tensor:
        if zscored:
            ev_root = self.zscore_activation_ev_save_root.format(subj, roi_name)
        else:
            ev_root = self.nonzscore_activation_ev_save_root.format(subj, roi_name)
        try:
            ev = torch.load(ev_root)
        except:
            print("extracting activation ev")
            ev, _ = self.compute_ev(subj=subj, roi_name=roi_name, zscored=zscored, save=True)
        return ev

    def load_avg_activation_value(self,
                      subj: int,
                      roi_name: str = "",
                      zscored: bool = True
                      ) -> torch.Tensor:
        if zscored:
            avg_activation_root = self.zscore_avg_activation_save_root.format(subj, roi_name)
        else:
            avg_activation_root = self.nonzscore_avg_activation_save_root.format(subj, roi_name)
        try:
            avg_activation = torch.load(avg_activation_root)
        except:
            print("extracting avg activation")
            _, avg_activation = self.compute_ev(subj=subj, roi_name=roi_name, zscored=zscored, save=True)
        return avg_activation


    def get_pure_activation(self,
                            subj: int,
                            roi_name: str = "",
                            zscored: bool = False,
                            save: bool = True
                            ) -> torch.Tensor:
        
        trail_root = self.image_trail_save_path.format(subj)

        if zscored:
            response_root = self.voxal_zscore_response_save_root.format(subj, roi_name)
        else:
            response_root = self.voxal_nonzscore_response_save_root.format(subj, roi_name)

        response_data = torch.load(response_root)
        trail_data = torch.load(trail_root)

        pure_list = []
        if response_data.shape[0] < 30000:
            new_response_data = np.zeros(size=(30000, response_data.shape[1]))
            new_response_data[:] = np.nan
            new_response_data[:response_data.shape[0], :] = response_data.copy()
            response_data = new_response_data

        for v in tqdm(range(response_data.shape[1]), desc="computing everage voxal activations"):
            repo_data_list = []
            for r in range(self.repo):
                repo_data_list.append(response_data[trail_data[:, r], v])
            
            repo_data_list = torch.from_numpy(repo_data_list).T.unsqueeze(1) # 10000 * 1 * 3
            
            pure_list.append(repo_data_list)
            
        pure_list = torch.cat(pure_list, dim=1) # 10000 * voxal_num * 3

        if save:
            pure_response_save_path = self.pure_response_save_root.format(subj, roi_name)
            check_path(pure_response_save_path)
            torch.save(pure_list, pure_response_save_path)

        return pure_list

# ==================================================================
# extract individual and same activation, image index and image root
# It's a list of Boolean
# ==================================================================


    def extract_individual_and_same_image_bool(self,
                                       subj: int,
                                       save = False,
                                       ) -> List[int]:
        
        individual_bool_list = torch.zeros(size=(10000,), dtype=torch.bool)
        same_bool_list = torch.zeros(size=(10000,), dtype=torch.bool)
        stim_info = pd.read_pickle(self.stimuli_info)
        key_0 = "subject{}_rep0".format(subj)
        key_1 = "subject{}_rep0".format(subj % 8 + 1)
        
        num = 0
        individual_list = list(map(lambda x, y: x & y, stim_info[key_0] != 0, stim_info[key_1] == 0))
        same_list = list(map(lambda x, y: x & y, stim_info[key_0] != 0, stim_info[key_1] != 0))
        num = 0
        for i in range(len(individual_list)):
            
            if individual_list[i]:
                individual_bool_list[num] = True
                num += 1
            elif same_list[i]:
                same_bool_list[num] = True
                num += 1

        if save:
            individual_save_root = self.individual_image_bool_save_root.format(subj)
            same_save_root = self.same_image_bool_save_root.format(subj)
            check_path(individual_save_root)
            check_path(same_save_root)
            torch.save(individual_bool_list, individual_save_root)
            torch.save(same_bool_list, same_save_root)

        return individual_bool_list, same_bool_list
    
    def load_individual_and_same_image_bool(self, 
                                             subj: int
                                             ) -> List[int]:
        try:
            individual_bool_list = torch.load(self.individual_image_bool_save_root.format(subj))
            same_bool_list = torch.load(self.same_image_bool_save_root.format(subj))
        except:
            print("extracting image bool list")
            individual_bool_list, same_bool_list = self.extract_individual_and_same_image_bool(subj, save=True)
        
        return individual_bool_list, same_bool_list
    