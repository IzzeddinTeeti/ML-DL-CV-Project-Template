"""
This is the dataloader class for ROAD dataset.
"""

from torch.utils.data import Dataset
import os
# import logging
from utils.logger import get_logger
import json
import numpy as np
# from PIL import Image
import torch

# from data.video_transforms import Resize_Clip, ToTensorStack
# from torchvision import transforms
# from torchvision.transforms import functional as F
import pickle
from feature_extraction.feat_extract import features_extract


logger = get_logger(__name__)


def filter_labels(ids: list, all_labels: list, used_labels: list) -> list:
    """
    Out of all class ids in the box, only return the used ones
    input:
        ids: list of integer ids like [0, 1]
        all_labels: list of all labels like ['car', 'truck'] for agent type
        used_labels: list of used labels like ['car'] for agent type
    output:
        used_ids: list of used integer ids like [0]
    """

    used_ids = [
        used_labels.index(all_labels[id]) for id in ids if all_labels[id] in used_labels
    ]
    return used_ids


def make_frames_ann(self, final_annots, num_label_type):
    """
    Get the labels for each frame in the video
    input:
        annots: dict of annotations for each frame in the video
        label_types: list of label types like ['agent', 'action', 'loc', 'duplex', 'triplet']
        used_labels: list of used labels like ['car'] for agent type
    output:
        labels: list of labels for each frame in the video
    """
    database = final_annots["db"]

    # array of (68x5) zeors to store classes counts in all boxes in all frames in all videos
    counts = np.zeros(
        (len(final_annots[self.label_types[-1] + "_labels"]), num_label_type),
        dtype=np.int32,
    )
    self.video_list = []  # list to store video names (1 x num of videos)
    self.numf_list = ([])  # list to store number of frames for each video (1 x num of videos)
    frame_level_list = ([])  # list to store frame level annotations for each video (num of videos x (num of frames))

    for videoname in sorted(database.keys()):

        # check if the video belongs to the currect subset train val test
        if not list(
            set(final_annots["db"][videoname]["split_ids"]) & set(self.subsets)
        ):
            continue

        numf = database[videoname]["numf"]
        self.numf_list.append(numf)
        self.video_list.append(videoname)

        frames = database[videoname]["frames"]  # list of frame names/ids
        # initiate a dictionary of 4 columns for frames in each video
        frame_level_annos = [
            {
                "labeled": False,
                "ego_label": -1,
                "boxes": np.asarray([]),
                "labels": np.asarray([]),
                "tube_ids": np.asarray([]),
            }
            for _ in range(numf)
        ]

        frame_nums = [int(f) for f in frames.keys()]  # convert frame names to numbers
        frames_with_boxes = 0
        
        # loop from start to last possible frame which can make a legit sequence
        for frame_num in sorted(frame_nums):  
            frame_id = str(frame_num)
            frame = frames[frame_id]
            if frame["annotated"] < 1:  # check if the frame is annotated
                continue
            
            # modify the dictionary of frames at the location of this frame
            frame_index = frame_num - 1
            frame_level_annos[frame_index]["labeled"] = True  
            frame_level_annos[frame_index]["ego_label"] = frames[frame_id]["av_action_ids"][0]

            # if it is not annotated, add an empty annotation
            if ("annos" not in frame.keys()):  
                frame = {"annos": {}}

            all_boxes = ([])  # initiate lists for all boxes (boxes num x 4) and all labels (boxes num x 149) in this frame
            all_labels = []
            all_tube_ids = []
            frame_annos = frame["annos"]

            # loop over all annotaions in this frame
            for (_, anno) in frame_annos.items():  
                width, height = frame["width"], frame["height"]
                box = anno["box"]  # access the box of the annotation

                # Check box and frame dimensions
                assert 0 <= box[0] < box[2] <= 1.01 and 0 <= box[1] < box[3] <= 1.01, (
                    "Invalid box dimensions",
                    box,
                )
                assert width == 1280 and height == 960, (
                    "Invalid frame dimensions",
                    width,
                    height,
                    box,
                )

                all_boxes.append(box)  # add the box to the list of boxes of the current frame
                box_labels = np.zeros(self.num_classes)  # initiate a list of 149 zeros for all classes for this box in the current frame
                list_box_labels = ([])  # (5 x (used ids)) since one box will have different types of labels (agent, action, etc)
                class_counter = 1

                for idx, name in enumerate(self.label_types):
                    filtered_ids = filter_labels(
                        anno[name + "_ids"],
                        final_annots["all_" + name + "_labels"],
                        final_annots[name + "_labels"],
                    )  # just to make sure the id is in the list of used labels
                    list_box_labels.append(filtered_ids)

                    for fid in filtered_ids:
                        box_labels[
                            fid + class_counter
                        ] = 1  # add 1 to the label of the box
                        box_labels[0] = 1
                    class_counter += self.num_classes_list[idx + 1]

                all_labels.append(box_labels)  # add the list of box labels to the list of current frame's labels
                all_tube_ids.append(anno["tube_uid"])

                # for index, box_labels in all_labels: increase the counts of the labels that contain 68 rows and 5 columns (one for each label type)
                for label_type, used_ids in enumerate(list_box_labels):
                    for id in used_ids:
                        counts[id, label_type] += 1

                # after finishing all the labels and boxes for this frame, convert the list to a numpy array
            all_labels = np.asarray(all_labels, dtype=np.float32)
            all_boxes = np.asarray(all_boxes, dtype=np.float32)


            frames_with_boxes += (
                1 if all_boxes.shape[0] > 0 else 0
            )  # increase the frames_with_boxes counter if there are boxes in this frame
            # if all_boxes.shape[0]>0: #if there are boxes in this frame increase the frames_with_boxes counter
            #     frames_with_boxes += 1
            frame_level_annos[frame_index]["boxes"] = all_boxes  # add the boxes and labels to the dictionary of frames for this video
            frame_level_annos[frame_index]["labels"] = all_labels
            frame_level_annos[frame_index]["tube_ids"] = all_tube_ids

        frame_level_list.append(frame_level_annos)  # Add the dictionary of annotations for this video to the list of all videos
        logger.info(
            "Frames with Boxes are {:d} out of {:d} in {:s}".format(
                frames_with_boxes, numf, videoname
            )
        )

        num_start_frames = 0
        # for frame_num in range(numf - 1 - self.pred_len, self.obs_len-1, -self.skip_step):
        for frame_num in range(self.obs_len-1, numf - 1 - self.pred_len, self.skip_step):
            video_id = self.video_list.index(videoname)
            self.ids.append([video_id, frame_num])
            num_start_frames += 1
        logger.info("number of start frames: {:d}".format(num_start_frames))

    return frame_level_list, counts



class ROAD_dataset(Dataset):
    """
    This is the dataloader class for ROAD dataset.
    """

    def __init__(self, cfg, mode="train", transform=None):
        self.data_root_path = cfg.DATA.DATA_PATH
        self.img_path = os.path.join(self.data_root_path, "rgb-images")
        self.mode = mode
        if self.mode == "train":
            self.subsets = cfg.DATA.TRAIN_SUBSETS
            anno_file = cfg.DATA.TRAIN_ANNO
        elif self.mode == "val":
            self.subsets = cfg.DATA.VAL_SUBSETS
            anno_file = cfg.DATA.TRAIN_ANNO
        elif self.mode == "test":
            self.subsets = cfg.DATA.TEST_SUBSETS
            anno_file = cfg.DATA.TEST_ANNO
        
        self.subsets = [
            val for val in self.subsets.split(",") if len(val) > 1
        ]  # if more than one subset is used
        self.anno_path = os.path.join(self.data_root_path, anno_file)
        
        logger.info('Start Loading {:s} Dataset'.format(self.mode))
        
        self.transform = transform
        self.FREQ = cfg.DATA.FREQUENCY  # Hz or frames per second
        self.mean = cfg.DATA.MEAN
        self.std = cfg.DATA.STD
        self.obs_time = cfg.PRIDECTION.OBS_TIME  # seconds
        self.obs_len = int(self.obs_time * self.FREQ)
        self.pred_time = cfg.PRIDECTION.PRED_TIME  # seconds
        self.pred_len = int(self.pred_time * self.FREQ)
        self.overlap = cfg.PRIDECTION.OVERLAP  # percentage
        self.skip_step = int(self.obs_len * (1 - self.overlap / 100))
        # logger.info(self.skip_step)
        self.ids = list()
        self._make_list()

        self.dataset_name = cfg.DATASET
        self.input_type = cfg.DATA.INPUT
        if self.input_type == "FEATURES":
            self.features = self._load_features(cfg)
            self.clip_feat_dim = 2048
        

    def _load_features(self, cfg):
        feat_name = self.mode + '_obs' + str(self.obs_len) + '_pred' + str(self.pred_len) + '_ol' + str(self.overlap)+ '_.pkl'
        feat_path = os.path.join('Datasets', self.dataset_name, feat_name)

        if not os.path.exists(feat_path):
            logger.info('Features for {:s} dataset ({:s}) not found, creating features'.format(self.dataset_name, self.mode))
            features_extract(cfg, self.mode, feat_path)
            
        
        logger.info('Loading features from %s', feat_path)
        features = pickle.load(open(feat_path, "rb"))
        return features
        
        # features = 

    def _make_list(self):
        """
        Load the annotations, loop over each annotation (label, class) in each frame in each
        video and make a list of all the frames in the dataset
        """
        with open(self.anno_path, "r") as fff:
            final_annots = json.load(fff)

        database = final_annots["db"]

        self.label_types = final_annots[
            "label_types"
        ]  # ['agent', 'action', 'loc', 'duplex', 'triplet']
        num_label_type = len(self.label_types)
        # logging.info('There are %d label types in total', num_label_type)

        self.num_classes = 1  # one for presence
        self.num_classes_list = [1]
        for name in self.label_types:
            logger.info(
                "Number of {:s}:: all : {:d} to use: {:d}".format(
                    name,
                    len(final_annots["all_" + name + "_labels"]),
                    len(final_annots[name + "_labels"]),
                )
            )
            numc = len(final_annots[name + "_labels"])
            self.num_classes_list.append(numc)
            self.num_classes += numc

        # show the number of all and used labels for each type, num classes=149
        logger.info("Total classes to use: {:d}".format(self.num_classes))

        self.ego_classes = final_annots["av_action_labels"]
        self.num_ego_classes = len(self.ego_classes)

        frame_level_list, counts = make_frames_ann(self, final_annots, num_label_type)

        # show the train/test distribution and save it as class's printed string
        ptrstr = ""
        self.frame_level_list = frame_level_list
        self.all_classes = [["agent_ness"]]
        for type_ind, name in enumerate(self.label_types):
            labels = final_annots[name + "_labels"]
            self.all_classes.append(labels)

            for label_ind, cls in enumerate(
                labels
            ):  # just to see the distribution of train and test sets
                ptrstr += "-".join(
                    self.subsets
                ) + " {:05d} label: ind={:02d} name:{:s}\n".format(
                    counts[label_ind, type_ind], label_ind, cls
                )

        ptrstr += "Number of ids are {:d}\n".format(len(self.ids))

        self.label_types = ["agent_ness"] + self.label_types
        self.childs = {
            "duplex_childs": final_annots["duplex_childs"],
            "triplet_childs": final_annots["triplet_childs"],
        }
        self.num_videos = len(self.video_list)
        self.print_str = ptrstr

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        video_id, frame_num = self.ids[idx]
        videoname = self.video_list[video_id]
        clip = torch.tensor(self.features[videoname][str(frame_num)]).to(torch.float32)
        
        unique_seq_ids = []
        max_tube_count = 50
        seq_boxes = torch.zeros(self.obs_len + self.pred_len, max_tube_count, 4) # len(unique_seq_ids)
        seq_labels = torch.zeros(self.obs_len + self.pred_len, max_tube_count, self.num_classes) # dtype=torch.float32
        seq_ego_labels = torch.full((self.obs_len + self.pred_len,), -1) # .to(torch.float32)

        seq_range = range(frame_num - self.obs_len + 1, frame_num + self.pred_len + 1)
        frame_ind = 0
        for cur_frame_num in seq_range:
            # cur_frame_num = frame_num - i
            # print('cur_frame_num', cur_frame_num)
            # if not self.frame_level_list[video_id][cur_frame_num]["labeled"]:
            #     frame_ind += 1
            #     continue
            # print('idx', idx, videoname, frame_num, cur_frame_num)
            cur_frame_tube_ids = self.frame_level_list[video_id][cur_frame_num]["tube_ids"]
            for tube_id in cur_frame_tube_ids:
                if tube_id not in unique_seq_ids:
                    unique_seq_ids.append(tube_id)
                # a = torch.from_numpy(self.frame_level_list[video_id][cur_frame_num]["boxes"][cur_frame_tube_ids.index(tube_id)])
                # print(a)
                # print(unique_seq_ids.index(tube_id), cur_frame_tube_ids.index(tube_id), unique_seq_ids)
                seq_boxes[frame_ind, unique_seq_ids.index(tube_id), :] = torch.from_numpy(self.frame_level_list[video_id][cur_frame_num]["boxes"][cur_frame_tube_ids.index(tube_id)].copy())
                seq_labels[frame_ind, unique_seq_ids.index(tube_id), :] = torch.from_numpy(self.frame_level_list[video_id][cur_frame_num]["labels"][cur_frame_tube_ids.index(tube_id)].copy())
            seq_ego_labels[frame_ind] = self.frame_level_list[video_id][cur_frame_num]["ego_label"]
            frame_ind += 1
        
        return (
            clip,
            seq_boxes,
            seq_labels,
            seq_ego_labels,
            unique_seq_ids,
            idx,
            videoname, 
            frame_num,
        )


    def custum_collate(batch):
        (
            clips,
            seq_boxes,
            seq_labels,
            seq_ego_labels,
            unique_seq_ids,
            idxs,
            videonames, 
            frame_nums,
        ) = zip(*batch)
        clips = torch.stack(clips, 0) # size of images is (#batches X 3 X #frames X H X W)
        # size of boxes is (batches, obs_len + self.pred_len, 50, 4)
        seq_boxes = torch.stack(seq_boxes, 0) 
        seq_labels = torch.stack(seq_labels, 0)
        seq_ego_labels = torch.stack(seq_ego_labels, 0)
        # unique_seq_ids = torch.stack(unique_seq_ids, 0)
        # idxs = torch.stack(idxs, 0)
        # videonames = torch.stack(videonames, 0)
        # frame_nums = torch.stack(frame_nums, 0)
      
        return clips, seq_boxes, seq_labels, seq_ego_labels, unique_seq_ids, idxs, videonames, frame_nums

    