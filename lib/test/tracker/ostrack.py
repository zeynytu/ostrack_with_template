import math

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        print("Yüklenen Model",self.params.checkpoint)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.similarity_average = pred_similarity_list()
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        self.non_tracking = False
        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        self.similarity_4_list = list()
        self.state_4_list = list()
        self.online_template = None
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.right, self.left, self.up, self.down = False, False, False, False
        self.imshow = False
    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        # cv2.imwrite(str(self.frame_id)+"_template.png", z_patch_arr )
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        
        
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                            output_sz=self.params.search_size)  # (x1, y1, w, h)
            
        self.frame_id += 1
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        if self.imshow: 
            self.imshow = False
            cv2.imshow("yeni template", x_patch_arr)
            cv2.waitKey(50)
            
        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            if self.online_template is not None :
                try:
                    self.online_template = self.online_template.tensors
                except:
                    pass
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ot=self.online_template, ce_template_mask=self.box_mask_z)
        

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        pred_similarity_score = out_dict['pred_similarity_score'].view(1).sigmoid().item()
        self.similarity_average.add_number(pred_similarity_score, 150)
        temp = self.state
        if self.frame_id % 150 == 0:
            self.online_template = None
            # if self.similarity_average.avg() <= 0.5:
            #     # print("düşük")
            #     self.similarity_average.similarity_score.clear()
            #     self.non_tracking = True
            #     for i in range(4):
            #         self.track2(image, i)
            #     the_highest_similarity = max(self.similarity_4_list)
            #     index_of_the_highest_similarity = self.similarity_4_list.index(max(self.similarity_4_list))
            #     # print(self.similarity_4_list)
            #     if the_highest_similarity > 0.7:
            #         # print("4_list ",self.state_4_list)
            #         self.state = self.state_4_list[index_of_the_highest_similarity]
            #     else:
            #         self.state = temp
            #     self.similarity_4_list.clear()
            #     self.state_4_list.clear()
            # elif self.similarity_average.avg() >0.9:
            if self.similarity_average.avg() >0.9:
                z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
                self.online_template = self.preprocessor.process(z_patch_arr, z_amask_arr)
               
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state,
                    "pred_similarity_score": pred_similarity_score
                    }
    def track2(self, image, counter):
        H, W, _ = image.shape 
        # if self.non_tracking:
            # x_center, y_center = self.calculate_center(self.state)
            # if x_center > round(W/2):
            #     self.right = True
            # else:
            #     self.left = True
            # if y_center > round(H/2):
            #     self.down = True
            # else:
            #     self.up = True
        if counter == 0:
            #print("1")
            self.state = []
            new_x = W/4 - W/16
            new_y = H/4 - H/16
            new_w = (W/16) * 2
            new_h = (H/16) * 2
        if counter ==  1:
            #print("2")
            new_x = W/4 - W/16 + W/2
            new_y = H/4 - H/16
            new_w = (W/16) * 2
            new_h = (H/16) * 2
        if counter == 2:
            #print("3")
            new_x = W/4 - W/16
            new_y = H/4 - H/16 + H/2
            new_w = (W/16) * 2
            new_h = (H/16) * 2
        if counter == 3:
            #print("4")
            new_x = W/4 - W/16 + W/2
            new_y = H/4 - H/16 + H/2
            new_w = (W/16) * 2
            new_h = (H/16) * 2
        #print("yenilendi")
        # self.non_tracking = False
        # self.imshow = True
        self.state = [new_x, new_y, new_w, new_h]
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                            output_sz=self.params.search_size)  # (x1, y1, w, h)
            
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        if self.imshow: 
            self.imshow = False
            cv2.imshow("yeni template", x_patch_arr)
            cv2.waitKey(50)
            
        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ot=None, ce_template_mask=self.box_mask_z)
        

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        pred_similarity_score = out_dict['pred_similarity_score'].view(1).sigmoid().item()
        self.state_4_list.append(self.state)
        self.similarity_4_list.append(pred_similarity_score)      
    def calculate_center(self, bbox):
        x_center = bbox[0] + round(bbox[2]/2)
        y_center = bbox[1] + round(bbox[3]/2)
        return x_center, y_center
    
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights
class pred_similarity_list():
    def __init__(self):
        self.similarity_score = list()
        self.average = 0
        self.sum = 0
        self.length = 0
    def add_number(self, pred_similarity, number):
        self.similarity_score.append(pred_similarity)
        if len(self.similarity_score) >= number:
            self.similarity_score.pop(0)
        
    def avg(self):
        self.sum = sum(self.similarity_score)
        self.average = self.sum / len(self.similarity_score)
        return self.average
def get_tracker_class():
    return OSTrack
