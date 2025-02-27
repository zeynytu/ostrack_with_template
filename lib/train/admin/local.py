class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/zeyn/tracking/OSTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/zeyn/tracking/OSTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/zeyn/tracking/OSTrack/pretrained_networks'
        self.lasot_dir = '/home/zeyn/Dataset/LaSOT'
        self.got10k_dir = '/home/zeyn/tracking/OSTrack/data/got10k/train'
        self.got10k_val_dir = '/home/zeyn/tracking/OSTrack/data/got10k/val'
        self.lasot_lmdb_dir = '/home/zeyn/tracking/OSTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/zeyn/tracking/OSTrack/data/got10k_lmdb'
        self.trackingnet_dir = '/home/zeyn/tracking/OSTrack/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/zeyn/tracking/OSTrack/data/trackingnet_lmdb'
        self.coco_dir = '/home/zeyn/tracking/OSTrack/data/coco'
        self.coco_lmdb_dir = '/home/zeyn/tracking/OSTrack/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/zeyn/tracking/OSTrack/data/vid'
        self.imagenet_lmdb_dir = '/home/zeyn/tracking/OSTrack/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
