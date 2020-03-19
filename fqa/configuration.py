import configparser

#cnf = configparser.ConfigParser()
#cnf.read("configuration.ini")

class Config:
    def __init__(self):
        
        self.__ini_params = ['test']

        # Recognition Params
        self.verification_threshhold = 1.15
        self.model = './model/models/20180402-114759/'
        self.image_tensor_size = 160
        self.notification_threshold = 5
        self.max_tracklet_images_check = 70
        self.KNN_k = 3

        # Quality
        self.faceQnet = './model/models/FaceQNet/FaceQnet.h5'

        # Directories
        self.dataset_base_dir = './model/dataset/'
        self.main_images_dir = './model/dataset'
        self.new_images_dir = './model/newPics'
        self.tracklet_save_dir = 'model/tracklet_images'
        self.detect_dir = 'model/images_to_detect'
 
        # Face Detection
        self.tracklet_save_padding = 30
        self.save_tracklet_skip_frames = 2
        self.image_width = 800
        self.detection_tracking_skip_frames = 10
        self.read_images_from = 'WebRTC'
        self.webRTC_file_name = 'camera.jpg'
        self.input_video_file_name = 'camera.avi'

        # Detecting Model
        self.detection = 'Retina' # Retina / MTCNN
        self.retina_model_prefix = './model/models/Retina/model-mnet/mnet.25'
        self.retina_epoch = 0
        self.retina_gpu_id = 0
        self.retina_network = 'net3' 

        #Tracking
        self.tracking = "centroid"
        self.centroid_max_disappeared = 40
        self.centroid_max_distance = 50

        # Align
        self.face_landmark = [
            [54.70657349, 73.85186005],
            [105.04542542, 73.57342529],
            [80.03600311, 102.48085785],
            [59.35614395, 131.95071411],
            [101.04272461, 131.72013855]
        ]

        # Log
        self.log_tracklet_check_time = True

        # Network
        self.hojre_server = '192.168.8.122'
        self.hojre_db_user = 'bahar'
        self.hojre_db_password = '123'
        self.hojre_db = 'shenasa2'

        #for k in self.__ini_params:
        #    setattr(self, k, self.get_ini(k))
        
    def get_ini(self, param):
        return cnf.get("setting", param)

    def set_ini(self, config_dict):
        for k in config_dict:
            if k in self.__ini_params:
                cnf["setting"][k] = config_dict[k]
                setattr(self, k, config_dict[k])

        with open("model/configuration.ini", "w") as config_file:
            cnf.write(config_file)

        return True
        
