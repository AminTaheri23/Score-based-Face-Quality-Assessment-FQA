# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from facenet import face
#from .config import verification_threshhold
from configuration import Config

config = Config()

class Verification:
    """
    تطابق یا عدم تطابق دو چهره
    """
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #self.session = tf.Session()
        self.images_placeholder = ''
        self.embeddings = ''
        self.phase_train_placeholder = ''
        self.embedding_size = ''
        self.session_closed = False

    def __del__(self):
        if not self.session_closed:
            self.session.close()

    def kill_session(self):
        self.session_closed = True
        self.session.close()

    def load_model(self, model):
        """
        بارگذاری مدل
        این تابع حتما باید قبل از توابع دیگر فراخوانی شود.
        ورودی این تابع مسیرمدل از قبل آموزش دیده برای استخراج ویژگی است.
        """
        print(str(id(self.session)) + '.....')
        face.load_model(model, self.session)

    def initial_input_output_tensors(self):
        """
        ایجاد تنسورهای ورودی و خروجی از روی مدل لود شده توسط تابع
        load_model
        """
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    def verify(self, img1, img2, image_size):
        """
        محاسبه فاصله ی دو تصویر چهره و تشخیص یکسانی یا تفاوت تصاویر.
        """
        images = face.make_images_tensor(img1, img2, False, False, image_size)
        
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder:False }
        emb_array = np.zeros((2, self.embedding_size))
        emb_array[0:2, :] = self.session.run(self.embeddings, feed_dict=feed_dict)

        diff = np.subtract(emb_array[0], emb_array[1])
        dist = np.sum(np.square(diff))
        if dist < config.verification_threshhold:
            return True, dist
        else:
            return False, dist

    def verify_with_embeddings(self, img, embeddings, label_list, image_size):
        """
        محاسبه فاصله ی تصویر تا embedding های دیتاست
        """
        emb = self.img_to_encoding(img, image_size)
        diff = np.subtract(embeddings, emb)
        dist = np.sum(np.square(diff), 1)
        indexes = dist < config.verification_threshhold
        return (np.array(embeddings)[indexes], np.array(label_list)[indexes], dist[indexes])

    def verify_with_embeddings_and_return_emb(self, emb, embeddings, label_list, image_size):
        """
        محاسبه فاصله ی تصویر تا embedding های دیتاست
        """
        # emb = self.img_to_encoding(img, image_size)
        diff = np.subtract(embeddings, emb)
        dist = np.sum(np.square(diff), 1)
        indexes = dist < 1000
        return dist[indexes]

    def img_to_encoding(self, img, image_size):
        """
        محاسبه embedding یک تصویر
        در اینجا یک وکتور 128 تایی برای هر تصویر
        """
        image = face.make_image_tensor(img, False, False, image_size)
        
        feed_dict = {self.images_placeholder: image, self.phase_train_placeholder:False }
        emb_array = np.zeros((1, self.embedding_size))
        emb_array[0, :] = self.session.run(self.embeddings, feed_dict=feed_dict)

        return np.squeeze(emb_array)


'''  
    def imgs_to_encodings(self, imgs, image_size):
        s = imgs.shape[0]
        images = face.make_batch_images_tensor(imgs, False, False, image_size)
        
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder:False }
        emb_array = np.zeros((s, self.embedding_size))
        emb_array[0:s, :] = self.session.run(self.embeddings, feed_dict=feed_dict)

        return emb_array
'''
