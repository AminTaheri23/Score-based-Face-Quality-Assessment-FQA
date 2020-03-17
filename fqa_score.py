import os
from random import shuffle

import cv2
import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage import data, exposure
#import gist
from retinaface import RetinaFace
from skimage import transform as trans
from keras import Input, Model
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from FaceToolKit import Verification


# # TODO Change this
#embedding_image
verification = Verification()
verification.load_model('20180204-160909/')  # this is squeeze net ( a light feature extractor)
verification.initial_input_output_tensors()


# def get_pixel(img, center, x, y):
#     new_value = 0
#     try:
#         if img[x][y] >= center:
#             new_value = 1
#     except IndexError:
#         pass
#     return new_value


# def lbp_calculated_pixel(img, x, y):
#     """
#      64 | 128 |   1
#     ----------------
#      32 |   0 |   2
#     ----------------
#      16 |   8 |   4
#     """
#     center = img[x][y]
#     val_ar = list()
#     val_ar.append(get_pixel(img, center, x - 1, y + 1))  # top_right
#     val_ar.append(get_pixel(img, center, x, y + 1))  # right
#     val_ar.append(get_pixel(img, center, x + 1, y + 1))  # bottom_right
#     val_ar.append(get_pixel(img, center, x + 1, y))  # bottom
#     val_ar.append(get_pixel(img, center, x + 1, y - 1))  # bottom_left
#     val_ar.append(get_pixel(img, center, x, y - 1))  # left
#     val_ar.append(get_pixel(img, center, x - 1, y - 1))  # top_left
#     val_ar.append(get_pixel(img, center, x - 1, y))  # top

#     power_val = [1, 2, 4, 8, 16, 32, 64, 128]
#     val = 0
#     for i in range(len(val_ar)):
#         val += val_ar[i] * power_val[i]
#     return val


# def lbp(image):
#     height, width = image.shape
#     img_lbp = np.zeros((height, width, 1), np.uint8)
#     for i in range(0, height):
#         for j in range(0, width):
#             img_lbp[i, j] = lbp_calculated_pixel(image, i, j)
#     img_lbp_flatten = img_lbp.flatten()
#     return img_lbp_flatten, img_lbp


def get_transform_matrix(landmark):
    """
    get landmarks from retina face and returns 
    SimilarityTransform for a 100 * 100 face
    this number are hard coded and gather from a sample 
    and can be unacurate, 
    we do this for face alignemnt
    """
    src = np.array([
        [25, 23],
        [73, 23],
        [43, 55],
        [24, 71],
        [74, 71]], dtype=np.float32)

    t_form = trans.SimilarityTransform()
    t_form.estimate(landmark, src)
    m = t_form.params[0:2, :]
    return m


def check_size(img):
    """
    check for size of pics that are 100*100 or 160*160
    """
    if (img.shape[0] != 100 or img.shape[1] != 100) and (img.shape[0] != 160 or img.shape[1] != 160):
        raise AttributeError
    else:
        return img


def align_image(image, _, points, size):
    try:
        m = get_transform_matrix(points[0])
        img = cv2.warpAffine(image, m, (size, size), borderValue=0.0)
        img = img[:size, :size]
    except:
        return image[:size, :size]
    return img


def get_embeddings(images):
    image_embeddings = list()
    for img in images:
        b, g, r = cv2.split(img)
        merged = cv2.merge([r, g, b])
        emb = verification.img_to_encoding(merged, 160)
        image_embeddings.append(emb)
    return np.array(image_embeddings)


def auto_encoder(encoding_dim, input_image_shape, x_train, x_test):
    # this is the size of our encoded representations
    # this is our input placeholder
    input_img = Input(shape=(input_image_shape,))

    encoded = Dense(encoding_dim * 150, activation='relu')(input_img)
    encoded = Dense(encoding_dim * 50, activation='relu')(encoded)
    encoded = Dense(encoding_dim * 5, activation='relu')(encoded)
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(encoding_dim * 5, activation='relu')(encoded)
    decoded = Dense(encoding_dim * 50, activation='relu')(decoded)
    decoded = Dense(encoding_dim * 150, activation='relu')(decoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_image_shape, activation='relu')(decoded)
    # this model maps an input to its reconstruction
    my_auto_encoder = Model(input_img, decoded)
    # intermediate result
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer_0 = my_auto_encoder.layers[-4](encoded_input)
    decoder_layer_1 = my_auto_encoder.layers[-3](decoder_layer_0)
    decoder_layer_2 = my_auto_encoder.layers[-2](decoder_layer_1)
    decoder_layer_3 = my_auto_encoder.layers[-1](decoder_layer_2)
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer_3)

    opt = Adam(lr=0.00001)
    my_auto_encoder.compile(optimizer=opt, loss='binary_crossentropy')
    my_auto_encoder.summary()
    # my_auto_encoder.fit(
    #     x_train,
    #     x_train,
    #     epochs=50,
    #     batch_size=100,
    #     shuffle=True,
    #     validation_data=(x_test, x_test)
    # )
    return my_auto_encoder, encoder, decoder


def read_data(img_pth, batch_num):
    list_image = list()
    list_label = list()
    for j in range(3):   # for 3 classes of 0_good, 1_average, 2_bad
        print('img path folders', img_pth + '/' + str(j) + '/')
        for _, _, file_names in os.walk(img_pth+'/'+str(j)+'/'):
            for i in range(0, len(file_names)):
                try:
                    read_img = cv2.imread(img_pth+'/'+str(j)+'/'+file_names[i])
                    if len(read_img.shape) == 3:
                        list_image.append(read_img)
                    list_label.append(img_pth+'/'+str(j)+'/'+file_names[i].split()[0])
                    if i % 100 == 0:
                        print('how many image have beed read? ', i)
                except:
                    print('we had an exception in reading, len of loaded image is', len(list_image))
    return list_image, list_label


def cropped_and_aligned(img_list):
    error_num = 0
    # retina face initialization
    model = RetinaFace('model-mnet/mnet.25', 0, 0, 'net3')
    img_list_cropped_gray = list()
    img_list_cropped_color = list()
    img_list_160 = list()
    for i in range(len(img_list)):
        # for Gray images
        try:
            if i % 1000 == 0:
                print('Crop and aligned', i)
            detections, point = model.detect(img_list[i])
            img_gray = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            img_aligned_g = align_image(img_gray, detections, point, size=100)
            img_cropped_g = check_size(img_aligned_g)
            img_list_cropped_gray.append(img_cropped_g)

            # for BGR images
            img_aligned_color = align_image(img_list[i], detections, point, size=100)
            img_cropped_color = check_size(img_aligned_color)
            img_list_cropped_color.append(img_cropped_color)

            # for cnn
            img_aligned_160 = align_image(img_list[i], detections, point, size=160)
            img_cropped_160 = check_size(img_aligned_160)
            img_list_160.append(img_cropped_160)
        except AttributeError:
            error_num += 1
            print('number of error ', error_num)
            pass
    return img_list_cropped_gray, img_list_cropped_color, img_list_160


def lbp_cal_batch(img_list):
    # settings for LBP
    radius = 5
    n_points = 5 * radius
    method = 'uniform'
    img_list_feature_lbp_flatten = np.zeros((len(img_list), 10000))
    for i, img in enumerate(img_list):
        # local binary pattern
        img_lbp_flatten = local_binary_pattern(img, n_points, radius, method)
        img_lbp_flatten = img_lbp_flatten.flatten()
        # TODO change this
        # cv2.imwrite('lbp_pics/'+str(i)+'.jpg', img_lbp)
        img_list_feature_lbp_flatten[i] = img_lbp_flatten
    return img_list_feature_lbp_flatten


def hog_cal_batch(img_list):
    img_list_hog_flatten = np.ones((len(img_list), 8100))
    for i, img in enumerate(img_list):
        # HOG is here
        img_hog_flatten_vec, img_hog = hog(
            img,
            orientations=9,  # default for hog
            # pixels_per_cell=(32, 32),  # based on paper
            feature_vector=True,
            visualize=True,
            multichannel=True,
            block_norm='L2-Hys'
        )
        # hog_image_rescaled = exposure.rescale_intensity(img_hog, in_range=(0, 10))
        # hog_feature_rescaled = exposure.rescale_intensity(img_hog_flatten_vec, in_range=(0, 10))
        #cv2.imwrite('hog_pics/'+str(i)+'.jpg', img_hog)
        if len(img_hog_flatten_vec) == 8100:
            img_list_hog_flatten[i] = img_hog_flatten_vec
        else:
            print(img_hog_flatten_vec.shape)
            print(i)
    return img_list_hog_flatten


def gist_cal_batch(img_list):
    img_list_gist_flatten = np.zeros((len(img_list), 33750))
    for i, img in enumerate(img_list):
        # GIST
        img_gist = gist.extract(
            img,
            nblocks=25,
            orientations_per_scale=(8, 5, 5)
        )
        # img_gist = np.pad(img_gist, pad_width=50, mode='mean')  # 50 * 2 is added to img_gist
        # cv2.imwrite('hog_pics/'+str(i)+'.jpg', img_gist)
        # print(img_gist.shape)
        # if len(img_gist) == 1000:
        img_list_gist_flatten[i] = img_gist
    return img_list_gist_flatten


def main():
    print('hi, this is main!')
    image_path = 'pics' # this is iamge path. in this folder we must have 3 folders that are named 0 to 2
                        # 0 for good, 1 for average, 2 for bad pics
                        
    for i in range(1, 2):  # TODO edit this, i try to make batch of read imgaes that we can utilize ram usage 
        (read_image, read_labels) = read_data(img_pth=image_path, batch_num=i)
        print("len of read_image ", len(read_image), len(read_image[0]))
        (img_list_cropped_gray, img_list_cropped_color, img_cnn) = cropped_and_aligned(read_image)
        print(
                "img_list_cropped_gray: ",
                len(img_list_cropped_gray),
                len(img_list_cropped_gray[0]),
                len(img_list_cropped_gray[0][0])
                )

        # What features u want? select one and comment others 
        # img_list_hog_flatten = hog_cal_batch(img_list_cropped_color)
        # img_list_feature_lbp_flatten = lbp_cal_batch(img_list_cropped_gray)
        #img_list_feature_gist_flatten = gist_cal_batch(img_list_cropped_color)
        img_list_feature_cnn = get_embeddings(img_cnn)

        # make sure that u change this vairable (the_array) 
        the_array = img_list_feature_cnn
        print(the_array.shape)

        auto_encoder1, my_encoder, my_decoder = auto_encoder(
            encoding_dim=50,
            input_image_shape=10000,  # for hog, assign this to 8100
            x_train=the_array[:2000*i],  # 90/10 is train test split of every batch, you cant change this maunally 
            # TODO Change this and the line above
            x_test=the_array[2000*i:]
            )
        print('########thearray shape of elemnt 1: ', the_array[0].shape)
        print(the_array[0])
        auto_encoder1.fit(
            the_array[:2000], 
            the_array[:2000],
            epochs=2,
            batch_size=100,
            shuffle=True,
            validation_data=(the_array[2000:], the_array[2000:])
        )
        
        # testing an image for manual visualization :)
        main_predict = auto_encoder1.predict(the_array[0:10])
        encoded_img = my_encoder.predict(the_array[0:10])
        decoded_img = my_decoder.predict(encoded_img)
        # i'm trying to write oroginal image and reconstrated for validating auto encoder 
        main_img = np.reshape(main_predict[1], (100,100))
        cv2.imwrite('main.jpg', main_img)
        img_enc = np.reshape(decoded_img[1], (100,100))
        img_org = np.reshape(the_array[1], (100,100))
        cv2.imwrite('mohammad(orginal).jpg', img_org)
        cv2.imwrite('mmd.jpg(compressed).jpg', img_enc)
        print('bye')


if __name__ == '__main__':
    main()
