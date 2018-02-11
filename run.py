import os
import numpy as np
import tensorflow as tf
import scipy.io
from matplotlib import pyplot as plt
import digitStruct
from meta import Meta


# this is going to be some CNN stuff
#svhn digit cinvenet
#reverse engineering depuis tensorpack

def create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file):
    print ('Saving meta file to %s...' % path_to_tfrecords_meta_file)
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.save(path_to_tfrecords_meta_file)


if __name__ == '__main__': # When we call the script directly ...

    # dataset should be in /data
    #SVHN_URL = "http://ufldl.stanford.edu/housenumbers/"

    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir,"data")
    img_dir = os.path.join(data_dir,'train')

    filename = tf.placeholder(tf.string)
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string)

    # WAIT mais en fait je veux pas faire ça si ?? sisi tkt
    offset_height = tf.placeholder(tf.int32)
    offset_width = tf.placeholder(tf.int32)
    target_height = tf.placeholder(tf.int32)
    target_width = tf.placeholder(tf.int32)

    image_crop = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width )

    resized_images = tf.cast(tf.image.resize_images(image_crop, [64, 64]),tf.uint8)



    path_to_image_files = tf.gfile.Glob(os.path.join(img_dir, '*.png'))
    total_files = len(path_to_image_files)
    print ( str(total_files) + " png files found in " + img_dir)


    num_examples = 0
    path_to_tfrecords_file = os.path.join(data_dir,'train.tfrecords')
    writer = tf.python_io.TFRecordWriter(path_to_tfrecords_file)


    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)


        img_num = 0
        dsFileName = "data\\train\\digitStruct.mat"

        for dsObj in digitStruct.yieldNextDigitStruct(dsFileName):
            img_num +=1
            if (img_num > 3) :
                break

            if (img_num < 65000):
                print("this is training")
            if (img_num >= 65000):
                print("this is validation")

            print("Processing image numba :")
            print(dsObj.name)
            print(img_num)

            image_label = [10, 10, 10, 10, 10]

            #very ugly sorry
            img_oh = 9999 # min
            img_ow = 9999 # min
            img_th = 0  # diff entre oh et
            img_tw = 0
            boxes = []

            attrs_width = []
            attrs_height =[]
            attrs_left = []
            attrs_top = []
            index_digit = 0
            for bbox in dsObj.bboxList:

                boxes.append(bbox)
                image_label[index_digit] = bbox.label

                attrs_width.append(bbox.width)
                attrs_height.append(bbox.height)
                attrs_top.append(bbox.top)
                attrs_left.append(bbox.left)

                index_digit += 1

            label_length = index_digit
            print(label_length)
            print(image_label)

            min_left, min_top = (min(attrs_left), min(attrs_top))

            max_right, max_bottom = (max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                    max(map(lambda x, y: x + y, attrs_top, attrs_height)))

            bbox_left, bbox_top = (min_left, min_top)
            bbox_width, bbox_height = (max_right - bbox_left, max_bottom - bbox_top)

            center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                 (min_top + max_bottom) / 2.0,
                                 max(max_right - min_left, max_bottom - min_top))

            bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                    center_y - max_side / 2.0,
                                                    max_side,
                                                    max_side)

            cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                            int(round(bbox_top - 0.15 * bbox_height)),
                                            int(round(bbox_width * 1.3)),
                                            int(round(bbox_height * 1.3)))

            bbox_left, bbox_top, bbox_width, bbox_height = (cropped_left, cropped_top, cropped_width, cropped_height)


            print('lol')
            print(bbox_left, bbox_top, bbox_width, bbox_height)

            image_path = os.path.join(img_dir,str(img_num)+'.png')



            png = sess.run(resized_images,feed_dict={filename: image_path,
                                                    offset_height : bbox_top,
                                                    offset_width: bbox_left,
                                                    target_height : bbox_height,
                                                    target_width : bbox_width })



            image = np.array(png).tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_length])),
                'digits': tf.train.Feature(int64_list=tf.train.Int64List(value=image_label))
            }))

            writer.write(example.SerializeToString())
            num_examples += 1
            # need a meta file


            print(png.shape)
            plt.imshow(png, interpolation='nearest')
            plt.show()
    #sess.run(iterator.initializer, feed_dict={filenames: training_filenames})


    path_to_tfrecords_meta_file = os.path.join(data_dir,'meta.json')
    create_tfrecords_meta_file(num_examples, 0, 0,
                                   path_to_tfrecords_meta_file)


    writer.close()
