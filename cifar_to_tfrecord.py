import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(img, lbl, x, y, z):
    return tf.train.Example(features=tf.train.Features(feature={
        'img': bytes_feature(img),
        'lbl': int64_feature(lbl),
        'x': int64_feature(x),
        'y': int64_feature(y),
        'z': int64_feature(z)}))


def _add_to_tfrecord(dataset_dir, data_files, tfrecord_writer):
    """ This file is used for writing the data to tfrecord files.

    From CIFAR10 website: Downloaded files 'data_batch_1', 'data_batch_2', 'test_batch' are dictionaries containing images and labels.
    img_lbl_dict[b'data'] is a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
    The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
    The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image """

    for i in range(len(data_files)):
        img_lbl_dict = unpickle(dataset_dir + data_files[i])
        img_list = img_lbl_dict[b'data']
        lbl_list = img_lbl_dict[b'labels']
        for j in range(len(img_list)):
            image = img_list[j]
            label = lbl_list[j]
            im_r = image[0:1024].reshape(_IMG_X, _IMG_Y)
            im_g = image[1024:2048].reshape(_IMG_X, _IMG_Y)
            im_b = image[2048:].reshape(_IMG_X, _IMG_Y)
            img = np.dstack((im_r, im_g, im_b))  # [32, 32, 3] image

            img = np.reshape(img, [_IMG_X * _IMG_Y * _IMG_Z])
            img = img.tobytes()
            example = image_to_tfexample(img, label, _IMG_X, _IMG_Y, _IMG_Z)

            tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.

    Args:
    dataset_dir: The directory where the temporary files are stored.
    split_name: The name of the train/test split.

    Returns:
    An absolute file path.
    """
    return '%s/cifar_%s.tfrecord' % (dataset_dir, split_name)


def unpickle(file):
    """ Provided in CIFAR10 website"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def display_image(image):
    """ Convert a [3072,] array to a [32,32,3] image and display it """
    im_r = image[0:1024].reshape(32, 32)
    im_g = image[1024:2048].reshape(32, 32)
    im_b = image[2048:].reshape(32, 32)
    img = np.dstack((im_r, im_g, im_b))
    plt.imshow(img)
    plt.show()
    return img


def read_tfrecord_file(tfrecord_filepath):
    # Read TfRecord
    tfrecord_file_queue = tf.train.string_input_producer([tfrecord_filepath], name='queue', num_epochs=None)

    # ------------------------------------ Solution ----------------------------
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecord_file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img': tf.io.FixedLenFeature([], tf.string),
            'lbl': tf.io.FixedLenFeature([], tf.int64),
            'x': tf.io.FixedLenFeature([], tf.int64),
            'y': tf.io.FixedLenFeature([], tf.int64),
            'z': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['img'], tf.uint8)
    height = tf.cast(features['x'], tf.int32)
    width = tf.cast(features['y'], tf.int32)
    depth = tf.cast(features['z'], tf.int32)
    image = tf.reshape(image, [_IMG_X, _IMG_Y, _IMG_Z])
    label = tf.cast(features['lbl'], tf.int32)

    #image.set_shape([303, 303, 3])
    # -----------------------------------------------------------------------
    return image, label


def main(dataset_dir):
    # names of the tfrecord files: eg: cifar_train.tfrecord
    train_filename = _get_output_filename(dataset_dir, 'train')
    valid_filename = _get_output_filename(dataset_dir, 'valid')
    test_filename = _get_output_filename(dataset_dir, 'test')

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # downloaded filenames for train and test datasets
    train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    valid_files = ['data_batch_5']
    test_files = ['test_batch']

    # Check the downloaded files: display any downloaded image
    batch_name = 'data_batch_2'
    image_no = 112
    img_lbl_dict = unpickle(dataset_dir + batch_name)
    display_image(img_lbl_dict[b'data'][image_no])
    print('Image class: ', classes[img_lbl_dict[b'labels'][image_no]])

    # Training Data:
    print('Converting Training-dataset:')
    with tf.python_io.TFRecordWriter(train_filename) as tfrecord_writer:
        # data_filename = os.path.join(dataset_dir, _TEST_DATA_FILENAME)
        # labels_filename = os.path.join(dataset_dir, _TEST_LABELS_FILENAME)
        _add_to_tfrecord(dataset_dir, train_files, tfrecord_writer)

    # Validation Data:
    print('Converting Validation-dataset:')
    with tf.python_io.TFRecordWriter(valid_filename) as tfrecord_writer:
        # data_filename = os.path.join(dataset_dir, _TEST_DATA_FILENAME)
        # labels_filename = os.path.join(dataset_dir, _TEST_LABELS_FILENAME)
        _add_to_tfrecord(dataset_dir, valid_files, tfrecord_writer)

    # Test Data:
    print('Converting Test-dataset:')
    with tf.python_io.TFRecordWriter(test_filename) as tfrecord_writer:
        # data_filename = os.path.join(dataset_dir, _TEST_DATA_FILENAME)
        # labels_filename = os.path.join(dataset_dir, _TEST_LABELS_FILENAME)
        _add_to_tfrecord(dataset_dir, test_files, tfrecord_writer)

    print('\nFinished converting the CIFAR-10 dataset!')

    # ------------------------------- Test: Read Tfrecord data file and display image ----------------------------------
    tfrecord_filepath = './cifar-10-batches-py/cifar_train.tfrecord'
    x, y = read_tfrecord_file(tfrecord_filepath)

    # Read an entire batch from a tfrecord-file; [x, y]: an image and a label as a single example
    output = tf.train.shuffle_batch([x, y], num_threads=4, batch_size=100, capacity=20000,
                                    min_after_dequeue=10000)

    # To evaluate x, need to launch x in a session
    global_init = tf.global_variables_initializer()  # Returns an Op that initializes global variables in graph
    local_init = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(global_init)
        sess.run(local_init)

        coord = tf.train.Coordinator()  # we need a controller to maintain the set of threads (know when main thread should terminate, request stopping of sub-threads, etc
        threads = tf.train.start_queue_runners(
            coord=coord)  # Starts all queue runners collected in the graph, Returns:A list of threads
        tf.train.start_queue_runners(sess=sess)

        for i in range(10,20):  #
            x_val, y_val = sess.run([x, y])  # y is hypothesis
            plt.imshow(x_val)
            plt.show()
            print('Image class: ', classes[y_val])

        out_im, out_lbl = sess.run(output)
        print(out_lbl)
        plt.imshow(out_im[90])
        plt.show()

        coord.request_stop()
        coord.join(threads)
        print('done')


_IMG_X = 32
_IMG_Y = 32
_IMG_Z = 3
dataset_directory = './cifar-10-batches-py/'
main(dataset_directory)