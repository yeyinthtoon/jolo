import keras_cv
import tensorflow as tf

feature_description = {
    "image/height": tf.io.FixedLenFeature([], tf.int64),
    "image/width": tf.io.FixedLenFeature([], tf.int64),
    "image/filename": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image/encoded": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "image/format": tf.io.FixedLenFeature([], tf.string, default_value="jpeg"),
    "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    "image/object/class/text": tf.io.VarLenFeature(tf.string),
    "image/object/class/label": tf.io.VarLenFeature(tf.int64),
    "image/object/mask/binary": tf.io.VarLenFeature(tf.string),
    "image/object/mask/polygon": tf.io.VarLenFeature(tf.float32),
}


def parse_example(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def decode_features(example):
    image_raw = example["image/encoded"]
    image = tf.image.decode_image(image_raw)
    width = tf.cast(example["image/width"], "float32")
    height = tf.cast(example["image/height"], "float32")
    tf.ensure_shape(image, [None, None, 3])
    image.set_shape([None, None, 3])
    xmins = tf.sparse.to_dense(example["image/object/bbox/xmin"]) * width
    ymins = tf.sparse.to_dense(example["image/object/bbox/ymin"]) * height
    xmaxs = tf.sparse.to_dense(example["image/object/bbox/xmax"]) * width
    ymaxs = tf.sparse.to_dense(example["image/object/bbox/ymax"]) * height
    classes = tf.sparse.to_dense(example["image/object/class/label"])
    bboxes = tf.stack([xmins, ymins, xmaxs, ymaxs], axis=-1)
    bboxes = tf.cast(bboxes, "float32")

    labels = {
        "classes": classes,
        "boxes": bboxes,
    }

    return {"images": tf.cast(image, tf.float32), "bounding_boxes": labels}


def load_tfrecords(tfrec_files):
    dataset = tf.data.Dataset.from_tensor_slices(tfrec_files)

    dataset = dataset.shuffle(buffer_size=100, reshuffle_each_iteration=True)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=tf.data.AUTOTUNE).map(
            parse_example, num_parallel_calls=tf.data.AUTOTUNE
        ),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    dataset = dataset.map(decode_features, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def augmenter(inputs):
    x = keras_cv.layers.RandomFlip(
        mode="horizontal", bounding_box_format="xyxy", dtype="float32"
    )(inputs)
    x = keras_cv.layers.RandomShear(
        x_factor=0.2, y_factor=0.2, bounding_box_format="xyxy", dtype="float32"
    )(x)
    x = keras_cv.layers.RandomColorJitter(
        value_range=(0, 255),
        brightness_factor=(-0.2, 0.5),
        contrast_factor=(0.5, 0.9),
        saturation_factor=(0.5, 0.9),
        hue_factor=(0.5, 0.9),
        dtype="float32",
    )(x)
    x = keras_cv.layers.JitteredResize(
        target_size=(640, 640),
        scale_factor=(0.4, 2),
        bounding_box_format="xyxy",
        dtype="float32",
    )(x)
    return x


resizing = keras_cv.layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(1, 1),
    bounding_box_format="xyxy",
    dtype="float32",
)


def dict_to_tuple_kyolo(inputs):
    images = inputs["images"]
    no_rag_labels = {}
    no_rag_labels["classes"] = tf.cast(
        inputs["bounding_boxes"]["classes"].to_tensor(
            default_value=0, shape=[None, 100]
        )[..., None],
        "int32",
    )
    no_rag_labels["bboxes"] = inputs["bounding_boxes"]["boxes"].to_tensor(
        default_value=0, shape=[None, 100, 4]
    )
    return images / 255.0, no_rag_labels


def get_dataset(data_paths, batch_size, training=True):
    ds = load_tfrecords(data_paths)
    ds = ds.shuffle(batch_size * 16)
    ds = ds.ragged_batch(batch_size, drop_remainder=True)
    if training:
        ds = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(dict_to_tuple_kyolo, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )
    return ds
