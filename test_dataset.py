import tensorflow as tf


def decode_masked_img(t):
    fws = tf.io.decode_proto(bytes=t,
                             message_type='MaskedImage',
                             field_names=['image_path', 'mask_path'],
                             output_types=[tf.string, tf.string],
                             descriptor_source="./src/dataset/proto/fss_dataset.desc")
    return fws[1]


def decode(t):
    fws = tf.io.decode_proto(bytes=t,
                             message_type='FewShotSample',
                             field_names=['query_image_path', 'support'],
                             output_types=[tf.string, tf.string],
                             descriptor_source="./src/dataset/proto/fss_dataset.desc")
    return fws[1][0], decode_masked_img(fws[1][1])

ds = tf.data.TFRecordDataset('./test.tfrecord')
ds = ds.map(decode)
elem = next(iter(ds))
# elem = decode(elem)
pass

