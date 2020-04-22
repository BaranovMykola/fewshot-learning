import tensorflow as tf


def show_sample(subplot, image, mask, classname, class_id):

    name = f'{classname.numpy().decode()} ({class_id})'
    assert image.shape == (224, 224, 3)
    assert mask.shape == (224, 224)

    # sample_masked = tf.concat([image[0, :, :, :3], tf.expand_dims(mask, axis=-1)], axis=-1)
    subplot.imshow(image.numpy())
    subplot.imshow(mask.numpy(), cmap='rainbow_alpha', alpha=0.4)
    subplot.title.set_text(name)
