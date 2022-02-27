import tensorflow as tf

def preprocess_valid(file_path):
    """функция производит загрузку и предварительную обработку изображения
    Args:
        file_path (str): путь к исходному изображению

    Returns:
        tf.Tensor: два тензора - исходное изображение и его маска
    """
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    # После считывания изображения производится нормализация (0,1)
    img = tf.cast(img, tf.float32) / 255.0
    # Считывание маски
    mask_path = tf.strings.regex_replace(file_path,"valid", "valid_mask")
    mask_path = tf.strings.regex_replace(mask_path,"jpg", "png")
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_jpeg(mask, channels=3)
    # Нормализация
    mask = tf.cast(mask, tf.float32) / 255.0
    #mask = tf.reshape(mask, [-1], name=None)
    return img, mask

def preprocess_train(file_path):
    """функция производит загрузку и предварительную обработку изображения
    Args:
        file_path (str): путь к исходному изображению

    Returns:
        tf.Tensor: два тензора - исходное изображение и его маска
    """
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    # После считывания изображения производится нормализация (0,1)
    img = tf.cast(img, tf.float32) / 255.0
    # Считывание маски
    mask_path = tf.strings.regex_replace(file_path,"train", "train_mask")
    mask_path = tf.strings.regex_replace(mask_path,"jpg", "png")
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_jpeg(mask, channels=3)
    # Нормализация
    mask = tf.cast(mask, tf.float32) / 255.0
    #mask = tf.reshape(mask, [-1], name=None)
    return img, mask

