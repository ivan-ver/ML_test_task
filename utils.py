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
    img = tf.image.resize(img, (256, 256))
    # После считывания изображения производится нормализация (0,1)
    img = tf.cast(img, tf.float32) / 255.0
    # Считывание маски
    mask_path = tf.strings.regex_replace(file_path,"valid", "valid_mask")
    mask_path = tf.strings.regex_replace(mask_path,"jpg", "png")
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_jpeg(mask, channels=3)
    mask = tf.image.resize(mask, (256, 256))
    # Нормализация
    mask = tf.cast(mask, tf.float32) 
    return img, mask / 255.0

def preprocess_train(file_path):
    """функция производит загрузку и предварительную обработку изображения
    Args:
        file_path (str): путь к исходному изображению

    Returns:
        tf.Tensor: два тензора - исходное изображение и его маска
    """
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    # После считывания изображения производится нормализация (0,1)
    img = tf.cast(img, tf.float32) / 255.0
    # Считывание маски
    mask_path = tf.strings.regex_replace(file_path,"train", "train_mask")
    mask_path = tf.strings.regex_replace(mask_path,"jpg", "png")
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_jpeg(mask, channels=3)
    mask = tf.image.resize(mask, (256, 256))
    # Нормализация
    mask = tf.cast(mask, tf.float32) 
    return img, mask / 255.0

