import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import UpSampling2D, Conv2D, Input, BatchNormalization, Activation

def build_pose_model(input_shape=(256, 256, 3), num_keypoints=1):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    x = base.output  # 输出大概是 (8, 8, 1280)
    
    # 逐步上采样回到原图大小 (256, 256)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 16x16
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 32x32
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 64x64
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 128x128
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 256x256
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 输出每个关键点的 heatmap
    output = Conv2D(num_keypoints, (1, 1), activation='sigmoid', padding='same')(x)

    model = tf.keras.Model(inputs=base.input, outputs=output)
    return model