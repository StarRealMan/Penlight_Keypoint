import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, Conv2D, BatchNormalization, Activation, Concatenate
from tensorflow.keras import mixed_precision

def focal_loss(alpha=1.0, gamma=2.0):
    """
    Focal Loss for keypoint heatmap regression.
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
    
    Returns:
        focal_loss_fixed: A loss function that can be used with Keras
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent numerical instability
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
        
        # Calculate focal loss components
        # For keypoint detection, we focus on hard examples
        pt = tf.where(y_true >= 0.5, y_pred, 1 - y_pred)
        
        # Calculate focal weight: (1 - pt)^gamma
        focal_weight = tf.pow(1 - pt, gamma)
        
        # Calculate cross entropy loss
        # For heatmap regression, we use binary cross entropy as base
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Apply focal weighting
        focal_loss_val = alpha * focal_weight * bce
        
        # Return mean loss
        return tf.reduce_mean(focal_loss_val)
    
    return focal_loss_fixed

# Set mixed precision policy for better performance
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def build_light_unet(input_shape=(192, 192, 3), num_output_channels=1):
    # MobileNetV2 基础网络
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    print(f"Base model input shape: {base_model.input_shape}")
    print(f"Base model output shape: {base_model.output_shape}")

    # 检查并获取中间层
    try:
        # 截断点：中间层作为 bottleneck
        bottleneck = base_model.get_layer('block_6_expand_relu').output  
        print(f"Bottleneck layer shape: {bottleneck.shape}")
        
        # 跳跃连接
        skip1 = base_model.get_layer('block_3_expand_relu').output   
        print(f"Skip1 layer shape: {skip1.shape}")
        
        skip2 = base_model.get_layer('block_1_expand_relu').output   
        print(f"Skip2 layer shape: {skip2.shape}")
        
    except ValueError as e:
        print(f"Layer not found error: {e}")
        # 如果找不到特定层名，使用基础网络的输出
        bottleneck = base_model.output
        # 创建简单的跳跃连接
        skip1 = base_model.layers[len(base_model.layers)//3].output
        skip2 = base_model.layers[len(base_model.layers)//6].output
        print(f"Using fallback layers:")
        print(f"Bottleneck (base output) shape: {bottleneck.shape}")
        print(f"Skip1 (layer {len(base_model.layers)//3}) shape: {skip1.shape}")
        print(f"Skip2 (layer {len(base_model.layers)//6}) shape: {skip2.shape}")

    # Decoder: 上采样 + Conv
    x = bottleneck
    print(f"Starting decoder with shape: {x.shape}")

    # 第1次上采样
    x = UpSampling2D()(x)
    print(f"After 1st upsampling: {x.shape}")
    x = Concatenate()([x, skip1])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print(f"After 1st conv block: {x.shape}")

    # 第2次上采样
    x = UpSampling2D()(x)
    print(f"After 2nd upsampling: {x.shape}")
    x = Concatenate()([x, skip2])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print(f"After 2nd conv block: {x.shape}")

    # 第3次上采样
    x = UpSampling2D()(x)
    print(f"After 3rd upsampling: {x.shape}")
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print(f"After 3rd conv block: {x.shape}")

    # 最终卷积层，直接输出而不进行第4次上采样
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print(f"After final conv block: {x.shape}")

    # 输出层（fp16下输出设为 float32 避免 tfjs 问题）
    output = Conv2D(num_output_channels, (1, 1), activation='sigmoid', dtype='float32')(x)
    print(f"Final output shape: {output.shape}")

    model = Model(inputs=base_model.input, outputs=output)
    
    # 验证最终模型的输入输出尺寸
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # 确保输出尺寸正确
    expected_output_shape = (None, input_shape[0], input_shape[1], num_output_channels)
    if model.output_shape != expected_output_shape:
        print(f"WARNING: Output shape mismatch!")
        print(f"Expected: {expected_output_shape}")
        print(f"Got: {model.output_shape}")
    
    return model