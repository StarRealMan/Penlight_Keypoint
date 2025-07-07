from model import build_light_unet, focal_loss
from data import load_dataset
from prediction_callback import PredictionVisualizationCallback
import tensorflow as tf
import os

# 自定义回调：每3个epoch保存一次模型
class PeriodicModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, period=3, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.period = period
        self.verbose = verbose
    
    def on_epoch_end(self, epoch, logs=None):
        # epoch从0开始，所以epoch+1是实际的epoch数
        if (epoch + 1) % self.period == 0:
            filepath = self.filepath.format(epoch=epoch+1)
            if self.verbose > 0:
                print(f'\nEpoch {epoch+1:02d}: saving model to {filepath}')
            self.model.save(filepath)

EPOCHS = 30
SHUFFLE = 20000
BS = 32
RESO = 192

# 创建必要的目录
os.makedirs("model_checkpoints", exist_ok=True)
os.makedirs("training_predictions", exist_ok=True)

model = build_light_unet(input_shape=(RESO, RESO, 3))
# 使用更小的学习率避免数值不稳定
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # 降低学习率从默认的1e-3到1e-4
model.compile(optimizer=optimizer, loss=focal_loss(alpha=1.0, gamma=2.0))  # Focal loss for keypoint detection

# 加载训练数据集（启用数据增强）
train_ds = load_dataset(
    "data.json", 
    image_size=(RESO, RESO), shuffle=SHUFFLE, batch_size=BS,
    enable_augmentation=True  # 训练集启用数据增强
)

# 创建验证数据集（禁用数据增强，用于一致的可视化）
val_ds = load_dataset(
    "val.json", 
    image_size=(RESO, RESO), shuffle=0, batch_size=BS,  # shuffle=0表示不打乱顺序
    enable_augmentation=False  # 验证集禁用数据增强
)

# 创建预测可视化回调（节省存储空间）
prediction_callback = PredictionVisualizationCallback(
    validation_data=val_ds,
    save_dir="training_predictions",
    save_frequency=1,  # 每10个epoch保存一次（减少频率）
    num_samples=10       # 只保存3个样本的预测结果（减少数量）
)

# 添加其他有用的回调
callbacks = [
    prediction_callback,
    # 每3个epoch保存一次模型
    PeriodicModelCheckpoint(filepath='model_checkpoints/model_epoch_{epoch:02d}.h5', period=3, verbose=1),
    # 另外保存最佳模型
    tf.keras.callbacks.ModelCheckpoint(
        'model_checkpoints/best_model.h5',
        monitor='val_loss',
        save_best_only=True,  # 只保存验证损失最小的模型
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger('training_log.csv'),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    # 添加早停回调，防止损失变成NaN时继续训练
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

model.save("final_model.h5", save_format="h5")
