from model import build_pose_model
from data import load_dataset

EPOCHS = 30
SHUFFLE = 1000
BS = 32
RESO = 256
NUM_KEYPOINTS = 1

model = build_pose_model(input_shape=(RESO, RESO, 3), num_keypoints=NUM_KEYPOINTS)
model.compile(optimizer='adam', loss='mse')  # MSE on heatmap
train_ds = load_dataset(
    "default.json", "images/WIN_20250627_21_44_32_Pro", 
    image_size=(RESO, RESO), shuffle=SHUFFLE, batch_size=BS, num_keypoints=NUM_KEYPOINTS
)
model.fit(train_ds, epochs=EPOCHS)

model.save("penlight_ks.h5")