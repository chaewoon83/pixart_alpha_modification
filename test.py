import numpy as np
x = np.load("data/coco_train/caption_feature_wmask/000000138713.npz")
print(x["caption_feature"].shape)
print(x["attention_mask"].shape)