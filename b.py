import os
import pickle

layer_id = 5
save_dir = f"qwen3_scales/layer_{layer_id}"
with open(os.path.join(save_dir, "k_scale.pkl"), "rb") as f:
    print(pickle.load(f))
with open(os.path.join(save_dir, "v_scale.pkl"), "rb") as f:
    print(type(pickle.load(f)))
