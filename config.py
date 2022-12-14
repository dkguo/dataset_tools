import os


dataset_path = os.path.expanduser('~/Data/kitchen_countertops')

resolution_width = 640  # pixels
resolution_height = 480  # pixels

models_path = f'{dataset_path}/models'
models_info_path = f'{models_path}/models_info.json'

ycb_model_names = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
    "026_sponge"
]

obj_model_paths = {}
obj_texture_paths = {}
obj_ply_paths = {}
for model_name in ycb_model_names:
    obj_id = int(model_name[:3])
    obj_model_paths[obj_id] = f'{models_path}/{model_name}/textured_simple.obj'
    obj_texture_paths[obj_id] = f'{models_path}/{model_name}/texture_map.png'
    obj_ply_paths[obj_id] = f'{models_path}/{model_name}/object.ply'
