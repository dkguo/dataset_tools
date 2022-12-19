import os

# make ~ work in python
dataset_path = os.path.expanduser('~/Data/kitchen_countertops')

resolution_width = 640  # pixels
resolution_height = 480  # pixels

models_path = f'{dataset_path}/models'
obj_model_paths = {1: f'{models_path}/002_master_chef_can/textured_simple.obj',
                   2: f'{models_path}/003_cracker_box/textured_simple.obj',
                   3: f'{models_path}/004_sugar_box/textured_simple.obj',
                   4: f'{models_path}/005_tomato_soup_can/textured_simple.obj',
                   5: f'{models_path}/006_mustard_bottle/textured_simple.obj',
                   6: f'{models_path}/007_tuna_fish_can/textured_simple.obj',
                   7: f'{models_path}/008_pudding_box/textured_simple.obj',
                   8: f'{models_path}/009_gelatin_box/textured_simple.obj',
                   9: f'{models_path}/010_potted_meat_can/textured_simple.obj',
                   10: f'{models_path}/011_banana/textured_simple.obj',
                   11: f'{models_path}/019_pitcher_base/textured_simple.obj',
                   12: f'{models_path}/021_bleach_cleanser/textured_simple.obj',
                   13: f'{models_path}/024_bowl/textured_simple.obj',
                   14: f'{models_path}/025_mug/textured_simple.obj',
                   15: f'{models_path}/035_power_drill/textured_simple.obj',
                   16: f'{models_path}/036_wood_block/textured_simple.obj',
                   17: f'{models_path}/037_scissors/textured_simple.obj',
                   18: f'{models_path}/040_large_marker/textured_simple.obj',
                   19: f'{models_path}/051_large_clamp/textured_simple.obj',
                   20: f'{models_path}/052_extra_large_clamp/textured_simple.obj',
                   21: f'{models_path}/061_foam_brick/textured_simple.obj'}

ply_models_path = f'{dataset_path}/ply_models'
ply_model_paths = {1: f'{ply_models_path}/obj_000001.ply',
                   2: f'{ply_models_path}/obj_000002.ply',
                   3: f'{ply_models_path}/obj_000003.ply',
                   4: f'{ply_models_path}/obj_000004.ply',
                   5: f'{ply_models_path}/obj_000005.ply',
                   6: f'{ply_models_path}/obj_000006.ply',
                   7: f'{ply_models_path}/obj_000007.ply',
                   8: f'{ply_models_path}/obj_000008.ply',
                   9: f'{ply_models_path}/obj_000009.ply',
                   10: f'{ply_models_path}/obj_000010.ply',
                   11: f'{ply_models_path}/obj_000011.ply',
                   12: f'{ply_models_path}/obj_000012.ply',
                   13: f'{ply_models_path}/obj_000013.ply',
                   14: f'{ply_models_path}/obj_000014.ply',
                   15: f'{ply_models_path}/obj_000015.ply',
                   16: f'{ply_models_path}/obj_000016.ply',
                   17: f'{ply_models_path}/obj_000017.ply',
                   18: f'{ply_models_path}/obj_000018.ply',
                   19: f'{ply_models_path}/obj_000019.ply',
                   20: f'{ply_models_path}/obj_000020.ply',
                   21: f'{ply_models_path}/obj_000021.ply'}

models_info_path = f'{ply_models_path}/models_info.json'



