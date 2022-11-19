from simple_parsing import ArgumentParser

from dataset_tools import config
from dataset_tools.record.multical.multical.app.calibrate import calibrate, Calibrate
from dataset_tools.record.multical.multical.config import run_with

if __name__ == '__main__':
    scene_name = 'scene_2211191054_ext'
    scene_path = f'{config.dataset_path}/{scene_name}'

    parser = ArgumentParser(prog='multical')
    parser.add_arguments(Calibrate, dest="app")
    program = parser.parse_args()

    program.app.paths.boards = './multical/example_boards/boards.yaml'
    program.app.paths.image_path = scene_path
    program.app.paths.limit_images = 2000
    program.app.vis = True

    program.app.optimizer.iter = 5
    program.app.optimizer.outlier_quantile = 0.5
    program.app.optimizer.outlier_threshold = 1.0
    program.app.optimizer.fix_intrinsic = True
    program.app.camera.calibration = f'{scene_path}/intrinsics.json'

    program.app.execute()