from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'ffhq_frontalize': {
		'transforms': transforms_config.FrontalizationTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'celebs_sketch_to_face': {
		'transforms': transforms_config.SketchToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_sketch'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test_sketch'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_segmentation'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test_segmentation'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'celebs_super_resolution': {
		'transforms': transforms_config.SuperResTransforms,
		'train_source_root': dataset_paths['celeba_train'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'font_style_transfer': {
		'transforms': transforms_config.FontTransforms,
		'train_source_root': dataset_paths['font_train_modern'],
		'train_target_root': dataset_paths['font_train_historical'],
		'test_source_root': dataset_paths['font_test_modern'],
		'test_target_root': dataset_paths['font_test_historical'],
	},
	'font_inversion': {
		'transforms': transforms_config.FontTransforms,
		'train_source_root': dataset_paths['font_train_inversion'],
		'train_target_root': dataset_paths['font_train_inversion'],
		'test_source_root': dataset_paths['font_test_inversion'],
		'test_target_root': dataset_paths['font_test_inversion'],
	},
	'font_colorize': {
		'transforms': transforms_config.FontTransforms,
		'train_source_root': dataset_paths['font_train_grayscale'],
		'train_target_root': dataset_paths['font_train_inversion'],
		'test_source_root': dataset_paths['font_test_grayscale'],
		'test_target_root': dataset_paths['font_test_inversion'],
	},
}
