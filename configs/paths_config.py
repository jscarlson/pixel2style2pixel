dataset_paths = {
	'celeba_train': '',
	'celeba_test': '',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '',
	'font_train_modern': '/mnt/data02/Japan/font_gen/paired_training_data/pr/labeled_validated_rendered_chars',
	'font_test_modern': '/mnt/data02/Japan/font_gen/paired_training_data/pr/rendered_chars_for_overfitting',
	'font_train_historical': '/mnt/data02/Japan/font_gen/paired_training_data/pr/labeled_validated_char_crops',
	'font_test_historical': '/mnt/data02/Japan/font_gen/paired_training_data/pr/char_crops_for_overfitting',
	'font_train_inversion': '/mnt/data01/AWS_S3_CONTAINER/personnel-records/1956/seg/firm/stylegan2_crops/pr',
	'font_test_inversion': '/mnt/data01/AWS_S3_CONTAINER/personnel-records/1956/seg/firm/stylegan2_crops_sample/pr/',
	'font_train_grayscale': '/mnt/data01/AWS_S3_CONTAINER/personnel-records/1956/seg/firm/stylegan2_crops_grayscale/pr',
	'font_test_grayscale': '/mnt/data01/AWS_S3_CONTAINER/personnel-records/1956/seg/firm/stylegan2_crops_grayscale_sample/pr',
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': '/mnt/workspace/pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': '/mnt/workspace/pretrained_models/moco_v2_800ep_pretrain.pt',
	'stylegan_font': '/mnt/workspace/github_repos_japan/stylegan2-pytorch/checkpoint/250000.pt'
}
