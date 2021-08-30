dataset_paths = {
	'celeba_train': '',
	'celeba_test': '',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '',
	'font_train_modern': '/content/drive/MyDrive/Japan/font_gen/paired_training_data/pr/rendered_chars_for_overfitting',
	'font_test_modern': '/content/drive/MyDrive/Japan/font_gen/paired_training_data/pr/rendered_chars_for_overfitting',
	'font_train_historical': '/content/drive/MyDrive/Japan/font_gen/paired_training_data/pr/char_crops_for_overfitting',
	'font_test_historical': '/content/drive/MyDrive/Japan/font_gen/paired_training_data/pr/char_crops_for_overfitting'
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': '/content/drive/MyDrive/pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': '/content/drive/MyDrive/pretrained_models/moco_v2_800ep_pretrain.pt',
	'stylegan_font': '/content/drive/MyDrive/stylegan2-ada-training-runs/00000-char_crops_for_stylegan2_ada-auto1/network-snapshot-002000.pt'
}
