from mmdet.apis import init_detector, inference_detector, show_result_ins


# config_file = 'configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
# config_file = 'configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py'
config_file = 'configs/solov2/solov2_light_512_dcn_r50_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = 'checkpoints/DECOUPLED_SOLO_R50_3x.pth'
# checkpoint_file = 'checkpoints/SOLOv2_LIGHT_448_R18_3x.pth'
checkpoint_file = 'checkpoints/SOLOv2_LIGHT_512_DCN_R50_3x.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = 'demo/MTC ELITE session OVERHEAD VIEW-videoonly.mp4_20210117_152536276.jpg'
result = inference_detector(model, img)

show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="demo/test-out.jpg")
