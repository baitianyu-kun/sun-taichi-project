executor_cfg = dict(
    name='Test Webcam',
    camera_id=0,
    camera_max_fps=30,
    nodes=[
        dict(
            type='MonitorNode',
            name='monitor',
            enable_key='m',
            enable=False,
            input_buffer='_frame_',
            output_buffer='display'),
        dict(
            type='TopDownPoseEstimatorNode',
            name='human pose estimator',
            model_config='../model/associative_embedding_hrnet_w32_coco_512x512.py',
            model_checkpoint='../model/hrnet_w32_coco_512x512-bcb8c247_20200816.pth',
            smooth=True,
            input_buffer='det_result',
            output_buffer='human_pose')
    ])