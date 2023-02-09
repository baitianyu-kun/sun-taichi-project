# Copyright (c) OpenMMLab. All rights reserved.
executor_cfg = dict(
    name='Test Webcam',
    camera_id=0,
    camera_max_fps=30,
    nodes=[
        dict(
            type='DetectorNode',
            name='detector',
            model_config='../demo/mmdetection_cfg/'
                         'ssdlite_mobilenetv2_scratch_600e_onehand.py',
            model_checkpoint='https://download.openmmlab.com/mmpose/'
                             'mmdet_pretrained/'
                             'ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth',
            input_buffer='_input_',
            output_buffer='det_result',
            multi_input=True),
        dict(
            type='MonitorNode',
            name='monitor',
            enable_key='m',
            enable=False,
            input_buffer='_frame_',
            output_buffer='display'),
        dict(
            type='RecorderNode',
            name='recorder',
            out_video_file='webcam_demo.mp4',
            input_buffer='display',
            output_buffer='_display_'
            # `_display_` is an executor-reserved buffer
        )
    ])
