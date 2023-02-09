import json

f=open('./data/h36m/h36m_coco.json')
data=json.load(f)
print(data['annotations'][0]['keypoints'])