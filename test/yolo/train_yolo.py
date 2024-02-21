from ultralytics import YOLO
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Union, Literal
import os

# script_dir = os.path.dirname(os.path.abspath(__file__))  
# os.chdir(script_dir)  

####################
model_type = "yolov8s"
conf_title = "no_snow_rain"
desc = "Train the model on the dataset without considering the snow and rain weather"
start_point = ""
mode = "train"  # train / val

###################

with open("/home/zekun/drivable/test/yolo/cfg_map.json", "r") as f:
    configs = json.load(f)[conf_title]
conf_file = configs["cfg_file"]

def get_timestamp():
    now = datetime.now()  
    timestamp = now.strftime('%y%m%d%H%M%S')
    return timestamp

@dataclass
class MetaInfo:
    title: str
    desc: str
    cfg_title: str
    cfg_file: str
    start_point: Union[str, None]
    work_dir: str
    mode: Union[Literal["train"], Literal["val"]]

    def to_dict(self):
        return self.__dict__

title = f"{model_type}_{conf_title}_{mode}_{get_timestamp()}"
meta = MetaInfo(
    title=title,
    desc=desc,
    cfg_title=conf_title,
    cfg_file=conf_file,
    start_point=start_point,
    work_dir=f"/home/zekun/drivable/runs/detect/{title}",
    mode=mode,
)

with open("/home/zekun/drivable/test/yolo/info.json", "+r") as f:
    all_info = json.load(f)
    all_info[meta.title] = meta.to_dict()
    f.seek(0)
    json.dump(all_info, f, indent=4)
    f.truncate()
    f.close()

if meta.start_point!=None and meta.start_point!="":
    model = YOLO(f'{meta.start_point}/weights/best.pt')
else:
    model = YOLO('yolov8s.pt')
# model = YOLO('/home/zekun/drivable/runs/detect/yolov8s_snowrain2/weights/best.pt')
# model = YOLO('/home/zekun/drivable/runs/detect/yolov8m_custom/weights/best.pt')
# model = YOLO('yolov8s.pt')    

if mode == "train":
    results = model.train(
        data=meta.cfg_file,
        imgsz=640,  # 640
        epochs=50,
        batch=64,
        name=meta.title,
        optimizer="SGD",
        lr0=0.0005,
        lrf=0.1,
        close_mosaic=0, # 0:disable
        warmup_epochs=0.0,
        # resume=True,
    )
else:
    results = model.val(
        data=meta.cfg_file,
        name=meta.title,
    )

print(results)