# mobilenetv3-ssd 
* train mobilenetv3-ssd use pytorch(provide **.pth* model)
* convert to ncnn model(provide **.param*, **.bin*)

## Backbone
Reference paper: MobileNetv3 https://arxiv.org/pdf/1905.02244.pdf

We mainly train mobilenetv3-ssd detection network rather than classification network, for convenience, we use trained mobiletnetv3-large network from https://github.com/xiaolai-sqlai/mobilenetv3 (**We are also trying to use** https://github.com/rwightman/gen-efficientnet-pytorch **provided mobilenetv3-large classification network**)

*open-source mobilenetv3-large classification network*

| mobilenetv3-large      | top-1 accuracy    |  params(million)  | flops/Madds(million) | 
| --------   | :-----:   | :----: | :------: | 
|   https://github.com/xiaolai-sqlai/mobilenetv3  | 75.5             |       3.96            |       272               |   
|   https://github.com/d-li14/mobilenetv3.pytorch         |  73.2             |   5.15            |   246              |       
| https://github.com/Randl/MobileNetV3-pytorch      |73.5             |  5.48           |  220               |    
| https://github.com/rwightman/gen-efficientnet-pytorch | 75.6 | 5.5 | 219 | 
| official mobilenetv3 | 75.2 | 5.4 | 219 | 
| official mobilenetv2 | 72.0 | 3.4 | 300 |
| official efficient B0 | 76.3 | 5.3 | 390 | 

For extra-body, we use **1x1 conv + 3x3 dw conv + 1x1 conv** block follow mobilenetv2-ssd setting(official tensorflow version), details below:
1x1 256 -> 3x3 256 s=2 -> 1x1 512
1x1 128 -> 3x3 128 s=2 -> 1x1 256
1x1 128 -> 3x3 128 s=2 -> 1x1 256
1x1 64  -> 3x3 64  s=2 -> 1x1 128

## Head
For head, we use **3x3 dw conv + 1x1 conv** block follow mobilenetv2-ssd-lite setting(official tensorflow version)

We train mobilenetv3-ssd use mmdetection framework(based on pytorch), it reaches 71.7mAP on VOC2007 test dataset.

## Convert mobilenetv3-ssd pytorch model to ncnn framework 
1. convert *.pth* model to onnx(not included priorbox layer, detection_output layer)
2. use onnx-simplifier to simplify onnx model
3. convert simplified *.onnx* model to ncnn
4. modify *.param* manually(add priorbox layer, detection_output layer, etc.)

## model link
mobilenetv3-ssd pytorch model 链接: https://pan.baidu.com/s/1N9HlCQjK2nsxf-AtOT-kKA 提取码: xnuw 
mobilenetv3-ssd ncnn model 链接: https://pan.baidu.com/s/1Zhp0_6asS5SRVyKJwJ7pkQ 提取码: xtcg 

