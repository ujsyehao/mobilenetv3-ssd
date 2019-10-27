# mobilenetv3-ssd 
## backbone
Reference paper: https://arxiv.org/pdf/1905.02244.pdf

We mainly train mobilenetv3-ssd detection network rather than classification network, for convenience, we use trained mobiletnetv3-large network from https://github.com/xiaolai-sqlai/mobilenetv3

### open-source mobilenetv3-large classification network
| mobilenetv3-large      | top-1 accuracy    |  params(million)  | flops/Madds(million) | 
| --------   | :-----:   | :----: | :------: | 
|   https://github.com/xiaolai-sqlai/mobilenetv3  | 75.5             |       3.96            |       272               |   
|   https://github.com/d-li14/mobilenetv3.pytorch         |  73.2             |   5.15            |   246              |       
| https://github.com/Randl/MobileNetV3-pytorch      |73.5             |  5.48           |  220               |    
| https://github.com/rwightman/gen-efficientnet-pytorch | 75.6 | 5.5 | 219 | 
| official mobilenetv3 | 75.2 | 5.4 | 219 | 
| official mobilenetv2 | 72.0 | 3.4 | 300 |
| efficient B0 | 76.3 | 5.3 | 390 | 

