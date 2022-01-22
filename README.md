# Our method 
Demo on FashionVC dataset. 

## Description 
This is a simplified demo for the paper, including:  
load_data.py: function to load data.  
img_net.py: net structure for image modality.  
txt_net.py: net structure for text modality.   
demo_FSHDCH.py: a demo for our method on FashionVC dataset.  
other: tools to evaluate the preformance of the method.  

## Version 
* Python  3.7
* Tensorflow  1.15.0

## How-to run 
1. Download the [FashionVC](https://dl.acm.org/doi/abs/10.1145/3209978.3209996) dataset, and put it into ./DataSet  
    you can download dataset from pan.baidu.com  
    	link: [https://pan.baidu.com/s/1Wc32Upkd4YatSM0L0EgSoQ](https://pan.baidu.com/s/1VZwdU8MhWkvVmpMjrFJktw)  
    	password: mghx   
2. Download the [Ssense](https://dl.acm.org/doi/10.1145/3331184.3331229) dataset, and put it into ./DataSet  
    you can download dataset from pan.baidu.com  
    	link: [https://pan.baidu.com/s/1Jc5i2_dAB7DSLmq7Fcu7pw](https://pan.baidu.com/s/1RZsSZY5pY2GSAQEu5ciqAw)  
    	password: t76c
3. Download the [HIER Food-71] dataset constructed by us, and put it into ./DataSet
    you can download dataset from pan.baidu.com
        link:[]()
        password:
5. Download the imagenet-vgg-f.mat, and put it into ./DataSet  
    you can download it from pan.baidu.com  
    	link: [https://pan.baidu.com/s/1jM9ZPXGLIykw4SLzk3_3Wg](https://pan.baidu.com/s/15maHx23TDMSqOSr2G9kd6A）  
     	password: wmg4
4. run demo_FSHDCH.py
