# colmap-360-gaussian-splatting   
colmap 全景数据gaussian splatting  
# 实现方法    
曲线救国： panorama sfm --> postprocess sfm to cube sfm --> pinhole gaussian splatting   
当机立断： panorama sfm --> panorama gaussian splatting   
# 详细步骤  
（1） 当机立断法：   
i. 全景colmap sfm    
<img width="1243" alt="image" src="https://github.com/user-attachments/assets/51d62b7f-7cee-44db-9707-0f5386ab3fca">
ii. 将colmap结果转为opensfm 格式   
<img width="1231" alt="image" src="https://github.com/user-attachments/assets/fb270488-3cd1-43d2-bc6b-3ebc29990e6d">
iii. 直接利用360-panorama gs 去训练   


# 代码步骤   
(1) python3 transform_opensfm.py -path ***************    
 注意：path 路径下包括sfm的结果 0 和 全景图像images    
(2) python3 train -s /opensfm --panorama --output   
# 麦当劳支持    
如果你觉得这个仓库对你有用的话，可以打点麦当劳一份作为辛苦费，谢谢   
<img width="307" alt="image" src="https://github.com/user-attachments/assets/01ae0fbb-6f6b-48bc-a651-e2fc4ae6af93">


