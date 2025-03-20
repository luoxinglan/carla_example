# **基于Carla的直播回放虚实结合系统**  

**个人实践项目，存在瑕疵，持续改进中……** （工作量集中在mycarla，对示例脚本的魔改）

---

## 目录

- [项目概述](#项目概述)  
- [功能特性](#功能特性)  
- [快速开始](#快速开始)  
  - [安装要求](#安装要求)  
  - [安装步骤](#安装步骤)  
- [使用示例](#使用示例)  
- [项目结构](#项目结构) 

---

## 项目概述  

- **一句话描述**：能够在指定场景下执行自动驾驶脚本，与仿真小车联动，并且进行记录与回放，可在pygame直接使用，也可以结合前后端进行网页访问。  
- **背景与意义**：基于Carla的虚实结合自动驾驶脚本模拟系统。  
- **目标用户**：用来学习Carla仿真的研究者。  

---

## 功能特性  

- 列出核心功能：  
  ✅ 功能1：支持复杂场景自动驾驶仿真，自动驾驶算法效果演示  
  ✅ 功能2：提供RESTful API接口，可供上传数据、实时视频流，进行直播。    
  ✅ 功能3：支持Pygame或者网页端录制与回放。
- 本人的工作量：集中体现在mycarla文件夹下
  - replay 回放记录读取、处理、加载：将内置方法记录下来的二进制文件处理为可读取的文件。
  - routes 路径记录与读取类。
  - sensors 多个传感器类：
    - rgb、lidar、segmentation等的sensormanager。
    - 追踪加速度、速度、TTC、油门开度、方向、刹车等的car_tracker。

  - world
    - generate_traffic_my.py 加载指定地图、生成交通流。
    - manual_control_my.py 以pygame方式打开client，进行手动控制以及记录的回放。


---

## 快速开始  

### 安装要求    

  - Python 3.8+  
  - Carla 0.9.13   

### 安装步骤  

分步说明如何部署项目，示例：  

```bash  
# 克隆仓库  
git clone git@github.com:luoxinglan/carla_example.git

# 安装依赖  
pip install -r requirements.txt  

# 启动服务  
./CarlaUE4.sh
python ./mycarla/world/manual_control_my.py  #pygame方式启动
python ./mycarla/sensors/cams_and_lidar.py #需要配合服务器启动
```

---

## 使用示例

- 提供代码片段或命令行示例：  

  ```python  
  from projectname import scheduler  
  scheduler.run(task="daily_report")  
  ```

- 效果展示
-   ![92984a2b9c11384c0041e90c1dc723e](https://github.com/user-attachments/assets/e281edb4-bd1d-4d12-8428-6c6e314df9c7)
-   ![image](https://github.com/user-attachments/assets/2ef4d70d-4d29-4583-b4c9-97f72e505743)




---

## 项目结构

```  
carla_example/  
├── carla/            # carla源代码目录  
├── examples/          # carla自带测试用例  
├── mycarla/           # 本人的工作量 
	├── logs/		# 回放记录文件
	├── replay/		# 回放记录读取、处理、加载类
	├── routes/		# 路径记录与读取类
	├── sensors/		# 多个传感器类
	├── world/
		├── generate_traffic_my.py		# 加载指定地图、生成交通流
		└── manual_control_my.py		# 以pygame方式打开client，进行手动控制以及记录的回放
	├── automatic_control_my.py		# 自动驾驶脚本，传输传感器图像流、数据流到后端服务器
	└── run_scripts.py		# 请忽略掉
└── util/         # 一些组件  
```

---
