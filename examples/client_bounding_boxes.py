#!/usr/bin/env python

# 版权声明
# Copyright (c) 2019 Aptiv
# 遵循MIT许可协议，详见<https://opensource.org/licenses/MIT>

"""
客户端3D边界框显示示例，包含基础车辆控制功能

操作说明：
    W            : 油门加速
    S            : 刹车减速
    AD           : 转向控制
    Space        : 手刹

    ESC          : 退出程序
"""

# ==============================================================================
# -- 添加CARLA模块路径 ----------------------------------------------------------
# ==============================================================================

import glob
import os
import sys

# 自动匹配当前Python版本的预编译库文件
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,  # Python主版本
        sys.version_info.minor,  # Python次版本
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])  # 根据操作系统选择
except IndexError:
    pass  # 如果找不到egg文件则忽略

# ==============================================================================
# -- 导入依赖库 -----------------------------------------------------------------
# ==============================================================================

import carla  # CARLA主库
import weakref  # 弱引用支持
import random  # 随机数生成

# 检查并导入Pygame库
try:
    import pygame
    from pygame.locals import K_ESCAPE, K_SPACE, K_a, K_d, K_s, K_w  # 键盘按键常量
except ImportError:
    raise RuntimeError('需要安装pygame库')

# 检查并导入NumPy库
try:
    import numpy as np
except ImportError:
    raise RuntimeError('需要安装numpy库')

# 视图参数配置
VIEW_WIDTH = 1920 // 2  # 视图宽度（原分辨率一半）
VIEW_HEIGHT = 1080 // 2  # 视图高度（原分辨率一半）
VIEW_FOV = 90  # 相机视野角度

BB_COLOR = (248, 64, 24)  # 边界框颜色（橙色）


# ==============================================================================
# -- 客户端3D边界框处理类 -------------------------------------------------------
# ==============================================================================

class ClientSideBoundingBoxes(object):
    """
    客户端3D边界框处理类，负责：
    - 生成车辆3D边界框
    - 在Pygame表面绘制边界框
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        获取所有车辆的边界框（过滤掉相机后方的物体）
        参数：
            vehicles : 车辆对象列表
            camera   : 相机传感器对象
        返回：
            处理后的边界框列表
        """
        # 生成所有车辆的边界框
        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # 过滤掉z轴坐标<=0的点（位于相机后方）
        return [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        在Pygame显示表面绘制3D边界框
        参数：
            display        : Pygame显示表面
            bounding_boxes : 要绘制的边界框列表
        """
        # 创建透明表面用于绘制
        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))  # 设置黑色为透明色

        for bbox in bounding_boxes:
            # 将3D坐标转换为2D屏幕坐标
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]

            # 绘制12条边界线
            # 底面四边形
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # 顶面四边形
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # 连接底面和顶面的立柱
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])

        # 将绘制好的表面叠加到主显示表面
        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        计算单个车辆的3D边界框（相机坐标系）
        参数：
            vehicle : 目标车辆对象
            camera  : 相机传感器对象
        返回：
            8个角点的相机坐标系坐标（形状：8x3）
        """
        # 生成车辆本地坐标系下的边界框点
        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        # 转换到相机坐标系
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        # 调整坐标轴顺序（Y->Y, Z->-Z, X->X）
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        # 应用相机内参矩阵进行投影
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        # 透视除法（归一化处理）
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        生成车辆本地坐标系下的边界框8个顶点
        参数：
            vehicle : 目标车辆对象
        返回：
            8个顶点的齐次坐标（形状：8x4）
        """
        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent  # 获取边界框半长宽高

        # 定义8个顶点的本地坐标（齐次坐标）
        # 底面四个点（z=-extent.z）
        cords[0, :] = [extent.x, extent.y, -extent.z, 1]  # 右前下
        cords[1, :] = [-extent.x, extent.y, -extent.z, 1]  # 左前下
        cords[2, :] = [-extent.x, -extent.y, -extent.z, 1]  # 左后下
        cords[3, :] = [extent.x, -extent.y, -extent.z, 1]  # 右后下
        # 顶面四个点（z=extent.z）
        cords[4, :] = [extent.x, extent.y, extent.z, 1]  # 右前上
        cords[5, :] = [-extent.x, extent.y, extent.z, 1]  # 左前上
        cords[6, :] = [-extent.x, -extent.y, extent.z, 1]  # 左后上
        cords[7, :] = [extent.x, -extent.y, extent.z, 1]  # 右后上
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        坐标转换：车辆本地坐标系 -> 世界坐标系 -> 传感器坐标系
        参数：
            cords  : 顶点坐标数组
            vehicle: 目标车辆对象
            sensor : 传感器对象
        返回：
            传感器坐标系下的坐标
        """
        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        坐标转换：车辆本地坐标系 -> 世界坐标系
        参数：
            cords  : 顶点坐标数组
            vehicle: 目标车辆对象
        返回：
            世界坐标系下的坐标
        """
        # 获取车辆边界框的变换矩阵
        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        # 获取车辆的世界变换矩阵
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        # 组合变换矩阵
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        # 应用变换矩阵
        return np.dot(bb_world_matrix, np.transpose(cords))

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        坐标转换：世界坐标系 -> 传感器坐标系
        参数：
            cords  : 顶点坐标数组
            sensor : 传感器对象
        返回：
            传感器本地坐标系下的坐标
        """
        # 获取传感器的世界变换矩阵并求逆
        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        # 应用逆变换矩阵
        return np.dot(world_sensor_matrix, cords)

    @staticmethod
    def get_matrix(transform):
        """
        将CARLA变换对象转换为4x4变换矩阵
        参数：
            transform : carla.Transform对象
        返回：
            4x4变换矩阵（包含旋转和平移）
        """
        rotation = transform.rotation
        location = transform.location

        # 计算旋转角度的三角函数值
        c_y = np.cos(np.radians(rotation.yaw))  # 偏航角（绕Z轴）
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))  # 横滚角（绕X轴）
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))  # 俯仰角（绕Y轴）
        s_p = np.sin(np.radians(rotation.pitch))

        # 构建4x4变换矩阵
        matrix = np.identity(4)
        # 平移部分
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        # 旋转部分（Z-Y-X欧拉角顺序）
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- 基础同步客户端类 -----------------------------------------------------------
# ==============================================================================

class BasicSynchronousClient(object):
    """
    基础同步客户端实现，主要功能：
    - 管理CARLA连接
    - 生成和控制主车辆
    - 管理相机传感器
    - 处理用户输入
    - 渲染游戏画面
    """

    def __init__(self):
        self.client = None  # CARLA客户端实例
        self.world = None  # 世界对象
        self.camera = None  # 相机传感器
        self.car = None  # 主控车辆
        self.display = None  # Pygame显示表面
        self.image = None  # 相机图像缓存
        self.capture = True  # 图像捕获标志

    def camera_blueprint(self):
        """创建并配置相机蓝图"""
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # 设置相机参数
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """设置同步/异步模式"""
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode  # True启用同步模式
        self.world.apply_settings(settings)

    def setup_car(self):
        """生成主控车辆"""
        # 随机选择车辆蓝图和生成点
        car_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """设置并绑定相机传感器"""
        # 相机相对车辆的安装位置（后方5.5米，高度2.8米，俯角15度）
        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(
            self.camera_blueprint(),
            camera_transform,
            attach_to=self.car
        )
        # 使用弱引用避免循环引用
        weak_self = weakref.ref(self)
        # 注册图像回调函数
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        # 计算相机内参矩阵（用于3D到2D投影）
        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0  # 光心X坐标
        calibration[1, 2] = VIEW_HEIGHT / 2.0  # 光心Y坐标
        # 计算焦距 f = width/(2*tan(fov/2))
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration  # 存储到相机对象

    def control(self, car):
        """
        处理键盘输入控制车辆
        返回True表示需要退出，否则False
        """
        keys = pygame.key.get_pressed()

        if keys[K_ESCAPE]:  # ESC退出
            return True

        control = car.get_control()
        control.throttle = 0  # 重置油门

        # 处理油门/刹车
        if keys[K_w]:  # W加速
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:  # S倒车
            control.throttle = 1
            control.reverse = True

        # 处理转向
        if keys[K_a]:  # A左转
            control.steer = max(-1.0, min(control.steer - 0.05, 0))
        elif keys[K_d]:  # D右转
            control.steer = min(1.0, max(control.steer + 0.05, 0))
        else:  # 自动回正
            control.steer = 0

        control.hand_brake = keys[K_SPACE]  # 空格手刹

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """图像回调函数，存储最新图像"""
        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False  # 标记已捕获

    def render(self, display):
        """将相机图像渲染到Pygame表面"""
        if self.image is not None:
            # 将原始字节数据转换为NumPy数组
            array = np.frombuffer(self.image.raw_data, dtype=np.uint8)
            # 重塑为图像尺寸（高度x宽度x4通道）
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]  # 去除Alpha通道
            array = array[:, :, ::-1]  # 将BGR转换为RGB
            # 创建Pygame表面并显示
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def game_loop(self):
        """主游戏循环"""
        try:
            # 初始化Pygame
            pygame.init()

            # 连接CARLA服务端
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            # 设置车辆和相机
            self.setup_car()
            self.setup_camera()

            # 初始化Pygame显示
            self.display = pygame.display.set_mode(
                (VIEW_WIDTH, VIEW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF  # 硬件加速+双缓冲
            )
            pygame_clock = pygame.time.Clock()

            # 启用同步模式
            self.set_synchronous_mode(True)
            # 获取所有车辆对象
            vehicles = self.world.get_actors().filter('vehicle.*')

            # 主循环
            while True:
                self.world.tick()  # 推进仿真世界

                self.capture = True  # 允许捕获下一帧
                pygame_clock.tick_busy_loop(20)  # 限制帧率约20FPS

                # 渲染相机图像
                self.render(self.display)
                # 获取并绘制边界框
                bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes)

                # 刷新显示
                pygame.display.flip()

                # 处理事件
                pygame.event.pump()
                if self.control(self.car):  # 检测退出信号
                    return

        finally:
            # 清理资源
            self.set_synchronous_mode(False)  # 退出前恢复异步模式
            if self.camera is not None:
                self.camera.destroy()
            if self.car is not None:
                self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- 主函数 --------------------------------------------------------------------
# ==============================================================================

def main():
    """程序入口函数"""
    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('程序退出')


if __name__ == '__main__':
    main()