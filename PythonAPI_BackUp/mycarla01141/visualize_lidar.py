import glob
import os
import sys
import requests
from io import BytesIO
from PIL import Image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time
import numpy as np

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('无法导入pygame，请确保已安装pygame包')


class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()


class SensorManager:
    def __init__(self, world, sensor_type, transform, attached, sensor_options):
        self.world = world
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('dropoff_general_rate',
                                   lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit',
                                   lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity',
                                   lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_lidar_image)

            return lidar

        else:
            return None

    def save_lidar_image(self, image):
        t_start = self.timer.time()

        disp_size = (800, 800)  # 固定显示大小
        lidar_range = 2.0 * float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # 确保正值
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))

        # 创建一个空图像
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        # 在图像上绘制点
        valid_indices = (lidar_data[:, 0] >= 0) & (lidar_data[:, 0] < disp_size[0]) & \
                        (lidar_data[:, 1] >= 0) & (lidar_data[:, 1] < disp_size[1])
        lidar_data = lidar_data[valid_indices]
        lidar_img[lidar_data[:, 1], lidar_data[:, 0]] = (255, 255, 255)

        # 将图像保存到字节缓冲区
        img_bytes = BytesIO()
        pil_image = Image.fromarray(lidar_img)
        pil_image.save(img_bytes, format="JPEG")
        img_str = img_bytes.getvalue()

        # 将 LiDAR 图像发送到 FastAPI 服务器
        response = requests.post("http://localhost:8001/upload-image/lidar", files={"file": ("lidar_frame.jpg", img_str)})
        print(f"LiDAR 图像响应来自服务器: {response.json()}")

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def destroy(self):
        self.sensor.destroy()


# UPDATE0114：固定到车辆
from mycarla.sensors.car_tracker import find_hero_vehicle


def main():
    """此函数使用传入的参数执行一次测试运行，并连接到 Carla 客户端。"""
    # 连接到 Carla 服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 查找名为 'hero' 的车辆
    vehicle = find_hero_vehicle(world)
    if not vehicle:
        print("未找到 hero 车辆。")
        return

    print(f"找到 hero 车辆: {vehicle.id}")

    # 添加 LiDAR 传感器
    sensor_manager = SensorManager(world, 'LiDAR', carla.Transform(carla.Location(x=0, z=2.4)), vehicle, {
        'channels': '64',
        'range': '100',
        'points_per_second': '1000000',
        'rotation_frequency': '20'
    })
    try:

        # 模拟循环
        call_exit = False
        timer = CustomTimer()
        while True:
            # Carla Tick
            world.tick()

            if call_exit:
                break

            # 让模拟运行
            time.sleep(0.2)
    finally:
        sensor_manager.destroy()
        client.apply_batch([carla.command.DestroyActor(vehicle)])


if __name__ == '__main__':
    main()