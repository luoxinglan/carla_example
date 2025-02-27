import glob
import os
import sys
import requests
from io import BytesIO
from PIL import Image
import queue
import threading
import time
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import random
import pygame
from pygame.locals import K_ESCAPE, K_q


class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()


class SensorManager:
    def __init__(self, world, sensor_type, transform, attached, sensor_options, index, image_queue):
        self.world = world
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()
        self.index = index  # 新增的下标
        self.image_queue = image_queue  # 共享队列
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

        elif sensor_type == 'SemanticSegmentationCamera':
            seg_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            seg_bp.set_attribute('image_size_x', '350')
            seg_bp.set_attribute('image_size_y', '260')

            for key in sensor_options:
                seg_bp.set_attribute(key, sensor_options[key])

            seg_camera = self.world.spawn_actor(seg_bp, transform, attach_to=attached)

            seg_camera.listen(self.save_semantic_image)

            return seg_camera

        elif sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '200')  # 设置图像宽度为200像素
            camera_bp.set_attribute('image_size_y', '150')  # 设置图像高度为150像素

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera

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

        # 将 LiDAR 图像放入对应队列
        self.image_queue.put(('lidar', img_str))
        print(f"LiDAR图像已放入队列")

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def save_semantic_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.CityScapesPalette)  # 转换为CityScapes调色板
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]

        # 将数组转换为图像并保存到字节缓冲区
        img_bytes = BytesIO()
        pil_image = Image.fromarray(array)
        pil_image.save(img_bytes, format="JPEG")
        img_str = img_bytes.getvalue()

        # 将语义分割图像放入对应队列
        self.image_queue.put(('semantic', img_str))
        print(f"语义分割图像已放入队列")

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)  # 将图像转换为原始格式
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))  # 将图像数据转换为numpy数组
        array = np.reshape(array, (image.height, image.width, 4))  # 重塑数组形状
        array = array[:, :, :3]  # 只取RGB通道
        array = array[:, :, ::-1]  # BGR to RGB

        # 将数组转换为图像并保存到字节缓冲区
        img_bytes = BytesIO()
        pil_image = Image.fromarray(array)
        pil_image.save(img_bytes, format="JPEG")
        img_str = img_bytes.getvalue()

        # 将RGB图像放入对应队列
        self.image_queue.put(('rgb', self.index, img_str))
        print(f"RGB图像{self.index}已放入队列")

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def destroy(self):
        self.sensor.destroy()  # 销毁传感器


def stitch_images(images):
    """拼接四张图像成一张2x2的图像"""
    images_np = [np.array(Image.open(BytesIO(img))) for img in images]
    height, width, _ = images_np[0].shape

    row1 = np.concatenate((images_np[0], images_np[1]), axis=1)
    row2 = np.concatenate((images_np[2], images_np[3]), axis=1)
    stitched_image = np.concatenate((row1, row2), axis=0)

    print("images stitched into shape(stitch_images):", stitched_image.shape)

    return Image.fromarray(stitched_image)


def upload_stitched_image(stitched_image):
    """上传拼接后的图像到FastAPI服务器"""
    print("upload_stitched_image......")
    img_bytes = BytesIO()
    stitched_image.save(img_bytes, format="JPEG")
    img_str = img_bytes.getvalue()

    response = requests.post("http://localhost:8001/upload-image/multicamera/0",
                             files={"file": ("stitched_frame.jpg", img_str)})
    print(f"拼接图像响应来自服务器: {response.json()}")


def upload_image(data, endpoint):
    """上传图像到FastAPI服务器"""
    img_bytes = BytesIO()
    data.save(img_bytes, format="JPEG")
    img_str = img_bytes.getvalue()

    response = requests.post(endpoint,
                             files={"file": ("frame.jpg", img_str)})
    print(f"图像响应来自服务器: {response.json()}")


def process_images(image_queue, vehicle_data_queue):
    """从队列中取出图像并进行拼接和上传"""
    images = {'lidar': None, 'semantic': None, 'rgb': [None] * 4}

    while True:
        try:
            item = image_queue.get(timeout=1)
            if item[0] == 'lidar':
                images['lidar'] = item[1]
            elif item[0] == 'semantic':
                images['semantic'] = item[1]
            elif item[0] == 'rgb':
                _, index, img_str = item
                images['rgb'][index] = img_str

            # 检查是否所有图像都已经收集到
            if all(images['rgb']) and images['lidar'] and images['semantic']:
                # 拼接RGB图像
                stitched_rgb_image = stitch_images(images['rgb'])
                upload_stitched_image(stitched_rgb_image)

                # 上传LiDAR图像
                lidar_pil_image = Image.open(BytesIO(images['lidar']))
                upload_image(lidar_pil_image, "http://localhost:8001/upload-image/lidar/5")

                # 上传语义分割图像
                semantic_pil_image = Image.open(BytesIO(images['semantic']))
                upload_image(semantic_pil_image, "http://localhost:8001/upload-image/semantic/5")

                # 重置图像槽位
                images = {'lidar': None, 'semantic': None, 'rgb': [None] * 4}
        except queue.Empty:
            print("队列为空，等待更多图像...")
        except Exception as e:
            print(f"处理图像时发生错误: {e}")

        try:
            data = vehicle_data_queue.get(timeout=1)
            # 打印或导出这些数据
            print(f"Hero Vehicle Data:")
            print(json.dumps(data, indent=4))

            # 发送数据到 FastAPI 服务器
            response = requests.post("http://localhost:8001/upload-vehicle-data/", json=data)
            print(f"Response from server: {response.json()}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"处理车辆数据时发生错误: {e}")


def calculate_ttc(hero_location, hero_velocity, other_location, other_velocity):
    """
    计算两个车辆之间的 TTC (Time to Collision)。
    """
    # 相对位置和相对速度
    relative_position = np.array([
        other_location.x - hero_location.x,
        other_location.y - hero_location.y,
        other_location.z - hero_location.z
    ])

    relative_velocity = np.array([
        other_velocity.x - hero_velocity.x,
        other_velocity.y - hero_velocity.y,
        other_velocity.z - hero_velocity.z
    ])

    # 相对速度的模
    speed_rel = np.linalg.norm(relative_velocity)

    if speed_rel == 0:
        return float('inf')  # 如果相对速度为零，TTC 无穷大

    # 相对距离的模
    dist_rel = np.dot(relative_position, relative_velocity) / speed_rel

    # TTC 计算
    ttc = dist_rel / speed_rel

    return max(ttc, 0)  # 确保 TTC 不为负数


def find_hero_vehicle(world):
    """
    查找名为 'hero' 的车辆。
    """
    vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in vehicles:
        if vehicle.is_alive and vehicle.attributes.get('role_name') == 'hero':
            return vehicle
    return None


def update_other_vehicles(world, hero_vehicle_id):
    """更新其他车辆的位置和速度"""
    all_vehicles = world.get_actors().filter('vehicle.*')
    other_vehicles = []
    for vehicle in all_vehicles:
        if vehicle.id != hero_vehicle_id and vehicle.is_alive:
            location = vehicle.get_location()
            velocity = vehicle.get_velocity()
            other_vehicles.append({
                'id': vehicle.id,
                'location': location,
                'velocity': velocity
            })
    return other_vehicles


def track_vehicle(world, vehicle_data_queue, other_vehicles_dict):
    """跟踪车辆并计算 TTC"""
    hero_vehicle = find_hero_vehicle(world)
    if not hero_vehicle:
        print("未找到 hero 车辆。")
        return

    print(f"找到 hero 车辆: {hero_vehicle.id}")

    while True:
        world.tick()
        time.sleep(0.1)

        # 获取 hero 车辆的位置、速度等数据
        hero_location = hero_vehicle.get_location()
        hero_velocity = hero_vehicle.get_velocity()
        acceleration = hero_vehicle.get_acceleration()

        v_speed = np.array([hero_velocity.x, hero_velocity.y, hero_velocity.z])
        velocity_modulus = np.linalg.norm(v_speed)

        v_acc = np.array([acceleration.x, acceleration.y, acceleration.z])
        acceleration_modulus = np.linalg.norm(v_acc)

        # 更新最近车辆信息
        min_ttc = float('inf')
        closest_vehicle = None

        for vehicle_info in other_vehicles_dict.values():
            other_location = vehicle_info['location']
            other_velocity = vehicle_info['velocity']

            ttc = calculate_ttc(
                hero_location,
                hero_velocity,
                other_location,
                other_velocity
            )

            if ttc is not None and ttc < min_ttc:
                min_ttc = ttc
                closest_vehicle = vehicle_info

        # 处理 min_ttc
        if min_ttc == float('inf'):
            min_ttc = None

        # 构造数据字典
        nearest_vehicle_id = closest_vehicle['id'] if closest_vehicle else None
        nearest_vehicle_name = closest_vehicle['type_id'].split('.')[1] if closest_vehicle else None

        data = {
            "timestamp": time.time(),
            "velocity_modulus": velocity_modulus,
            "acceleration_modulus": acceleration_modulus,
            "ttc": min_ttc,  # 最小 TTC
            "nearest_vehicle_id": nearest_vehicle_id,  # 最近车辆的ID
            "nearest_vehicle_name": nearest_vehicle_name  # 最近车辆的名称
        }

        # 放入车辆数据队列
        vehicle_data_queue.put(data)


def run_simulation(client, call_exit=None):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    vehicle = None
    timer = CustomTimer()

    try:

        # Getting the world and
        world = client.get_world()
        original_settings = world.get_settings()

        # 创建一个共享队列来存储每个摄像头的图像
        image_queue = queue.Queue(maxsize=16)
        vehicle_data_queue = queue.Queue(maxsize=16)

        sensor_managers = [
            SensorManager(world, 'LiDAR', carla.Transform(carla.Location(x=0, z=2.4)), find_hero_vehicle(world), {
                'channels': '64',
                'range': '100',
                'points_per_second': '1000000',
                'rotation_frequency': '20'
            }, index=None, image_queue=image_queue),

            SensorManager(world, 'SemanticSegmentationCamera',
                          carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=0)), find_hero_vehicle(world),
                          {}, index=None, image_queue=image_queue),

            SensorManager(world, 'RGBCamera',
                          carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90)),
                          find_hero_vehicle(world), {}, 0, image_queue),
            SensorManager(world, 'RGBCamera',
                          carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)),
                          find_hero_vehicle(world), {}, 1, image_queue),
            SensorManager(world, 'RGBCamera',
                          carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)),
                          find_hero_vehicle(world), {}, 2, image_queue),
            SensorManager(world, 'RGBCamera',
                          carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=180)),
                          find_hero_vehicle(world), {}, 3, image_queue)
        ]

        # 启动图像处理线程
        processing_thread = threading.Thread(target=process_images, args=(image_queue, vehicle_data_queue), daemon=True)
        processing_thread.start()

        # 初始化 hero 车辆
        hero_vehicle = find_hero_vehicle(world)
        if not hero_vehicle:
            print("未找到 hero 车辆。")
            return

        print(f"找到 hero 车辆: {hero_vehicle.id}")

        # 初始化其他车辆字典
        other_vehicles_dict = {}

        # 定期更新其他车辆信息的线程
        def update_other_vehicles_periodically():
            nonlocal other_vehicles_dict
            while True:
                updated_vehicles = update_other_vehicles(world, hero_vehicle.id)
                new_vehicles_dict = {}
                for vehicle in updated_vehicles:
                    new_vehicles_dict[vehicle['id']] = vehicle
                other_vehicles_dict = new_vehicles_dict
                time.sleep(1.0)  # 每秒更新一次

        updating_thread = threading.Thread(target=update_other_vehicles_periodically, daemon=True)
        updating_thread.start()

        # 启动车辆跟踪线程
        tracking_thread = threading.Thread(target=track_vehicle, args=(world, vehicle_data_queue, other_vehicles_dict), daemon=True)
        tracking_thread.start()

        # Simulation loop
        call_exit = False

        while True:
            # Carla Tick
            world.tick()

            if call_exit:
                break

            # 让模拟运行
            time.sleep(0.3)

    finally:
        for sensor_manager in sensor_managers:
            sensor_manager.destroy()  # 销毁所有传感器
        world.apply_settings(original_settings)  # 恢复原始设置


def main():
    try:
        # 连接到 Carla 服务器
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        run_simulation(client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()



