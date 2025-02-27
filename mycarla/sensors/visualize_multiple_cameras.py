import glob
import os
import queue
import sys
from io import BytesIO
from PIL import Image
import requests

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
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


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
    # def __init__(self, world, sensor_type, transform, attached, sensor_options, index):
        self.world = world
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()
        self.index = index  # 新增的下标
        self.image_queue = image_queue  # 共享队列#UPDATE0117：拼接图像
        self.time_processing = 0.0
        self.tics_processing = 0

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '200')  # 设置图像宽度为800像素
            camera_bp.set_attribute('image_size_y', '150')  # 设置图像高度为600像素

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera

        else:
            return None

    def get_sensor(self):
        return self.sensor

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
        self.image_queue.put((self.index, img_str))
        print(f"RGB图像{self.index}已放入队列")

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def destroy(self):
        self.sensor.destroy()  # 销毁传感器

# UPDATE0117：拼接图像
def stitch_images(images):
    """拼接四张图像成一张2x2的图像"""
    images_np = [np.array(Image.open(BytesIO(img))) for img in images]
    height, width, _ = images_np[0].shape

    row1 = np.concatenate((images_np[0], images_np[1]), axis=1)
    row2 = np.concatenate((images_np[2], images_np[3]), axis=1)
    stitched_image = np.concatenate((row1, row2), axis=0)

    print("images stitched into shape(stitch_images):",stitched_image.shape)

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

def process_images(image_queue):
    """从队列中取出图像并进行拼接和上传"""
    images = [None] * 4  # 初始化四个槽位

    while True:
        try:
            index, img_str = image_queue.get(timeout=1)
            images[index] = img_str

            # 检查是否所有四个图像都已经收集到
            if all(images):
                stitched_image = stitch_images(images)# 拼接图像
                upload_stitched_image(stitched_image)
                images = [None] * 4  # 重置图像槽位
                print("# 重置图像槽位")
        except queue.Empty:
            print("队列为空，等待更多图像...")
        except Exception as e:
            print(f"处理图像时发生错误: {e}")

# UPDATE0114：固定到车辆
from mycarla.sensors.car_tracker import find_hero_vehicle
import threading


def run_simulation(client, call_exit=None):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    vehicle = None
    vehicle_list = []
    timer = CustomTimer()

    try:

        # Getting the world and
        world = client.get_world()
        vehicle = find_hero_vehicle(world)
        original_settings = world.get_settings()

        # 查找名为 'hero' 的车辆
        vehicle = find_hero_vehicle(world)
        if not vehicle:
            print("未找到 hero 车辆。")
            return

        print(f"找到 hero 车辆: {vehicle.id}")

        # 创建一个共享队列来存储每个摄像头的图像
        image_queue = queue.Queue(maxsize=16)

        sensor_managers = [
            SensorManager(world, 'RGBCamera',
                          carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90)),
                          vehicle, {}, 0, image_queue),
            SensorManager(world, 'RGBCamera',
                          carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)),
                          vehicle, {}, 1, image_queue),
            SensorManager(world, 'RGBCamera',
                          carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)),
                          vehicle, {}, 2, image_queue),
            SensorManager(world, 'RGBCamera',
                          carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=180)),
                          vehicle, {}, 3, image_queue)
        ]

        # 启动图像处理线程
        processing_thread = threading.Thread(target=process_images, args=(image_queue,), daemon=True)
        processing_thread.start()

        # Simulation loop
        call_exit=False

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
