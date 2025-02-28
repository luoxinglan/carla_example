import argparse
import glob
import json
import os
import sys
import requests
from io import BytesIO
from PIL import Image
import queue
import threading
from car_tracker import calculate_ttc
import time
import numpy as np

from mycarla.sensors.car_tracker import send_vehicle_data

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
        self.world = world
        # self.client=
        self.sensor = None
        self.sensor_options = sensor_options
        self.timer = CustomTimer()
        self.index = index  # 新增的下标
        self.image_queue = image_queue  # 共享队列#UPDATE0214：拼接图像
        self.time_processing = 0.0
        self.tics_processing = 0
        self.start_time = time.time()  # 记录传感器启动时间
        self.data = None
        self.vehicle = attached  # 保存车辆引用以便后续使用
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)

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

            if self.index == 4:
                camera_bp.set_attribute('image_size_x', '1280')  # 设置图像宽度为1280像素
                camera_bp.set_attribute('image_size_y', '720')  # 设置图像高度为720像素

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

        # 将 LiDAR 图像放入对应队列UPDATE0219
        timestamp = time.time() - self.start_time  # 使用相对于启动时间的时间戳
        self.image_queue.put(('lidar', timestamp, img_str))
        # print(f"LiDAR图像已放入队列，时间戳: {timestamp}")

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

        # 图像放入对应队列UPDATE0219
        timestamp = time.time() - self.start_time  # 使用相对于启动时间的时间戳
        self.image_queue.put(('semantic', timestamp, img_str))
        # print(f"语义分割图像已放入队列，时间戳: {timestamp}")

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

        # 将RGB图像放入对应队列UPDATE0219
        timestamp = time.time() - self.start_time  # 使用相对于启动时间的时间戳
        self.image_queue.put(('rgb', self.index, timestamp, img_str))
        # print(f"RGB图像{self.index}已放入队列，时间戳: {timestamp}")

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()  # 销毁传感器


def wait_for_map_load(world):
    """等待地图加载完成"""
    """UPDATE0217解决第一次运行的时候UnboundLocalError: local variable 'sensor_managers' referenced before assignment"""
    timeout = 10  # 最大等待时间（秒）
    start_time = time.time()
    map_loaded = False

    while not map_loaded and (time.time() - start_time) < timeout:
        actors = list(world.get_actors())
        vehicles = [actor for actor in actors if actor.type_id.startswith('vehicle')]
        if len(vehicles) > 0:
            map_loaded = True
        else:
            time.sleep(1)  # 每秒检查一次

    if not map_loaded:
        print("地图加载超时，未能找到任何车辆。")
    else:
        print("地图加载完成。")


def find_hero_vehicle(world):
    """查找名为 'hero' 的车辆"""
    for vehicle in world.get_actors().filter('*vehicle*'):
        if vehicle.attributes.get('role_name') == 'hero':
            print("find_hero_vehicle找到hero车辆。")
            return vehicle
    print("find_hero_vehicle未能找到任何车辆。")
    return None


def stitch_images(images):
    """拼接四张图像成一张2x2的图像"""
    images_np = [np.array(Image.open(BytesIO(img))) for img in images]
    height, width, _ = images_np[0].shape

    row1 = np.concatenate((images_np[0], images_np[1]), axis=1)
    row2 = np.concatenate((images_np[2], images_np[3]), axis=1)
    stitched_image = np.concatenate((row1, row2), axis=0)

    # print("images stitched into shape(stitch_images):", stitched_image.shape)

    return Image.fromarray(stitched_image)


def upload_stitched_image(stitched_image, timestamp):
    """上传拼接后的图像到FastAPI服务器"""
    # print("upload_stitched_image......")
    img_bytes = BytesIO()
    stitched_image.save(img_bytes, format="JPEG")
    img_str = img_bytes.getvalue()

    response = requests.post(
        "http://localhost:8001/upload-image/multicamera/0",
        files={"file": ("stitched_frame.jpg", img_str)},
        data={"timestamp": timestamp}
    )
    # print(f"拼接图像响应来自服务器: {response.json()}")


def upload_image(data, endpoint, timestamp):
    """上传图像到FastAPI服务器"""
    # print("上传图像到FastAPI服务器upload_image......")
    img_bytes = BytesIO()
    data.save(img_bytes, format="JPEG")
    img_str = img_bytes.getvalue()

    response = requests.post(
        endpoint,
        files={"file": ("frame.jpg", img_str)},
        data={"timestamp": timestamp}
    )
    # print(f"图像响应来自服务器: {response.json()}")


# TODO:UPDATE0226:使其能够处理车辆数据。
def send_vehicle_data(file_path, index):
    """
    Read car JSON data from file
    :param file_path: CARLA client instance
    :param index: Name of the map to load (e.g., 'Town01', 'Town02')
    """

    # 读取车辆数据文件
    try:
        with open(file_path, 'r') as f:
            vehicle_data = json.load(f)
    except Exception as e:
        print(f"读取文件有问题{e}")

    # 遍历所有数据点
    data_point = vehicle_data[index]
    try:
        send_vehicle_data_for_1(data_point)
    except requests.exceptions.RequestException as e:
        print(f"请求异常：{str(e)}")


# TODO:UPDATE0226:使其能够处理车辆数据。
def send_vehicle_data_for_1(data_point):
    """
    传送一帧车辆数据到服务器
    """
    # 设置请求头
    headers = {
        "Content-Type": "application/json"
    }
    try:
        print(json.dumps(data_point, indent=9))
        # 发送POST请求
        response = requests.post(
            "http://localhost:8001/upload-vehicle-data/",
            json=data_point,
            headers=headers
        )

        # 检查响应状态
        if response.status_code != 200:
            print(f"发送失败：时间点 {data_point['time']}，状态码 {response.status_code}")
            print(f"响应内容: {response.json()}")
        # else:
            # print(f"成功发送：时间点 {data_point['time']}")
            # print(f"响应内容: {response.json()}")

    except requests.exceptions.RequestException as e:
        print(f"请求异常：{str(e)}")


def start_vehicle_thread(file_path):
    """启动独立线程的入口函数"""
    # 创建守护线程 (主线程退出时自动终止)
    thread = threading.Thread(
        target=send_vehicle_data,
        args=(file_path,),
        daemon=True,
        name="VehicleDataUploadThread"
    )

    # 启动线程
    thread.start()

    # 返回线程对象以便后续控制
    return thread


def process_and_upload_all_images(image_queue, replay=False):
    """从队列中取出图像并进行拼接和上传"""
    if replay:
        images = {'lidar': None, 'semantic': None, 'rgb': [None] * 5, 'vehicle': None, 'timestamps': {}}
    else:
        images = {'lidar': None, 'semantic': None, 'rgb': [None] * 4, 'vehicle': None, 'timestamps': {}}

    while True:
        try:
            if replay:
                item = image_queue.get(timeout=1)
                if item[0] == 'lidar':
                    images['lidar'] = item[2]  # 数据
                    images['timestamps']['lidar'] = item[1]  # 时间戳
                elif item[0] == 'semantic':
                    images['semantic'] = item[2]  # 数据
                    images['timestamps']['semantic'] = item[1]  # 时间戳
                elif item[0] == 'rgb':
                    _, index, timestamp, img_str = item
                    images['rgb'][index] = img_str
                    images['timestamps'][f'rgb_{index}'] = timestamp

                    # 如果是索引为4的RGB图像，则单独上传
                    if index == 4:
                        rgb_pil_image = Image.open(BytesIO(img_str))
                        upload_image(rgb_pil_image, "http://localhost:8001/upload-image/camera/5", timestamp)
                        # print(f"RGB图像{index}已单独上传到 http://localhost:8001/upload-image/camera/5")
            else:
                item = image_queue.get(timeout=1)
                if item[0] == 'lidar':
                    images['lidar'] = item[2]  # 数据
                    images['timestamps']['lidar'] = item[1]  # 时间戳
                elif item[0] == 'semantic':
                    images['semantic'] = item[2]  # 数据
                    images['timestamps']['semantic'] = item[1]  # 时间戳
                elif item[0] == 'rgb':
                    _, index, timestamp, img_str = item
                    images['rgb'][index] = img_str
                    images['timestamps'][f'rgb_{index}'] = timestamp

            # print("#################开始上传图像……process_and_upload_all_images################")
            # 检查是否所有图像都已经收集到
            if all(images['rgb']) and images['lidar'] and images['semantic']:
                # 获取最早的时间戳
                timestamps = list(images['timestamps'].values())
                earliest_timestamp = min(timestamps)

                # 拼接RGB图像
                stitched_rgb_image = stitch_images(images['rgb'])
                upload_stitched_image(stitched_rgb_image, earliest_timestamp)

                # 上传LiDAR图像
                lidar_pil_image = Image.open(BytesIO(images['lidar']))
                upload_image(lidar_pil_image, "http://localhost:8001/upload-image/lidar/5",
                             images['timestamps']['lidar'])

                # 上传语义分割图像
                semantic_pil_image = Image.open(BytesIO(images['semantic']))
                upload_image(semantic_pil_image, "http://localhost:8001/upload-image/semantic/5",
                             images['timestamps']['semantic'])

                # # 上传车辆数据
                # hero_vehicle_data = images['vehicle']
                # upload_vehicle_data(hero_vehicle_data, "http://localhost:8001/upload-vehicle-data/",
                #                     images['timestamps']['vehicle'])
                # 重置图像槽位
                if replay:
                    images = {'lidar': None, 'semantic': None, 'rgb': [None] * 5, 'vehicle': None, 'timestamps': {}}
                else:
                    images = {'lidar': None, 'semantic': None, 'rgb': [None] * 4, 'vehicle': None, 'timestamps': {}}
        except queue.Empty:
            print("队列为空，等待更多图像...")
        except Exception as e:
            print(f"处理图像时发生错误: {e}")


# def cycling_replay(client, start, duration, camera, timer, start_time_for_cycle, replay=False):
#     """
#     此函数应单独搞一个线程，用来计时。
#     """
#     if replay:
#         # Calculate elapsed time
#         elapsed_time = timer.time() - start_time_for_cycle  # 在这一段循环中过去了多久
#
#         # Check if duration has been reached
#         if elapsed_time >= duration:
#             print(f"Duration of {duration} seconds reached. Restarting replay.")
#             client.replay_file("recording.log", start, duration, camera)
#             start_time = timer.time()  # Reset start time for new replay


def run_simulation(client, call_exit=None, replay=False, start=0, duration=0, camera=0):
    """This function performs one test run using the args parameters
    and connecting to the carla client passed.
    """

    vehicle = None
    timer = CustomTimer()
    # start_time = time.time()  # 记录模拟开始时间

    try:

        # Getting the world and settings
        world = client.get_world()
        original_settings = world.get_settings()
        settings = original_settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # 读取并加载地图
        if replay:
            print("Loading recording...")
            # TODO:recording.log
            client.replay_file("recording.log", start, duration, camera)
            # print(client.show_recorder_file_info("recording.log"))
            print("Replay started.")
            # 等待地图加载完成UPDATE0217
            wait_for_map_load(world)

        print("Starting simulation...")

        # 查找名为 'hero' 的车辆
        vehicle = find_hero_vehicle(world)
        while True:
            if vehicle:
                print(f"找到 hero 车辆: {vehicle.id}")
                # print("未找到 hero 车辆。")
                break
            else:
                vehicle = find_hero_vehicle(world)

        # print(f"###找到 hero 车辆: {vehicle.id}###")

        # 创建一个共享队列来存储每个摄像头的图像
        # print("###创建队列image_queue###")
        image_queue = queue.Queue(maxsize=16)  # 键值、时间戳、数据

        # print("###创建队列image_queue完成，即将初始化sensor_managers列表###")
        if replay:
            sensor_managers = [
                SensorManager(world, 'LiDAR', carla.Transform(carla.Location(x=0, z=2.4)), vehicle, {
                    'channels': '64',
                    'range': '100',
                    'points_per_second': '1000000',
                    'rotation_frequency': '20'
                }, index=None, image_queue=image_queue),

                SensorManager(world, 'SemanticSegmentationCamera',
                              carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=0)), vehicle,
                              {}, index=None, image_queue=image_queue),

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
                              vehicle, {}, 3, image_queue),
                # UPDATE0217
                SensorManager(world, 'RGBCamera',
                              carla.Transform(carla.Location(x=-5, z=2.4), carla.Rotation(pitch=-10, yaw=0)),
                              vehicle, {}, 4, image_queue),  # 正对车辆正前方
            ]
        else:
            sensor_managers = [
                SensorManager(world, 'LiDAR', carla.Transform(carla.Location(x=0, z=2.4)), vehicle, {
                    'channels': '64',
                    'range': '100',
                    'points_per_second': '1000000',
                    'rotation_frequency': '20'
                }, index=None, image_queue=image_queue),

                SensorManager(world, 'SemanticSegmentationCamera',
                              carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=0)), vehicle,
                              {}, index=None, image_queue=image_queue),

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

        # print("###初始化sensor_managers列表完成，即将启动图像处理线程###")
        # 启动图像处理线程
        process_and_upload_thread = threading.Thread(target=process_and_upload_all_images, args=(image_queue, replay,),
                                                     daemon=True)
        process_and_upload_thread.start()

        # Simulation loop
        call_exit = False
        # start_time_for_cycle = timer.time()  # 启动simulation的时间
        index = 0
        while True:

            frame_id = world.tick()  # 推进仿真时间0.05秒
            send_vehicle_data("../logs/vehicle_stats0227.json", index)  # TODO: json
            index += 1

            if replay:

                # TODO 帧
                if index >= 2139:
                    # print(f"Duration of {duration} seconds reached. Restarting replay.")
                    client.replay_file("recording.log", start, duration, camera)
                    start_time_for_cycle = timer.time()  # 启动simulation的时间
                    index = 0

            # print(f"Calculation for {frame_id} is done.")

            if call_exit:
                break







    finally:
        for sensor_manager in sensor_managers:
            sensor_manager.destroy()  # 销毁所有传感器
        world.apply_settings(original_settings)  # 恢复原始设置
        # UPDATE0217
        if replay:
            print("Replay stopped.")


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-s',
        '--server-ip',
        metavar='S',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p',
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-m',
        '--mode',
        metavar='M',
        choices=['replay', 'live'],
        default='replay',
        help='Mode to run: replay or live (default: replay)')
    argparser.add_argument(
        '-d',
        '--duration',
        metavar='D',
        default=106.9,  # TODO duration
        type=int,
        help='Duration of recording (default: 81.5), replay time is in seconds of duration (default: 81.5)')

    args = argparser.parse_args()

    # Connect to Carla server
    client = carla.Client(args.server_ip, args.port)
    client.set_timeout(20.0)

    if args.mode == 'replay':
        print("Running in replay mode.")
        run_simulation(client, replay=True, duration=args.duration)
    elif args.mode == 'live':
        print("Running in live mode.")
        run_simulation(client, replay=False)


if __name__ == '__main__':
    main()
