import argparse
import glob
import json
import os
import pathlib
import sys
import requests
from io import BytesIO
from PIL import Image
import queue
import threading

from numpy.f2py.crackfortran import endifs

from mycarla.routes.routes_recorder import RealtimeLocationLogger
# from car_tracker import calculate_ttc
import time
import numpy as np

from mycarla.replay.process_log import process_record

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
def send_vehicle_data(vehicle, file_path, index, previous_state):
    """
    Read car JSON data from file
    :param file_path: CARLA client instance
    :param index: Name of the map to load (e.g., 'Town01', 'Town02')
    """
    acceleration = 0
    control = vehicle.get_control()
    # print(control)
    # print("vehicle.get_velocity()", vehicle.get_velocity())
    # print("vehicle.get_acceleration()", vehicle.get_acceleration())
    current_location = vehicle.get_location()
    current_time = time.time()

    # 初始化默认值
    speed = 0.0
    acceleration = 0.0
    new_velocity = None
    # 解包前次状态
    prev_location, prev_velocity, prev_time = previous_state

    if prev_location and prev_time:
        delta_time = current_time - prev_time

        # 避免除零错误
        if delta_time > 1e-6:  # 0.000001秒阈值
            # 计算三维速度向量
            delta_location = current_location - prev_location
            new_velocity = delta_location / delta_time
            speed = new_velocity.length()  # 标量速度

            # 计算加速度
            if prev_velocity:
                print(f"prev_velocity: {prev_velocity}, new_velocity: {new_velocity}")
                delta_velocity = new_velocity.length() - prev_velocity.length()
                acceleration = delta_velocity / delta_time  # 加速度标量值
    data = {
        "time": current_time,
        "speed": round(speed, 3),
        "acceleration": round(acceleration, 3),
        "min_ttc": 0,
        "steering": control.steer,
        "throttle": control.throttle,
        "brake": control.brake,
        "handbrake": control.hand_brake,
        "gear": control.gear,
    }
    print(data)
    send_vehicle_data_for_1(data)

    return current_location, new_velocity, current_time
    # # 读取车辆数据文件
    # try:
    #     with open(file_path, 'r') as f:
    #         vehicle_data = json.load(f)
    # except Exception as e:
    #     print(f"读取文件有问题{e}")
    #
    # # 遍历所有数据点
    # data_point = vehicle_data[index]
    # try:
    #     send_vehicle_data_for_1(data_point)
    # except requests.exceptions.RequestException as e:
    #     print(f"请求异常：{str(e)}")


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
        # print(json.dumps(data_point, indent=9))
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
        #     print(f"成功发送：时间点 {data_point['time']}")
        #     print(f"响应内容: {response.json()}")

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


def draw_route(world, vehicle, route, color=carla.Color(0, 255, 0), life_time=0, index=0):
    """
    绘制路线
    :param world:world
    :param vehicle:车辆
    :param route:
    :param index:下标呗
    :param color:路线的颜色
    :param life_time:图形在渲染时的生命周期（秒），建议设置为与仿真时间步长匹配（如60秒），避免路径过早消失。
    :return:None
    """
    # for i in range(index, len(route) - 1):
    for i in range(0, len(route) - 1):
        print(
            f"开始绘制路线，route ({index},{len(route) - 1}) in draw_route, ({route[i].transform.location.x},{route[i].transform.location.y})",
            i)
        # 绘制箭头连接相邻两个waypoint
        # world.debug.draw_arrow(
        #     begin=route[i].transform.location + carla.Location(z=5),
        #     end=route[i + 1].transform.location + carla.Location(z=5),
        #     thickness=0,
        #     arrow_size=0.1,  # 调整箭头和点的大小以提高可视性。
        #     color=color,
        #     life_time=life_time
        # )
        print(vehicle.get_transform())
        # 在waypoint位置绘制标记点
        world.debug.draw_point(
            location=route[i].transform.location + carla.Location(z=5),
            size=0.1,
            color=color,
            life_time=life_time
        )


def draw_permanent_route(world, vehicle, route, color=carla.Color(0, 255, 0), life_time=0, index=0):
    """
    绘制路线
    :param world:world
    :param vehicle:车辆
    :param index:下标呗
    :param color:路线的颜色
    :param life_time:图形在渲染时的生命周期（秒），建议设置为与仿真时间步长匹配（如60秒），避免路径过早消失。
    :return:None
    """

    # print(f"路径点{index}已被添加")
    # print(vehicle.get_transform())
    # print(vehicle.get_transform().location)
    # print(route[index].transform.location)
    # 在waypoint位置绘制标记点
    world.debug.draw_point(
        location=vehicle.get_transform().location + carla.Location(z=0.1),
        size=0.05,
        color=color,
        life_time=life_time
    )


def run_simulation(client, frame_count=0, replay=False, start=0, duration=0, camera=0, record_file='recording.log',
                   json_file='../logs/vehicle_status0306.json', is_route=False, route_file=''):
    """
    This function performs one test run using the args parameters
    and connecting to the carla client passed.
    :param client: carla client
    :param frame_count: number of frames
    :param replay: whether to replay the last frame
    :param start: start time
    :param duration: duration
    :param camera: camera id
    :param record_file: record file
    :param json_file: json file
    :param is_route: 是否存在路径文件
    :param route_file: 路径文件的位置
    :return: None
    """

    vehicle = None
    timer = CustomTimer()
    # start_time = time.time()  # 记录模拟开始时间

    try:

        # Getting the world and settings
        world = client.get_world()
        original_settings = world.get_settings()
        settings = original_settings
        # settings.synchronous_mode = True
        # settings.fixed_delta_seconds = 0.05  # 仿真时间步长
        world.apply_settings(settings)

        # 读取并加载地图
        if replay:
            print("Loading recording...")
            # TODO:recording.log
            client.replay_file(record_file, start, duration, camera)
            # print(client.show_recorder_file_info(record_file))
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

        if not is_route:
            print("路线信息不存在")
            logger = RealtimeLocationLogger(route_file, buffer_size=50)
        else:  # 如果路线读取出来了，绘制路线
            logger = RealtimeLocationLogger(route_file, is_route=True)
            print("路线信息存在")
            waypoint_list = logger.load_waypoints_from_csv(world)
            # for waypoint in waypoint_list:
            #     print(waypoint.transform)
            print("waypoint_list", len(waypoint_list))
            # draw_route(world,index=, waypoint_list,life_time=duration)

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
        start_time_for_cycle = timer.time()  # 启动simulation的时间
        index = 0
        key = True
        previous_state = (None, None, None)
        # draw_route(world, vehicle, waypoint_list, index=index, life_time=duration)
        while True:

            frame_id = world.tick()  # 推进仿真时间0.05秒
            # print("vehicle.get_velocity()", vehicle.get_velocity())
            # print("vehicle.get_acceleration()", vehicle.get_acceleration())
            # print("vehicle.get_angular_velocity()", vehicle.get_angular_velocity())

            previous_state = send_vehicle_data(vehicle, json_file, index, previous_state)  # TODO: json

            if key:
                draw_permanent_route(world, vehicle, waypoint_list, index=index, life_time=0)
            # TODO: 在这里调用函数，检查是否存在route文件，没有的话就记录，在下一次运行的时候读取。
            if not is_route:
                logger.record(vehicle)
                # 每10秒强制写入一次
                if time.time() % 10 < 0.1:
                    logger.flush()

            index += 1

            if replay:

                # TODO 帧
                # if index >= frame_count:
                if timer.time() - start_time_for_cycle >= duration:
                    # print(f"Duration of {duration} seconds reached. Restarting replay.")
                    client.replay_file(record_file, start, duration, camera)
                    start_time_for_cycle = timer.time()  # 启动simulation的时间
                    index = 0
                    key = False
                    # draw_route(world,vehicle, waypoint_list, index=index, life_time=duration)

                    if not is_route:
                        logger.flush()  # 写入剩余的数据
                        break

            # print(f"Calculation for {frame_id} is done.")

            if call_exit:
                break

            # time.sleep(0.05)






    finally:
        for sensor_manager in sensor_managers:
            sensor_manager.destroy()  # 销毁所有传感器
        world.apply_settings(original_settings)  # 恢复原始设置
        # UPDATE0217
        if replay:
            print("Replay stopped.")


def run_recording_route(client, frame_count=0, replay=False, start=0, duration=0, camera=0, record_file='recording.log',
                        is_route=False, route_file=''):
    """
    This function performs one test run using the args parameters
    and connecting to the carla client passed.
    :param client: carla client
    :param frame_count: number of frames
    :param replay: whether to replay the last frame
    :param start: start time
    :param duration: duration
    :param camera: camera id
    :param record_file: record file
    :param is_route: 是否存在路径文件
    :param route_file: 路径文件的位置
    :return: None
    """

    timer = CustomTimer()
    # start_time = time.time()  # 记录模拟开始时间

    try:

        # Getting the world and settings
        world = client.get_world()
        original_settings = world.get_settings()
        settings = original_settings
        settings.synchronous_mode = False
        # settings.fixed_delta_seconds = 0.05  # 仿真时间步长
        world.apply_settings(settings)

        # 读取并加载地图
        if replay:
            print("Loading recording...")
            # TODO:recording.log
            client.replay_file(record_file, start, duration, camera)
            # print(client.show_recorder_file_info(record_file))
            print("Replay and storing the route started.")
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

        if not is_route:
            logger = RealtimeLocationLogger(route_file, buffer_size=50)

        # Simulation loop
        call_exit = False
        # start_time_for_cycle = timer.time()  # 启动simulation的时间
        index = 0
        while True:

            frame_id = world.tick()  # 推进仿真一帧
            # TODO: 在这里调用函数，检查是否存在route文件，没有的话就记录，在下一次运行的时候读取。
            if not is_route:
                logger.record(vehicle)
                # 每10秒强制写入一次
                if time.time() % 10 < 0.1:
                    logger.flush()

            index += 1

            if replay:

                # TODO 帧
                if index >= frame_count:
                    # print(f"Duration of {duration} seconds reached. Restarting replay.")
                    client.replay_file(record_file, start, duration, camera)
                    start_time_for_cycle = timer.time()  # 启动simulation的时间
                    index = 0
                    if not is_route:
                        logger.flush()  # 写入剩余的数据
                        break

            # print(f"Calculation for {frame_id} is done.")

            if call_exit:
                break

    finally:
        world.apply_settings(original_settings)  # 恢复原始设置
        # UPDATE0217
        if replay:
            print("Replay and storing the route stopped.")


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
        # TODO: this is required.回放帧数，由process_record返回值确定。不需要修改默认值，在执行时会更新
        '-c', '--count_of_frames',
        metavar='C',
        default=0,
        type=int,
        help='frames of recorder json data,this can not be ignored in replay mode ')
    argparser.add_argument(
        # TODO: this is required.回放时长，根据帧数计算出来。不需要修改默认值，在执行时会更新
        '-d',
        '--duration',
        metavar='D',
        default=0,
        help='Duration of recording (default: 81.5), replay time is in seconds of duration (default: 0)')
    argparser.add_argument(
        # TODO: this is required.carla回放文件，须加载。运行前需确认。
        '-l', '--log_filename',
        metavar='L',
        default="recording0306.log",
        help='recorder carla data filename (recording.log)')
    argparser.add_argument(
        # TODO: this is required.未处理的可阅读日志文件。运行前需确认。
        '-i', '--input_filename',
        metavar='I',
        default="../logs/record0306.txt",
        help='未处理的可阅读日志文件recorder filename (record0306.txt)')
    argparser.add_argument(
        # TODO: this is required.处理后输出json文件以读取json的路径。运行前需确认。
        '-j', '--json_filename',
        metavar='J',
        default="../logs/vehicle_status0306.json",
        help='recorder json data filename (../logs/vehicle_status0306.json)')
    argparser.add_argument(
        # TODO: this is required.route的路径文件。运行前需确认。
        '-r', '--route_file',
        metavar='R',
        default="../routes/vehicle_route0306.csv",
        help='recorder route_file data filename (../routes/vehicle_route0306.csv)')
    argparser.add_argument(
        '-t',
        '--target_id',
        metavar='T',
        default='141',
        help='hero car id (or index) (default: 141)')

    args = argparser.parse_args()

    # Connect to Carla server
    client = carla.Client(args.server_ip, args.port)
    client.set_timeout(20.0)

    if args.mode == 'replay':
        print("Running in replay mode.")
        args.count_of_frames = process_record(args.input_filename, args.json_filename, args.target_id)

        print("Count of frames: %d" % args.count_of_frames)
        args.duration = (args.count_of_frames - 1) * 0.05

        print("Duration of recording: %d" % args.duration)
        if pathlib.Path(args.route_file).exists():
            run_simulation(client, frame_count=args.count_of_frames, replay=True, duration=args.duration,
                           record_file=args.log_filename, json_file=args.json_filename, is_route=True,
                           route_file=args.route_file)
        else:  # 如果路径文件不存在，就先运行完，存储到route的json文件中，等待下一次运行。
            run_recording_route(client, frame_count=args.count_of_frames, replay=True, duration=args.duration,
                                record_file=args.log_filename, is_route=False,
                                route_file=args.route_file)
            # run_simulation(client, frame_count=args.count_of_frames, replay=True, duration=args.duration,
            #                record_file=args.log_filename, json_file=args.json_filename, is_route=True,
            #                route_file=args.route_file)
    elif args.mode == 'live':
        print("Running in live mode.")
        run_simulation(client, replay=False)


if __name__ == '__main__':
    main()
