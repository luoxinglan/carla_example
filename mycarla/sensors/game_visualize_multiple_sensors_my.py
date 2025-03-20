#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Script that render multiple sensors in the same pygame window

By default, it renders four cameras, one LiDAR and one Semantic LiDAR.
It can easily be configure for any different number of sensors. 
To do that, check lines 290-308.
"""

import glob
import json
import os
import pathlib
import sys
from io import BytesIO

from PIL import Image

from mycarla.replay.process_log import parse_frame_blocks, parse_block, process_frames, process_record

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
import time
import numpy as np

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

from mycarla.routes.routes_recorder import RealtimeLocationLogger


class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()


class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        # UPDATE0108：设置窗口名称
        pygame.display.set_caption("VISUALIZATION MULTIPLE SENSORS")

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0] / self.grid_size[1]), int(self.window_size[1] / self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None


class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera

        elif sensor_type == 'SemanticSegmentationCamera':
            seg_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            disp_size = self.display_man.get_display_size()
            seg_bp.set_attribute('image_size_x', str(disp_size[0]))
            seg_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                seg_bp.set_attribute(key, sensor_options[key])

            seg_camera = self.world.spawn_actor(seg_bp, transform, attach_to=attached)

            seg_camera.listen(self.save_semantic_image)

            return seg_camera

        elif sensor_type == 'LiDAR':
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

        elif sensor_type == 'SemanticLiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '100')

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_semanticlidar_image)

            return lidar

        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(self.save_radar_image)

            return radar

        else:
            return None

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def save_semantic_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.CityScapesPalette)  # 转换为CityScapes调色板
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def save_lidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0 * float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def save_semanticlidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0 * float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def save_radar_image(self, radar_data):
        t_start = self.timer.time()
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


class ReplayManager:
    def __init__(self, frame_count, replay, start, duration, camera, record_file='recording.log',
                 json_file='../logs/vehicle_status0306.json', is_route=False, route_file='', world=None):
        self.frame_count = frame_count
        self.replay = replay
        self.start = start
        self.duration = duration
        self.camera = camera
        self.record_file = record_file
        self.json_file = json_file
        self.is_route = is_route
        self.route_file = route_file
        self.world = world

    def wait_for_map_load(self):
        """等待地图加载完成"""
        """UPDATE0217解决第一次运行的时候UnboundLocalError: local variable 'sensor_managers' referenced before assignment"""
        timeout = 10  # 最大等待时间（秒）
        start_time = time.time()
        map_loaded = False

        while not map_loaded and (time.time() - start_time) < timeout:
            actors = list(self.world.get_actors())
            vehicles = [actor for actor in actors if actor.type_id.startswith('vehicle')]
            if len(vehicles) > 0:
                map_loaded = True
            else:
                time.sleep(1)  # 每秒检查一次

        if not map_loaded:
            print("地图加载超时，未能找到任何车辆。")
        else:
            print("地图加载完成。")

    def find_hero_vehicle(self):
        """查找名为 'hero' 的车辆"""
        # 一直找，找5秒钟，没找到就退出。找到了返回车辆
        start_time = time.time()
        while True:
            for vehicle in self.world.get_actors().filter('*vehicle*'):
                if vehicle.attributes.get('role_name') == 'hero':
                    print("find_hero_vehicle找到hero车辆。")
                    return vehicle
            if time.time() - start_time > 10:
                print("find_hero_vehicle未能找到任何车辆。")
                return None

    def find_location_points(self):
        if not self.is_route:
            print("路线信息不存在")
            logger = RealtimeLocationLogger(self.route_file, buffer_size=50)
            return logger, None

        else:  # 如果路线读取出来了，绘制路线
            logger = RealtimeLocationLogger(self.route_file, is_route=True)
            print("路线信息存在")
            waypoint_list = logger.load_waypoints_from_csv(self.world)
            # for waypoint in waypoint_list:
            #     print(waypoint.transform)
            print("waypoint_list", len(waypoint_list))
            # draw_route(world,index=, waypoint_list,life_time=duration)
            return logger, waypoint_list

    def draw_permanent_route(self, vehicle, route, color=carla.Color(0, 255, 0), life_time=0, index=0):
        """
        绘制路线
        :param world:world
        :param vehicle:车辆
        :param index:下标呗
        :param color:路线的颜色
        :param life_time:图形在渲染时的生命周期（秒），建议设置为与仿真时间步长匹配（如60秒），避免路径过早消失。
        :return:None
        """

        print(f"路径点{index}已被添加")
        print(vehicle.get_transform())
        print(vehicle.get_transform().location)
        # print(route[index].transform.location)
        # 在waypoint位置绘制标记点
        self.world.debug.draw_point(
            location=vehicle.get_transform().location + carla.Location(z=0.1),
            size=0.05,
            color=color,
            life_time=life_time
        )

    # def get_car_data(self,vehicle):



def run_simulation(args, client, replayer):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()

    try:

        # Getting the world and
        world = client.get_world()
        original_settings = world.get_settings()
        settings = original_settings
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None  # 仿真时间步长
        world.apply_settings(settings)

        # if args.sync:
        #     traffic_manager = client.get_trafficmanager(8000)
        #     settings = world.get_settings()
        #     traffic_manager.set_synchronous_mode(True)
        #     settings.synchronous_mode = True
        #     settings.fixed_delta_seconds = 0.05
        #     world.apply_settings(settings)

        # 读取并加载地图
        if replayer.replay:
            print("Loading recording...")
            # TODO:recording.log
            client.replay_file(replayer.record_file, replayer.start, replayer.duration, replayer.camera)
            # print(client.show_recorder_file_info(record_file))
            print("Replay started.")
            # 等待地图加载完成UPDATE0217
            replayer.wait_for_map_load()

        print("Starting simulation...")
        # 查找名为 'hero' 的车辆
        vehicle = replayer.find_hero_vehicle()
        if vehicle is None:
            print("No hero vehicle.")
            return None  # 没找到车辆，直接终止进程

        # 查找路线文件route_file
        logger, waypoint_list = replayer.find_location_points()

        # # Instanciating the vehicle to which we attached the sensors
        # bp = world.get_blueprint_library().filter('charger_2020')[0]
        # vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
        # vehicle_list.append(vehicle)
        # vehicle.set_autopilot(True)

        # TODO:Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[1, 1], window_size=[args.width, args.height])

        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position, 
        if replayer.replay:
            # SensorManager(world, display_manager, 'RGBCamera',
            #               carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90)),
            #               vehicle, {}, display_pos=[0, 0])
            SensorManager(world, display_manager, 'RGBCamera',
                          carla.Transform(carla.Location(x=-7, z=2.4), carla.Rotation(yaw=+00)),
                          vehicle, {}, display_pos=[0, 0])
            # SensorManager(world, display_manager, 'RGBCamera',
            #               carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)),
            #               vehicle, {}, display_pos=[0, 2])
            # SensorManager(world, display_manager, 'RGBCamera',
            #               carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=180)),
            #               vehicle, {}, display_pos=[1, 1])

            # SensorManager(world, display_manager, 'LiDAR', carla.Transform(carla.Location(x=0, z=2.4)),
            #               vehicle,
            #               {'channels': '64', 'range': '100', 'points_per_second': '250000', 'rotation_frequency': '20'},
            #               display_pos=[1, 0])
            # SensorManager(world, display_manager, 'SemanticSegmentationCamera',
            #               carla.Transform(carla.Location(x=0, z=2.4)),
            #               vehicle, {}, display_pos=[1, 2])
        else:
            return None

        # Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        index = 0
        key = True
        while True:
            # Carla Tick
            # def __init__(self, frame_count, replay, start, duration, camera, record_file='recording.log',
            #              json_file='../logs/vehicle_status0306.json', is_route=False, route_file='', world=None):
            #     self.frame_count = frame_count
            #     self.replay = replay
            #     self.start = start
            #     self.duration = duration
            #     self.camera = camera
            #     self.record_file = record_file
            #     self.json_file = json_file
            #     self.is_route = is_route
            #     self.route_file = route_file
            #     self.world = world
            if args.sync:
                world.tick()
                # send_vehicle_data(json_file, index)  # TODO: json
                if key:
                    replayer.draw_permanent_route(vehicle, waypoint_list, index=index, life_time=0)
                if not replayer.is_route:
                    logger.record(vehicle)
                    # 每10秒强制写入一次
                    if time.time() % 10 < 0.1:
                        logger.flush()
                index += 1

                if replayer.replay:

                    # TODO 帧
                    if index >= replayer.frame_count:
                        # print(f"Duration of {duration} seconds reached. Restarting replay.")
                        client.replay_file(replayer.record_file, replayer.start, replayer.duration, replayer.camera)
                        start_time_for_cycle = timer.time()  # 启动simulation的时间
                        index = 0
                        key = False
                        # draw_route(world,vehicle, waypoint_list, index=index, life_time=duration)

                        if not replayer.is_route:
                            logger.flush()  # 写入剩余的数据
                            break
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

            if call_exit:
                break

            # time.sleep(0.02)

    finally:
        if display_manager:
            display_manager.destroy()

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        world.apply_settings(original_settings)


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')
    argparser.set_defaults(sync=True)
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
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

    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        # TODO: 修改这里的参数列表
        if args.mode == 'replay':
            print("Running in replay mode.")
            args.count_of_frames = process_record(args.input_filename, args.json_filename, args.target_id)

            print("Count of frames: %d" % args.count_of_frames)
            args.duration = (args.count_of_frames - 1) * 0.05

            print("Duration of recording: %d" % args.duration)

            if pathlib.Path(args.route_file).exists():
                replayer = ReplayManager(frame_count=args.count_of_frames, replay=True, start=0, duration=args.duration,
                                         record_file=args.log_filename, json_file=args.json_filename, is_route=True,
                                         route_file=args.route_file, camera=0, world=client.get_world())
                run_simulation(args, client, replayer)
            else:  # 如果路径文件不存在，就先运行完，存储到route的json文件中，等待下一次运行。
                return None
        elif args.mode == 'live':
            print("Running in live mode.")
            replayer = ReplayManager(frame_count=args.count_of_frames, replay=True, start=0, duration=args.duration,
                                     record_file=args.log_filename, json_file=args.json_filename, is_route=True,
                                     route_file=args.route_file, camera=0, world=client.get_world())
            run_simulation(args, client, replayer)


    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
