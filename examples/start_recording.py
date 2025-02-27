#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# 导入系统相关库
import glob
import os
import sys

# 添加CARLA PythonAPI路径（根据编译生成的.egg文件）
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# 导入CARLA库
import carla

# 导入其他依赖库
import argparse
import random
import time
import logging


def main():
    # 配置命令行参数解析器
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='CARLA服务端IP地址 (默认: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP监听端口 (默认: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='生成的车辆数量 (默认: 10)')
    argparser.add_argument(
        '-d', '--delay',
        metavar='D',
        default=2.0,
        type=float,
        help='车辆生成间隔时间（秒）(默认: 2.0)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='启用安全模式（避免生成易发生事故的车辆）')
    argparser.add_argument(
        '-f', '--recorder_filename',
        metavar='F',
        default="test1.log",
        help='录制日志文件名 (默认: test1.log)')
    argparser.add_argument(
        '-t', '--recorder_time',
        metavar='T',
        default=0,
        type=int,
        help='自动停止录制的时间（秒）')
    args = argparser.parse_args()

    actor_list = []  # 保存所有生成的车辆actor
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    try:
        # 连接CARLA服务端
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        world = client.get_world()

        # 获取所有车辆蓝图
        blueprints = world.get_blueprint_library().filter('vehicle.*')

        # 获取地图生成点并打乱顺序
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        print('找到 %d 个车辆生成点' % len(spawn_points))

        # 初始化计数器
        count = args.number_of_vehicles

        # 开始录制交通场景
        print("开始录制到文件: %s" % client.start_recorder(args.recorder_filename))

        # 安全模式过滤（移除非常规车辆）
        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        # 再次获取有效生成点数量
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # 检查生成点数量是否足够
        if count < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif count > number_of_spawn_points:
            msg = '请求生成 %d 辆车，但只找到 %d 个有效生成点'
            logging.warning(msg, count, number_of_spawn_points)
            count = number_of_spawn_points

        # 准备批量生成命令
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= count:
                break
            # 随机选择车辆蓝图
            blueprint = random.choice(blueprints)

            # 设置车辆颜色（如果有该属性）
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

            # 设置角色名称并添加到生成批次
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        # 执行批量生成命令
        for response in client.apply_batch_sync(batch):
            if response.error:
                logging.error(response.error)
            else:
                actor_list.append(response.actor_id)

        print('成功生成 %d 辆车辆，按 Ctrl+C 退出' % len(actor_list))

        # 根据参数设置录制时间
        if args.recorder_time > 0:
            time.sleep(args.recorder_time)
        else:
            # 持续运行直到手动停止
            while True:
                world.wait_for_tick()
                # 如需降低CPU占用可取消注释下方代码
                # time.sleep(0.1)

    finally:
        # 清理阶段：销毁所有生成的车辆
        print('\n正在销毁 %d 个actors' % len(actor_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in actor_list])

        # 停止录制
        print("停止录制")
        client.stop_recorder()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\n程序执行完毕')