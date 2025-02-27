import math

import carla
import time
import numpy as np
import requests
import json

def find_hero_vehicle(world):
    """查找名为 'hero' 的车辆"""
    for vehicle in world.get_actors().filter('*vehicle*'):
        if vehicle.attributes.get('role_name') == 'hero':
            print("find_hero_vehicle找到hero车辆。")
            return vehicle
    print("find_hero_vehicle未能找到任何车辆。")
    return None


def calculate_ttc(target_pos, target_vel, other_pos, other_vel):
    """
    计算两个车辆之间的 TTC (Time to Collision)。
    """
    dx = other_pos[0] - target_pos[0]
    dy = other_pos[1] - target_pos[1]
    distance = math.hypot(dx, dy)

    if distance == 0:
        return 0.0

    rel_vx = other_vel[0] - target_vel[0]
    rel_vy = other_vel[1] - target_vel[1]

    dir_x = dx / distance
    dir_y = dy / distance
    closing_speed = -(rel_vx * dir_x + rel_vy * dir_y)

    if closing_speed <= 1e-6:  # 避免除以0
        return float('inf')

    return distance / closing_speed


def send_vehicle_data(file_path):
    # 读取车辆数据文件
    with open(file_path, 'r') as f:
        vehicle_data = json.load(f)

    # 设置请求头
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "VehicleDataSender/1.0"
    }

    # 遍历所有数据点
    for data_point in vehicle_data:
        try:
            # 发送POST请求
            response = requests.post(
                "http://localhost:8001/upload-vehicle-data/",
                json=data_point,
                headers=headers
            )

            # 检查响应状态
            if response.status_code != 200:
                print(f"发送失败：时间点 {data_point['time']}，状态码 {response.status_code}")
            else:
                print(f"成功发送：时间点 {data_point['time']}")

        except requests.exceptions.RequestException as e:
            print(f"请求异常：{str(e)}")

        # 严格保持0.05秒间隔
        time.sleep(0.05)


def main():
    # 连接到 Carla 服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 查找名为 'hero' 的车辆
    hero_vehicle = find_hero_vehicle(world)
    if not hero_vehicle:
        print("No hero vehicle found.")
        return

    print(f"Found hero vehicle: {hero_vehicle.id}")



    try:
        # 让模拟世界运行一段时间以捕获图像
        while True:
            world.tick()
            time.sleep(0.1)

            # UPDATE0113：获取 hero 车辆的位置、速度等数据
            hero_location = hero_vehicle.get_location()
            hero_velocity = hero_vehicle.get_velocity()
            acceleration = hero_vehicle.get_acceleration()

            # print(hero_location.x, hero_location.y, hero_location.z)

            v_location = [hero_location.x,hero_location.y]

            # print(v_location)

            v_speed = [hero_velocity.x, hero_velocity.y]
            velocity_modulus = np.linalg.norm(v_speed)

            v_acc = np.array([acceleration.x, acceleration.y, acceleration.z])
            acceleration_modulus = np.linalg.norm(v_acc)

            # 获取所有其他车辆的位置、速度等数据
            all_vehicles = world.get_actors().filter('vehicle.*')
            min_ttc = float('inf')
            closest_vehicle = None

            for vehicle in all_vehicles:
                if vehicle.id != hero_vehicle.id and vehicle.is_alive:
                    location = vehicle.get_location()
                    velocity = vehicle.get_velocity()
                    other_location = [location.x, location.y]
                    other_velocity = [velocity.x, velocity.y]

                    ttc = calculate_ttc(
                        # hero_location,
                        v_location,
                        # hero_velocity,
                        v_speed,
                        other_location,
                        other_velocity
                    )

                    if ttc is not None and ttc < min_ttc:
                        min_ttc = ttc
                        closest_vehicle = vehicle

            # 处理 min_ttc
            if min_ttc == float('inf'):
                min_ttc = None

            # 构造数据字典
            nearest_vehicle_id = closest_vehicle.id if closest_vehicle else None
            nearest_vehicle_name = closest_vehicle.type_id.split('.')[1] if closest_vehicle else None

            data = {
                "timestamp": time.time(),
                # "location": {"x": hero_location.x, "y": hero_location.y, "z": hero_location.z},
                # "velocity": {"x": hero_velocity.x, "y": hero_velocity.y, "z": hero_velocity.z},
                "velocity_modulus": velocity_modulus,
                # "acceleration": {"x": acceleration.x, "y": acceleration.y, "z": acceleration.z},
                "acceleration_modulus": acceleration_modulus,
                "ttc": min_ttc,  # 最小 TTC
                # "nearest_vehicle_id": nearest_vehicle_id,  # 最近车辆的ID
                # "nearest_vehicle_name": nearest_vehicle_name  # 最近车辆的名称
            }

            # 打印或导出这些数据
            print(f"Hero Vehicle Data:")
            print(json.dumps(data, indent=4))

            # 发送数据到 FastAPI 服务器
            response = requests.post("http://localhost:8001/upload-vehicle-data/", json=data)
            print(f"Response from server: {response.json()}")

    finally:
        # 清理资源
        print("Cleaned up resources")

if __name__ == '__main__':
    main()