import carla
import time
import numpy as np
import requests
import json

def find_hero_vehicle(world):
    """
    查找名为 'hero' 的车辆。
    """
    vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in vehicles:
        if vehicle.is_alive and vehicle.attributes.get('role_name') == 'hero':
            return vehicle
    return None

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

    # 附加一个摄像机到车上（可选）
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=hero_vehicle)
    print(f"Attached camera to vehicle at {camera_transform.location}")

    def _parse_image(image):
        # 将图像转换为数组以便处理
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        # 在这里可以添加图像处理逻辑来检测和跟踪车辆
        # 为了简化示例，我们假设摄像机始终跟踪其附着的车辆

    # 设置摄像机监听回调函数（可选）
    camera.listen(_parse_image)

    try:
        # 让模拟世界运行一段时间以捕获图像
        while True:
            world.tick()
            time.sleep(0.1)

            # UPDATE0113：获取 hero 车辆的位置、速度等数据
            hero_location = hero_vehicle.get_location()
            hero_velocity = hero_vehicle.get_velocity()
            acceleration = hero_vehicle.get_acceleration()

            v_speed = np.array([hero_velocity.x, hero_velocity.y, hero_velocity.z])
            velocity_modulus = np.linalg.norm(v_speed)

            v_acc = np.array([acceleration.x, acceleration.y, acceleration.z])
            acceleration_modulus = np.linalg.norm(v_acc)

            # 获取所有其他车辆的位置、速度等数据
            all_vehicles = world.get_actors().filter('vehicle.*')
            min_ttc = float('inf')
            closest_vehicle = None

            for vehicle in all_vehicles:
                if vehicle.id != hero_vehicle.id and vehicle.is_alive:
                    other_location = vehicle.get_location()
                    other_velocity = vehicle.get_velocity()

                    ttc = calculate_ttc(
                        hero_location,
                        hero_velocity,
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
        camera.destroy()
        print("Cleaned up resources")

if __name__ == '__main__':
    main()