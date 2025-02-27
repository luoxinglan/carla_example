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

            # 获取跟踪车辆的位置、速度、加速度等数据
            location = hero_vehicle.get_location()
            velocity = hero_vehicle.get_velocity()
            acceleration = hero_vehicle.get_acceleration()

            # 构造数据字典
            data = {
                "timestamp": time.time(),
                "location": {"x": location.x, "y": location.y, "z": location.z},
                "velocity": {"x": velocity.x, "y": velocity.y, "z": velocity.z},
                "acceleration": {"x": acceleration.x, "y": acceleration.y, "z": acceleration.z}
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