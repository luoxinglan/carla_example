import csv
import time
from collections import deque
from threading import Lock
import carla


class RealtimeLocationLogger:
    def __init__(self, filename, buffer_size=50,is_route=False):
        self.filename = filename
        self.buffer = deque(maxlen=buffer_size)  # 内存缓冲区
        self.lock = Lock()
        if not is_route:
            self._init_file()

    def _init_file(self):
        """初始化文件并写入表头"""
        with open(self.filename, 'w', buffering=1) as f:  # 行缓冲模式
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'x', 'y', 'z'])

    def _write_buffer(self):
        """原子化写入操作"""
        with self.lock:
            if not self.buffer:
                return

            with open(self.filename, 'a', buffering=1) as f:
                writer = csv.writer(f)
                while self.buffer:
                    writer.writerow(self.buffer.popleft())

    def record(self, vehicle):
        """记录单帧数据"""
        try:
            transform = vehicle.get_transform()
            data = (
                time.time(),
                transform.location.x,
                transform.location.y,
                transform.location.z
            )
            self.buffer.append(data)

            print("vehicle.get_transform().location: ", transform.location)

            # 缓冲触发写入条件
            if len(self.buffer) >= self.buffer.maxlen:
                self._write_buffer()

        except (AttributeError, RuntimeError) as e:
            print(f"记录失败：{str(e)}")
            self._write_buffer()  # 紧急保存缓冲数据

    def flush(self):
        """强制写入剩余数据"""
        self._write_buffer()

    def load_waypoints_from_csv(self, world, accuracy=0.5):
        """
        读取CSV文件并转换为CARLA路径点
        参数：
            world : carla.World对象
            accuracy : 路径点匹配精度（单位：米）
        返回：
            list[carla.Waypoint] 路径点列表
        """
        waypoints = []
        map = world.get_map()
        print("map: ", map)

        with open(self.filename, 'r') as f:
            # print("文件内容示例：", f.readlines()[:3])  # 打印前三行
            f.seek(0)  # 重置文件指针
            reader = csv.DictReader(f)
            # print("reader.fieldnames: ", reader.fieldnames)
            for row in reader:
                # print("row: ", row)
                try:
                    # 解析坐标数据
                    location = carla.Location(
                        x=float(row['x']),
                        y=float(row['y']),
                        z=float(row['z'])
                    )

                    # 获取最近的路径点
                    waypoint = map.get_waypoint(
                        location,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving
                    )

                    if waypoint:
                        waypoints.append(waypoint)
                        # print(f"成功匹配路径点：{location}")
                    else:
                        print(f"警告：坐标{location}未找到有效路径点")

                except (KeyError, ValueError) as e:
                    print(f"数据解析错误：{str(e)}，跳过该行")
                except Exception as e:
                    print(f"未知错误：{str(e)}")

        # 路径点优化（去除重复点）
        return waypoints
        # return self._optimize_waypoints(waypoints, accuracy)

    def _optimize_waypoints(self, waypoints, accuracy):
        """
        优化路径点序列
        参数：
            waypoints : 原始路径点列表
            accuracy : 去重精度阈值
        返回：
            list[carla.Waypoint] 优化后的路径点
        """
        optimized = []
        prev_location = None

        for wp in waypoints:
            current = wp.transform.location
            if prev_location is None or \
                    prev_location.distance(current) > accuracy:
                optimized.append(wp)
                prev_location = current
        return optimized
