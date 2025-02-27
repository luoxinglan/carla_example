import threading
import time


class CarTracker:
    def __init__(self, sensor_freq=10):
        self.sensor_freq = sensor_freq  # 数据采集频率（Hz）
        self.latest_data = {"speed": 0, "position": None, "timestamp": None}
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        """启动数据采集线程"""
        self.running = True
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.start()

    def stop(self):
        """停止数据采集线程"""
        self.running = False
        if self.thread:
            self.thread.join()

    def _update_loop(self):
        """数据采集循环"""
        while self.running:
            # 实际传感器数据采集逻辑（示例）
            new_data = {
                "speed": get_speed_from_hardware(),  # 替换为实际获取速度的方法
                "position": get_position_from_gps(),  # 替换为实际获取位置的方法
                "timestamp": time.time()
            }

            with self.lock:
                self.latest_data = new_data

            time.sleep(1 / self.sensor_freq)  # 按指定频率采集

    def get_latest_data(self):
        """线程安全获取最新数据"""
        with self.lock:
            return self.latest_data.copy()