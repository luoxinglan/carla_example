import subprocess
import time

def run_script(script_name):
    print(f"Starting {script_name}")
    process = subprocess.Popen(['python', script_name])
    return process

if __name__ == "__main__":
    scripts = ['generate_traffic_my.py', 'automatic_control_my.py', 'car_tracker.py', 'cams_and_lidar.py']
    processes = []

    for script in scripts:
        process = run_script(script)
        processes.append(process)
        time.sleep(10)  # 等待1秒后再启动下一个脚本

    for process in processes:
        process.wait()  # 等待所有进程完成

    print("All scripts have finished running.")