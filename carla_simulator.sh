#!/bin/bash

# 设置CARLA的路径（根据实际情况修改）
CARLA_ROOT="/home/heihuhu/Projects/CARLA_0.9.13_safebench"
CARLA_SERVER="$CARLA_ROOT/CarlaUE4.sh"

# 设置Python脚本的路径
SIMULATION_SCRIPT="./mycarla/sensors/cams_and_lidar.py"

# 设置CARLA服务器端口
PORT=2001

# 检查端口是否已被占用
if sudo lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Please check and free it."
    exit 1
fi

# 启动CARLA服务器
echo "Starting CARLA server on port $PORT..."
$CARLA_SERVER -world-port=$PORT &
CARLA_PID=$!

# 等待几秒钟确保服务器已经启动
sleep 5

# 运行模拟脚本
echo "Running simulation script with mode 'replay' and duration 60 seconds..."
python $SIMULATION_SCRIPT --server-ip="127.0.0.1" --port=$PORT --mode="replay" --duration=60 &

# 获取模拟脚本的PID
SIMULATION_PID=$!
x
# 定义清理函数
cleanup() {
    echo "Stopping CARLA server..."
    kill -9 $CARLA_PID
    echo "Stopping simulation script..."
    kill -9 $SIMULATION_PID
    echo "Test process terminated by user."
    exit 1
}

# 注册信号处理器
trap cleanup SIGINT

# 等待模拟脚本完成
wait $SIMULATION_PID

echo "Test process completed."