import argparse
import re
import json
import math


def parse_frame_blocks(content):
    """
    分割内容

    :param content:文件内容
    """
    frame_blocks = []
    current_block = []
    for line in content.split('\n'):
        if line.startswith('Frame'):
            # 如果当前块不为空，则将其添加到帧块列表中
            if current_block:
                frame_blocks.append(current_block)
            # 开始一个新的帧块
            current_block = [line]
        else:
            # 将当前行添加到当前帧块中
            current_block.append(line)
    # 添加最后一个帧块（如果存在）
    if current_block:
        frame_blocks.append(current_block)
    return frame_blocks


def parse_block(block):
    """
    分割每一个块的内容

    :param block:一个块
    """
    frame_data = {'time': 0.0, 'positions': {}, 'velocities': {}, 'controls': {}}
    # 解析时间戳
    time_match = re.search(r'Frame \d+ at ([\d.]+) seconds', block[0])
    if time_match:
        frame_data['time'] = float(time_match.group(1))
        # print(f"解析时间: {frame_data['time']}")

    # 解析位置信息
    positions_section = False
    for line in block:
        if line.strip().startswith('Positions:'):
            positions_section = True
            continue
        if positions_section and line.strip() == '':
            positions_section = False
            continue
        if positions_section:
            match = re.match(r'\s*Id: (\d+) Location: \(([-\d.]+), ([-\d.]+), [-\d.]+\)', line)
            if match:
                entity_id = match.group(1)
                x = float(match.group(2))
                y = float(match.group(3))
                frame_data['positions'][entity_id] = (x, y)
                # if entity_id == '91':
                # print(f"解析位置信息 - 实体ID {entity_id}: ({x}, {y})")

    # 解析速度信息
    dynamic_section = False

    for line in block:
        if line.strip().startswith('Dynamic actors: '):
            dynamic_section = True
            continue
        if dynamic_section and line.strip() == '':
            dynamic_section = False
            continue
        if dynamic_section:
            match = re.match(r'\s*Id: (\d+) linear_velocity: \(([-\d.eE+-]+), ([-\d.eE+-]+), [-\d.eE+-]+\)', line)
            if match:
                entity_id = match.group(1)
                vx = float(match.group(2))
                vy = float(match.group(3))
                # vz = float(match.group(4))
                # velocities[entity_id] = (vx, vy, vz)
                frame_data['velocities'][entity_id] = (vx, vy)

                # 调试特定车辆
                # if entity_id == '91':
                # print(f"解析速度信息 - 实体ID {entity_id} | 线速度分量: ({vx}, {vy})")

    # 解析控制信息
    frame_data['controls'] = parse_vehicle_control_section(block)

    # print(f"这是{frame_data['time']}帧的结果：")
    # print(json.dumps(frame_data, indent=4))
    return frame_data


def parse_vehicle_control_section(block, target_id='141'):
    """
    解析block的控制信息。
    :param block:一个块
    """
    control_section = False
    vehicle_controls = {}

    for line in block:
        if line.strip().startswith('Vehicle animations:'):
            control_section = True
            continue
        if control_section and line.strip() == '':
            control_section = False
            continue
        if control_section:
            match = re.match(
                r'\s*Id: (\d+) Steering: ([-\d.eE+-]+) Throttle: ([-\d.eE+-]+) Brake ([-\d.eE+-]+) Handbrake: ([-\d.eE+-]+) Gear: (\d+)',
                line)
            if match:
                entity_id = match.group(1)
                steering = float(match.group(2))
                throttle = float(match.group(3))
                brake = float(match.group(4))
                handbrake = int(match.group(5))
                gear = int(match.group(6))
                vehicle_controls[entity_id] = {
                    'steering': steering,
                    'throttle': throttle,
                    'brake': brake,
                    'handbrake': handbrake,
                    'gear': gear
                }

                # 调试特定车辆
                # if entity_id == target_id:
                #     print(f"解析控制信息 - 实体ID: {entity_id} | {{Steering: {steering}, Throttle: {throttle}, Brake: {brake}, Handbrake: {handbrake}, Gear: {gear}}}")
    return vehicle_controls


def calculate_ttc(target_pos, target_vel, other_pos, other_vel):
    """
    计算目标车辆的TTC

    :param target_pos:目标车辆的位置
    :param target_vel:目标车辆的速度
    :param other_pos:参照车辆的位置
    :param other_vel:参照车辆的位置
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


def process_frames(frames_data, target_id):
    """
    处理每一帧的数据。

    :param frames_data:所有数据
    :param target_id:目标车辆的id
    """
    results = []
    prev_speed = None
    prev_time = None

    for frame in frames_data:
        current_time = frame['time']
        target_pos = frame['positions'].get(target_id)
        target_vel = frame['velocities'].get(target_id, (0, 0))

        if not target_pos:
            print(f"未找到目标ID {target_id} 在时间 {current_time} 的位置信息")
            continue

        # 计算速度
        speed = math.hypot(target_vel[0], target_vel[1])

        # 计算加速度
        acceleration = None
        if prev_speed is not None and prev_time is not None and current_time > prev_time:
            dt = current_time - prev_time
            if dt > 0:
                acceleration = (speed - prev_speed) / dt

        # 计算最小TTC
        min_ttc = float('inf')
        for entity_id in frame['positions']:
            if entity_id == target_id:
                continue

            entity_pos = frame['positions'][entity_id]
            entity_vel = frame['velocities'].get(entity_id, (0, 0))

            ttc = calculate_ttc(target_pos, target_vel, entity_pos, entity_vel)
            if ttc < min_ttc:
                min_ttc = ttc

        # 处理结果TODO id
        result = {
            'time': round(current_time, 2),
            'speed': round(speed, 3),
            'acceleration': round(acceleration, 3) if acceleration is not None else None,
            'min_ttc': round(min_ttc, 3) if min_ttc != float('inf') else None,
            'steering': round(frame['controls'][target_id]['steering'], 3),
            'throttle': round(frame['controls'][target_id]['throttle'], 3),
            'brake': round(frame['controls'][target_id]['brake'], 3),
            'handbrake': round(frame['controls'][target_id]['handbrake'], 3),
            'gear': frame['controls'][target_id]['gear']
        }
        results.append(result)
        # print(f"处理帧数据 - 时间 {current_time}: 速度={speed}, 加速度={acceleration}, 最小TTC={min_ttc}")
        # print(json.dumps(results, indent=9))

        # 更新前值
        prev_speed = speed
        prev_time = current_time

    return results


def process_record(input_file, json_filename, target_id):
    """
    将未处理的可阅读日志文件转换为格式化的json文件
    :param input_file:未处理的可阅读日志文件
    :param json_filename:输出的格式化的json文件
    :param target_id:hero的id
    """
    # 读取文件
    with open(input_file, 'r') as f:
        content = f.read()

    # 分解成帧
    frame_blocks = parse_frame_blocks(content)
    print(f"解析出 {len(frame_blocks)} 帧块。")

    # 逐个分析帧
    frames_data = [parse_block(block) for block in frame_blocks]
    results = process_frames(frames_data, target_id)

    # 将解析后的数据写入json
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存到 {json_filename}")
    return len(results)


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        # TODO: this is required.
        '-i', '--input_filename',
        metavar='I',
        default="../logs/record0306.txt",
        help='recorder filename (record0306.txt)')
    argparser.add_argument(
        # TODO: this is required too.
        '-j', '--json_filename',
        metavar='J',
        default="../logs/vehicle_status0306.json",
        help='save result to file (specify name and extension such as \'../logs/vehicle_status0306.json\', and path before it if you need it)')
    argparser.add_argument(
        # TODO: this is required too.
        '-t',
        '--target_id',
        metavar='T',
        default='141',
        help='hero car id (or index) (default: 141)')
    args = argparser.parse_args()
    # 使用示例（需要替换实际文件路径和目标ID）
    process_record(args.input_filename, args.json_filename, args.target_id)


if __name__ == '__main__':
    main()
