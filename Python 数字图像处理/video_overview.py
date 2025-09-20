import cv2
import numpy as np
from datetime import timedelta
import sys
import os
import subprocess
import json


# 配置文件路径
CONFIG_FILE = os.path.expanduser("~/.video_overview_config.json")


def load_last_path():
    """
    加载上次使用的文件路径
    :return: 上次的目录路径
    """
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('last_directory', os.path.expanduser('~'))
    except Exception as e:
        print(f"加载配置文件失败: {e}")

    return os.path.expanduser('~')  # 默认返回用户主目录


def save_last_path(file_path):
    """
    保存当前选择的文件路径的目录
    :param file_path: 选择的文件路径
    """
    try:
        directory = os.path.dirname(file_path)
        config = {'last_directory': directory}

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存配置文件失败: {e}")


def select_video_file():
    """
    使用系统默认的文件选择对话框选择视频文件，记住上次路径
    :return: 选择的视频文件路径
    """
    try:
        # 获取上次使用的目录
        last_directory = load_last_path()

        # 使用zenity调用系统文件选择对话框
        result = subprocess.run([
            'zenity', '--file-selection',
            '--title=选择视频文件',
            '--filename=' + last_directory + '/',  # 设置起始目录
            '--file-filter=视频文件 | *.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.m4v *.3gp',
            '--file-filter=所有文件 | *'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            file_path = result.stdout.strip()
            # 保存这次选择的目录
            save_last_path(file_path)
            return file_path
        else:
            print("用户取消了文件选择")
            return ""

    except FileNotFoundError:
        print("系统未安装zenity，请安装: sudo apt install zenity")
        print("回退到命令行输入模式：")

        # 显示上次的目录作为提示
        last_directory = load_last_path()
        print(f"上次使用的目录: {last_directory}")
        print("请输入视频文件的完整路径:")

        file_path = input().strip()
        file_path = file_path.strip('"').strip("'")

        if os.path.exists(file_path):
            # 保存这次选择的目录
            save_last_path(file_path)
            return file_path
        else:
            print("文件不存在，请检查路径是否正确")
            return ""
    except Exception as e:
        print(f"打开文件对话框时出错: {e}")
        return ""


def show_image(image_path):
    """
    使用Ubuntu系统自带的图片查看器打开图片
    :param image_path: 图片文件路径
    """
    try:
        subprocess.run(["xdg-open", image_path])
        print(f"已调用系统图片查看器打开: {image_path}")
    except Exception as e:
        print(f"无法打开图片查看器: {e}")
        print(f"请手动查看图片: {os.path.abspath(image_path)}")


def check_video_support(video_path):
    """
    检查视频文件是否可以正常打开
    :param video_path: 视频文件路径
    :return: True if supported, False otherwise
    """
    cap = cv2.VideoCapture(video_path)
    is_opened = cap.isOpened()
    if is_opened:
        # 尝试读取第一帧确认格式支持
        ret, frame = cap.read()
        cap.release()
        return ret
    return False


def get_frame_at_time(video_path, target_time_sec, max_retries=5):
    """
    使用二分法思想精确定位到视频的特定时间点并提取帧，增加容错机制
    :param video_path: 视频文件路径
    :param target_time_sec: 目标时间（秒）
    :param max_retries: 最大重试次数
    :return: 该时间点的帧（numpy数组）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    # 获取视频总时长
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps if fps > 0 else 0

    # 确保目标时间在有效范围内
    target_time_sec = max(0, min(target_time_sec, total_duration - 1))

    # 尝试不同的时间偏移来获取有效帧
    offsets = [0, -1, 1, -2, 2, -5, 5, -10, 10]

    for offset in offsets:
        try:
            test_time = max(0, min(target_time_sec + offset, total_duration - 1))
            cap.set(cv2.CAP_PROP_POS_MSEC, test_time * 1000)

            # 尝试读取多次以跳过损坏的帧
            for _ in range(max_retries):
                ret, frame = cap.read()
                if ret and frame is not None:
                    # 检查帧是否有效（不是全黑或全白）
                    if frame.mean() > 5 and frame.mean() < 250:
                        cap.release()
                        return frame
        except Exception as e:
            continue

    cap.release()
    raise ValueError(f"无法在时间 {target_time_sec}s 附近获取有效帧")


def add_timestamp_to_frame(frame, timestamp_sec, scale_factor=0.5):
    """
    在帧上添加时间戳文本
    :param frame: 原始帧
    :param timestamp_sec: 时间戳（秒）
    :param scale_factor: 缩放因子，用于调整字体大小
    :return: 添加了时间戳的帧
    """
    if frame is None:
        return None

    time_str = str(timedelta(seconds=int(timestamp_sec)))

    # 根据缩放因子调整字体大小
    font_scale = 1.8 if scale_factor < 1 else 2.5
    thickness = max(5, int(8 * scale_factor))

    # 添加黑色描边效果（抗锯齿）
    cv2.putText(frame, time_str, (60, 80),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # 添加白色文字（抗锯齿）
    cv2.putText(frame, time_str, (60, 80),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame


def extract_frames_binary(video_path, num_frames):
    """
    使用二分法递归地选择时间点并提取帧，增加错误处理
    :param video_path: 视频文件路径
    :param num_frames: 要提取的帧数
    :return: 包含时间戳和帧的列表
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"视频总时长: {timedelta(seconds=int(total_duration_sec))}")
    print(f"视频帧率: {fps:.2f} FPS")
    print(f"总帧数: {int(total_frames)}")

    if total_duration_sec <= 0:
        raise ValueError("无法获取视频时长信息")

    # 生成二分时间点
    time_points = []

    def generate_times(start, end, count):
        if count <= 0:
            return
        mid = (start + end) / 2
        time_points.append(mid)
        generate_times(start, mid, count // 2)
        generate_times(mid, end, count // 2)

    generate_times(0, total_duration_sec, num_frames)
    time_points = sorted(set(time_points))[:num_frames]

    frames = []
    successful_extractions = 0

    for i, time in enumerate(time_points):
        try:
            print(f"正在提取第 {i+1}/{len(time_points)} 个帧 (时间: {timedelta(seconds=int(time))})...")
            frame = get_frame_at_time(video_path, time)
            frame_with_ts = add_timestamp_to_frame(frame, time)

            if frame_with_ts is not None:
                frames.append((time, frame_with_ts))
                successful_extractions += 1
                print(f"✓ 成功提取第 {i+1} 个帧")
            else:
                print(f"✗ 第 {i+1} 个帧提取失败")

        except Exception as e:
            print(f"✗ 第 {i+1} 个帧提取失败: {e}")
            continue

    print(f"\n成功提取 {successful_extractions}/{len(time_points)} 个帧")

    if successful_extractions == 0:
        raise ValueError("未能提取到任何有效帧")

    return frames


def create_frame_collage(frames, cols=3, scale_factor=1.2):
    """
    将多个帧排列到一张大图上
    :param frames: 帧列表（每个元素为(time, frame)）
    :param cols: 排列的列数
    :param scale_factor: 缩放因子
    :return: 组合后的大图
    """
    if not frames:
        raise ValueError("没有可用的帧来创建拼图")

    processed_frames = []

    for time, frame in frames:
        if frame is None:
            continue

        height, width = frame.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        processed_frames.append(resized)

    if not processed_frames:
        raise ValueError("没有有效的帧可以创建拼图")

    # 计算大图的行数和尺寸
    rows = (len(processed_frames) + cols - 1) // cols
    sample_frame = processed_frames[0]
    frame_h, frame_w = sample_frame.shape[:2]

    collage = np.zeros((frame_h * rows, frame_w * cols, 3), dtype=np.uint8)

    for i, frame in enumerate(processed_frames):
        row = i // cols
        col = i % cols
        y_start = row * frame_h
        y_end = y_start + frame_h
        x_start = col * frame_w
        x_end = x_start + frame_w
        collage[y_start:y_end, x_start:x_end] = frame

    return collage


# 主程序
if __name__ == "__main__":
    try:
        print("=== 视频帧提取工具 ===")

        # 选择视频文件（会记住上次路径）
        video_path = select_video_file()

        if not video_path:
            print("未选择视频文件，程序退出")
            sys.exit(1)

        print(f"选择的视频文件: {video_path}")

        # 检查视频格式支持
        if not check_video_support(video_path):
            print("错误: 不支持的视频格式或文件损坏")
            sys.exit(1)

        print("开始解析视频...")

        # 提取9个帧
        frames = extract_frames_binary(video_path, 9)

        if frames:
            print("正在生成拼图...")
            # 创建拼图
            collage = create_frame_collage(frames, cols=3, scale_factor=0.5)

            # 保存结果到当前目录
            output_path = "video_collage.jpg"
            cv2.imwrite(output_path, collage)
            print(f"✓ 拼图已保存为 {output_path}")

            # 显示结果
            show_image(output_path)

            # 打印时间戳信息
            print("\n成功提取的帧时间戳:")
            for i, (time, frame) in enumerate(frames):
                print(f"第{i+1}帧: {timedelta(seconds=int(time))}")

            print("\n处理完成！")
        else:
            print("未能成功提取任何帧")

    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        sys.exit(1)