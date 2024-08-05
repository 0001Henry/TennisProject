import cv2

def resize_video(input_path, output_path, width, height):
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频编码格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 使用mp4格式
    # 打开视频写入对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 调整帧的大小
        resized_frame = cv2.resize(frame, (width, height))
        # 写入调整后的帧到输出视频文件
        out.write(resized_frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 调用函数进行视频分辨率调整
input_video_path = 'input_video.mp4'
output_video_path = 'output_video_1280x720.mp4'
resize_video(input_video_path, output_video_path, 1280, 720)
