import tempfile
import cv2
import json
import pyaudio
from vosk import Model, KaldiRecognizer
import pyttsx3
import dashscope


# 加载 Vosk 模型
model = Model(r"E:\AI_source_code\ASR-LLM-TTS-master\models\vosk-model-cn-0.22")  # 替换为你的模型路径
recognizer = KaldiRecognizer(model, 16000)

# 初始化文本转语音引擎
engine = pyttsx3.init()

def speak(text):
    """直接将文本转换为语音并播放"""
    engine.say(text)
    engine.runAndWait()


def send_to_model(video_frame, recognized_text):
    """将视频帧和识别到的文本发送给多模态大模型"""
    # video_frame 是要编码的图像帧
    _, buffer = cv2.imencode('.jpg', video_frame)
    video_frame_bytes = buffer.tobytes()
    # # 创建一个临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        # 将字节流写入临时文件
        temp_file.write(video_frame_bytes)
        temp_file_path = temp_file.name  # 获取临时文件的路径

    # 现在可以使用 temp_file_path 访问临时文件
    print(f"临时文件已保存到: {temp_file_path}")
    messages = [
        {
            "role": "user",
            "content": [
                {"image": temp_file_path},
                {"text": recognized_text}
            ]
        }
    ]
    responses = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="your_api_key",
        model='qwen-vl-max',
        messages=messages

    )
    # 提取并打印 content 内容
    response = responses['output']['choices'][0]['message']['content']

    return response

def main():
    # 启动视频捕捉
    cap = cv2.VideoCapture(0)

    # 初始化音频流
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 读取音频流并进行识别
        data = stream.read(4000, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            result_json = json.loads(result)
            if 'text' in result_json and result_json['text']:
                command = result_json['text']
                print(f"你说: {command}")

                # 语音反馈
                response_text = f"你刚才说了: {command}"
                speak(response_text)

                # 发送数据到多模态大模型
                model_response = send_to_model(frame, command)
                print("模型响应:", model_response)
                speak(model_response)

        # 显示视频流
        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
            break

    # 清理资源
    stream.stop_stream()
    stream.close()
    p.terminate()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
