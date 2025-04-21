import mediapipe as mp
import cv2 as cv
import numpy as np
from collections import deque
import time
import tkinter as tk
from tkinter import ttk
import threading
import csv 
from AppKit import NSScreen


# 設定

consecutive_frames = 2  # 瞬きとみなすための連続フレーム数
calibration_duration = 5  # キャリブレーション時間（秒）

# CLOSED_EYE_CONSEC_FRAMES = 15 # 目が閉じ続けているとみなすフレーム数

gensyou = 3 #ゲージの減少速度
mabataki = 100 #瞬きの速度
syokiti = 150 #ゲージの初期値


# Mediapipeの設定
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 左右の目のランドマーク
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

def calculate_eye_aspect_ratio(eye_landmarks, face_landmarks):
    # 垂直方向の距離を計算
    v1 = np.linalg.norm(
        np.array([face_landmarks[eye_landmarks[1]].x, face_landmarks[eye_landmarks[1]].y]) -
        np.array([face_landmarks[eye_landmarks[5]].x, face_landmarks[eye_landmarks[5]].y])
    )
    v2 = np.linalg.norm(
        np.array([face_landmarks[eye_landmarks[2]].x, face_landmarks[eye_landmarks[2]].y]) -
        np.array([face_landmarks[eye_landmarks[4]].x, face_landmarks[eye_landmarks[4]].y])
    )
    # 水平方向の距離を計算
    h = np.linalg.norm(
        np.array([face_landmarks[eye_landmarks[0]].x, face_landmarks[eye_landmarks[0]].y]) -
        np.array([face_landmarks[eye_landmarks[3]].x, face_landmarks[eye_landmarks[3]].y])
    )
    # アスペクト比を計算（hが0の場合を避ける）
    ear = (v1 + v2) / (2.0 * h) if h != 0 else 0
    return ear

class BlinkCounterApp:
    def __init__(self, root):
        self.root = root

        # macOSの有効な画面領域を取得
        screen_frame = NSScreen.mainScreen().visibleFrame()
        screen_width = int(screen_frame.size.width)
        screen_height = int(screen_frame.size.height)
        screen_x = int(screen_frame.origin.x)
        screen_y = int(screen_frame.origin.y)

        # ウィンドウの高さを設定（必要に応じて調整）
        window_height = 80  # ゲージの高さに合わせて設定

        # ウィンドウのY座標を計算（macOSの座標系に合わせる）
        window_y = screen_y + screen_height - window_height 

        # ウィンドウのサイズと位置を設定（画面下部に配置）
        self.root.geometry(f"{screen_width}x{window_height}+{screen_x}+{int(window_y)+20}")

        self.root.resizable(False, False)
        self.root.overrideredirect(None)  # ウィンドウ枠を消す
        self.root.after(0, self.root.lower)

        # ウィンドウの透明度を設定（必要に応じて調整）
        self.root.attributes("-alpha", 0.95)
        self.root.configure(bg='black')  # 背景色を設定

        # ゲージ（プログレスバー）の追加。ゲージは減ってはいけない
        self.gauge_value = tk.DoubleVar(value=syokiti)


        # スタイルの定義
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # 緑色のスタイル
        self.style.configure("green.Horizontal.TProgressbar",
                             troughcolor='black',
                             background='green',
                             thickness=window_height)

        # 黄色のスタイル
        self.style.configure("yellow.Horizontal.TProgressbar",
                             troughcolor='black',
                             background='yellow',
                             thickness=window_height)

        # 赤色のスタイル
        self.style.configure("red.Horizontal.TProgressbar",
                             troughcolor='black',
                             background='red',
                             thickness=window_height)

        # ゲージの初期スタイルを緑に設定
        self.gauge = ttk.Progressbar(self.root, maximum=100, variable=self.gauge_value, style="green.Horizontal.TProgressbar")
        self.gauge.place(x=0, y=0, width=screen_width, height=window_height)

        # 共有データとロックの初期化
        self.blink_count = 0
        self.eye_closed = False  # 目が閉じているかどうかのフラグ
        self.lock = threading.Lock()
        self.blink_data = []

        # データを保存するCSVファイルを開く
        self.csv_file = open('blink_data.csv', 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp', 'Blink Count'])

        # スレッドの開始
        self.running = True
        self.thread = threading.Thread(target=self.blink_detection)
        self.thread.start()

        # ゲージの更新
        self.update_gauge()

    def blink_detection(self):
        cap = cv.VideoCapture(0)  # デフォルトカメラを使用
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            self.running = False
            return

        # 瞬き検出の変数
        blink_count = 0
        frame_counter = 0
        eye_closed_counter = 0

        EAR_AVERAGE_POINTS = 3
        left_ear_deque = deque(maxlen=EAR_AVERAGE_POINTS)
        right_ear_deque = deque(maxlen=EAR_AVERAGE_POINTS)

        # キャリブレーションの設定
        calibration_start_time = None
        calibration_ears = []

        calibrated = False
        ear_threshold = 0.0  # 初期閾値

        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while self.running:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # # 画像の前処理
                # image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
                # image = cv.resize(image, (640, 480)) 
                # image.flags.writeable = False
                # results = face_mesh.process(image)

                # 画像の前処理
                image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)

                image.flags.writeable = True
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # 左右のEARを計算
                        left_ear = calculate_eye_aspect_ratio(LEFT_EYE_LANDMARKS, face_landmarks.landmark)
                        right_ear = calculate_eye_aspect_ratio(RIGHT_EYE_LANDMARKS, face_landmarks.landmark)

                        # EARの平滑化
                        left_ear_deque.append(left_ear)
                        right_ear_deque.append(right_ear)
                        left_ear_avg = sum(left_ear_deque) / len(left_ear_deque)
                        right_ear_avg = sum(right_ear_deque) / len(right_ear_deque)
                        ear = (left_ear_avg + right_ear_avg) / 2.0

                        # キャリブレーション処理
                        if not calibrated:
                            if calibration_start_time is None:
                                calibration_start_time = time.time()

                            calibration_ears.append(ear)
                            elapsed_time = time.time() - calibration_start_time
                            if elapsed_time >= calibration_duration:
                                calibrated = True
                                baseline_ear = np.mean(calibration_ears)
                                ear_threshold = baseline_ear * 0.85  # 基準EARの85%を閾値に設定
                                print(f"キャリブレーション完了。基準EAR: {baseline_ear:.3f}, 閾値: {ear_threshold:.3f}")
                        else:
                            # 瞬きと長時間の目の閉じを判定
                            if ear < ear_threshold:
                                frame_counter += 1
                                eye_closed_counter += 1

                                # 目を閉じ続けているか確認
                                # if eye_closed_counter >= CLOSED_EYE_CONSEC_FRAMES:
                                #     with self.lock:
                                #         self.eye_closed = True
                            else:
                                if frame_counter >= consecutive_frames:
                                    blink_count += 1
                                    with self.lock:
                                        self.blink_count = blink_count
                                    print(f"瞬き検出！ 総瞬き数: {blink_count}")

                                    # 瞬きでゲージを増加
                                    with self.lock:
                                        new_gauge_value = self.gauge_value.get() + mabataki  # 増加量を調整
                                        if new_gauge_value > 100:
                                            new_gauge_value = 100
                                        self.gauge_value.set(new_gauge_value)

                                    # 瞬きのタイムスタンプを記録
                                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                                    with self.lock:
                                        self.blink_data.append((timestamp, blink_count))
                                        self.csv_writer.writerow([timestamp, blink_count])
                                        self.csv_file.flush()
                                frame_counter = 0
                                # eye_closed_counter = 0
                                with self.lock:
                                    self.eye_closed = False
                else:
                    pass  # 顔が検出されない場合の処理

                # ループの実行速度を制御
                time.sleep(0.03)

        # リソースの解放
        cap.release()

    def update_gauge(self):
        with self.lock:
            # 目を開けている場合、ゲージを減少
            new_value = self.gauge_value.get() - gensyou  # 減少速度を調整
            if new_value < 0:
                new_value = 0
            self.gauge_value.set(new_value)

            # ゲージの値に応じてスタイルを変更
            current_value = self.gauge_value.get()
            if current_value > 50:
                self.gauge.config(style="green.Horizontal.TProgressbar")
            elif current_value > 25:
                self.gauge.config(style="yellow.Horizontal.TProgressbar")
            else:
                self.gauge.config(style="red.Horizontal.TProgressbar")

        if self.running:
            self.root.after(100, self.update_gauge)  # 100msごとに更新

    def on_close(self):
        self.running = False
        self.thread.join()
        self.csv_file.close()  # ファイルを閉じる
        self.root.destroy()

def main():
    root = tk.Tk()
    app = BlinkCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()