from PyQt6.QtWidgets import *
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QImage
import cv2 as cv
import numpy as np
import sys
import winsound
from PIL import Image  # Import at the beginning of your code

class CameraThread(QThread):
    frameCaptured = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._running = True

    def run(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            print("카메라 연결 실패")
            return
        
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                self.frameCaptured.emit(frame)

        self.cap.release()

    def stop(self):
        self._running = False

class Panorama(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("파노라마 영상")
        self.setGeometry(200, 200, 700, 500)

        collectButton = QPushButton("영상 수집", self)
        self.showButton = QPushButton("영상 보기", self)
        self.stitchButton = QPushButton("봉합", self)
        self.saveButton = QPushButton("저장", self)
        quitButton = QPushButton("나가기", self)
        self.label = QLabel("환영합니다", self)

        # Combo box for selecting stitching mode
        self.modeComboBox = QComboBox(self)
        self.modeComboBox.addItem("Panorama Mode")
        self.modeComboBox.addItem("Mosaic Mode")

        self.displayLabel = QLabel(self)
        self.displayLabel.setGeometry(10, 130, 640, 450)
        self.displayLabel.setStyleSheet("background-color: black")

        collectButton.setGeometry(10, 25, 100, 30)
        self.showButton.setGeometry(110, 25, 100, 30)
        self.stitchButton.setGeometry(210, 25, 100, 30)
        self.saveButton.setGeometry(310, 25, 100, 30)
        quitButton.setGeometry(450, 25, 100, 30)
        self.modeComboBox.setGeometry(570, 25, 100, 30)
        self.label.setGeometry(10, 60, 600, 30)

        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)

        collectButton.clicked.connect(self.start_collecting)
        self.showButton.clicked.connect(self.showFunction)
        self.stitchButton.clicked.connect(self.stitchFunction)
        self.saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.camera_thread = CameraThread()
        self.camera_thread.frameCaptured.connect(self.update_frame)
        self.imgs = []

    def start_collecting(self):
        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        self.label.setText("c를 여러 번 눌러 수집하고 끝나면 q를 눌러 비디오를 끈다")
        self.imgs = []
        self.camera_thread.start()

    def update_frame(self, frame):
        self.latest_frame = frame
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.displayLabel.setPixmap(pixmap)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_C:
            if hasattr(self, 'latest_frame'):
                self.imgs.append(cv.resize(self.latest_frame.copy(), (640, 480)))  # 이미지 크기를 줄여 메모리 사용량 절약
                self.label.setText(f"수집된 영상: {len(self.imgs)}장")
        elif event.key() == Qt.Key.Key_Q:
            self.stop_collecting()

    def stop_collecting(self):
        self.camera_thread.stop()
        self.camera_thread.wait()
        
        if len(self.imgs) >= 2:
            self.showButton.setEnabled(True)
            self.stitchButton.setEnabled(True)
            self.saveButton.setEnabled(True)

    def showFunction(self):
        self.label.setText(f"수집된 영상은 {len(self.imgs)}장입니다.")
        stack = cv.resize(self.imgs[0], dsize=(0, 0), fx=0.25, fy=0.25)
        for i in range(1, len(self.imgs)):
            stack = np.hstack((stack, cv.resize(self.imgs[i], dsize=(0, 0), fx=0.25, fy=0.25)))
        rgb_stack = cv.cvtColor(stack, cv.COLOR_BGR2RGB)
        qt_image = QImage(rgb_stack.data, rgb_stack.shape[1], rgb_stack.shape[0], QImage.Format.Format_RGB888)
        self.displayLabel.setPixmap(QPixmap.fromImage(qt_image))

    def stitchFunction(self):
        selected_mode = self.modeComboBox.currentText()

        if selected_mode == "Panorama Mode":
            # Panorama stitching
            try:
                stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
                status, self.img_stitched = stitcher.stitch(self.imgs)
                
                if status == cv.Stitcher_OK:
                    rgb_stitched = cv.cvtColor(self.img_stitched, cv.COLOR_BGR2RGB)
                    qt_image = QImage(rgb_stitched.data, rgb_stitched.shape[1], rgb_stitched.shape[0], QImage.Format.Format_RGB888)
                    self.displayLabel.setPixmap(QPixmap.fromImage(qt_image))
                    self.label.setText("파노라마 제작 성공!")
                else:
                    self.label.setText("파노라마 제작 실패: 이미지가 불일치할 수 있음.")
                    print(f"Stitching failed with status code: {status}")
            except Exception as e:
                self.label.setText(f"오류 발생: {str(e)}")
                print(f"Error during stitching: {e}")

        elif selected_mode == "Mosaic Mode":
            # Mosaic stitching
            rows = int(np.ceil(np.sqrt(len(self.imgs))))
            cols = rows
            img_h, img_w, _ = self.imgs[0].shape
            mosaic_img = np.zeros((img_h * rows, img_w * cols, 3), dtype=np.uint8)

            for idx, img in enumerate(self.imgs):
                row = idx // cols
                col = idx % cols
                mosaic_img[row*img_h:(row+1)*img_h, col*img_w:(col+1)*img_w] = img

            self.img_stitched = mosaic_img
            rgb_mosaic = cv.cvtColor(mosaic_img, cv.COLOR_BGR2RGB)
            qt_image = QImage(rgb_mosaic.data, rgb_mosaic.shape[1], rgb_mosaic.shape[0], QImage.Format.Format_RGB888)
            self.displayLabel.setPixmap(QPixmap.fromImage(qt_image))
            self.label.setText("모자이크 제작 성공!")

    def saveFunction(self):
        fname, _ = QFileDialog.getSaveFileName(self, "파일 저장", "./", "Image Files (*.png *.jpg *.jpeg)")
        
        if fname:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fname += '.png'  # Default to .png if no extension is provided

            try:
                rgb_image = cv.cvtColor(self.img_stitched, cv.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                pil_image.save(fname)
                self.label.setText("파일이 저장되었습니다.")
            except Exception as e:
                self.label.setText("파일 저장 실패.")
                print(f"파일 저장 중 오류 발생: {e}")

    def quitFunction(self):
        self.stop_collecting()
        self.close()

app = QApplication(sys.argv)
win = Panorama()
win.show()
sys.exit(app.exec())
