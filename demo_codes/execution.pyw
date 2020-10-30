'''
    My Application for demo
'''

import sys, pydicom, cv2, os, csv
import numpy as np
import pydicom
import fcn, display_metadata, brightness_contrast_control, black_white_windowing, windowing, skull, canny_edge_detection, histogram_equalization, skull_super_pixel_clustering
import copy
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte
from keras.models import model_from_json
from PIL import ImageEnhance, Image

IMG_SIZE = 512


class MyApp(QWidget):

    def __init__(self):
        super().__init__()

        # load model when program starts
        json_file = open("200529_model.json", "r")

        loaded_model_json = json_file.read()
        json_file.close()

        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights("200529_model.h5")

        # load UI when program starts
        self.initUI()

    def initUI(self):
        # setup window setting
        self.setWindowTitle('INFINITT healthcare')
        self.move(0, 0)
        self.resize(1850, 800)

        # setup file I/O
        self.pushButton = QPushButton("File Open")
        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.comboBox_recent_files = QComboBox()
        self.comboBox_recent_files.addItem("---- recent files ----")
        self.comboBox_recent_files.setCurrentIndex(0)
        self.comboBox_recent_files.activated.connect(self.comboBoxRecentFilesClicked)
        self.comboBox_recent_files.setMaxCount(10)

        # setup image window
        self.lbl_picture_origin = QLabel("input dicom image")
        self.lbl_picture_origin.setAlignment(Qt.AlignCenter)
        self.lbl_picture_origin.setFixedSize(600, 600)
        self.lbl_picture_origin.setStyleSheet("border: 1px solid black;")
        self.lbl_picture_predict = QLabel("result image")
        self.lbl_picture_predict.setAlignment(Qt.AlignCenter)
        self.lbl_picture_predict.setFixedSize(600, 600)
        self.lbl_picture_predict.setStyleSheet("border: 1px solid black;")

        # setup multiple image analyze buttons
        self.textLine_multiple_image = QLabel("# 다중 이미지 분석")
        self.textLine_multiple_image.setFixedHeight(60)
        self.textLine_multiple_image.setStyleSheet("border: 1px solid black; padding: 0px")
        self.pushButton_open_directory = QPushButton("Open Directory")
        self.pushButton_open_directory.clicked.connect(self.pushButtonOpenDirectoryClicked)
        self.pushButton_open_directory.setAutoDefault(True)

        self.pushButton_analyze = QPushButton("Analyze")
        self.pushButton_analyze.clicked.connect(self.pushButtonAnalyzeClicked)
        self.pushButton_analyze.setAutoDefault(True)

        self.pushButton_download_csv = QPushButton("Download CSV file")
        self.pushButton_download_csv.clicked.connect(self.pushButtonDownloadCSVClicked)
        self.pushButton_download_csv.setAutoDefault(True)

        # setup image preprocessing buttons
        self.textLine_info = QLabel("# 이미지 전처리")
        self.textLine_info.setFixedHeight(60)
        self.textLine_info.setStyleSheet("border: 1px solid black; padding: 0px")

        self.textLine_histogram_equalize = QLabel("Histogram Equalize")
        self.pushButton_histogram_equalize = QPushButton("Enter")
        self.pushButton_histogram_equalize.setFixedWidth(160)
        self.pushButton_histogram_equalize.clicked.connect(self.pushButtonHistEqualizeClicked)
        self.pushButton_histogram_equalize.setAutoDefault(True)

        self.textLine_enhancer_1 = QLabel("B: ")
        self.textLine_enhancer_2 = QLineEdit()
        self.textLine_enhancer_3 = QLabel("C: ")
        self.textLine_enhancer_4 = QLineEdit()
        self.pushButton_enhancer = QPushButton("Enter")
        self.pushButton_enhancer.clicked.connect(self.pushButtonEnhancerClicked)
        self.pushButton_enhancer.setAutoDefault(True)

        self.textLine_windowing = QLabel("Window/Level Tool")
        self.pushButton_windowing = QPushButton("Reset")
        self.pushButton_windowing.setFixedWidth(160)
        self.pushButton_windowing.clicked.connect(self.pushButtonWindowingClicked)
        self.pushButton_windowing.setAutoDefault(True)
        self.slider_windowing_wl = QSlider(Qt.Horizontal, self)
        self.wl_min = -500
        self.wl_max = 500
        self.slider_windowing_wl.setRange(self.wl_min, self.wl_max)
        self.slider_windowing_wl.setSingleStep(2)
        self.slider_windowing_wl.setValue(127)
        self.textLine_windowing_wl = QLabel("WL", alignment=Qt.AlignCenter)
        self.textLine_windowing_wl.setFixedSize(40, 20)
        self.label_windwoing_wl_min = QLabel(str(self.wl_min), alignment=Qt.AlignLeft)
        self.label_windwoing_wl_min.setFixedHeight(20)
        self.label_windwoing_wl_max = QLabel(str(self.wl_max), alignment=Qt.AlignRight)
        self.label_windwoing_wl_max.setFixedHeight(20)
        self.slider_windowing_ww = QSlider(Qt.Horizontal, self)
        self.ww_min = 1
        self.ww_max = 2000
        self.slider_windowing_ww.setRange(self.ww_min, self.ww_max)
        self.slider_windowing_ww.setSingleStep(2)
        self.slider_windowing_ww.setValue(255)
        self.textLine_windowing_ww = QLabel("WW", alignment=Qt.AlignCenter)
        self.textLine_windowing_ww.setFixedSize(40, 20)
        self.label_windwoing_ww_min = QLabel(str(self.ww_min), alignment=Qt.AlignLeft)
        self.label_windwoing_ww_min.setFixedHeight(20)
        self.label_windwoing_ww_max = QLabel(str(self.ww_max), alignment=Qt.AlignRight)
        self.label_windwoing_ww_max.setFixedHeight(20)
        self.slider_windowing_wl.valueChanged.connect(self.sliderWindowingChanged)
        self.slider_windowing_ww.valueChanged.connect(self.sliderWindowingChanged)

        # setup image analyze buttons
        self.textLine_analyze = QLabel("# 영상 분석 결과")
        self.textLine_analyze.setFixedHeight(60)
        self.textLine_analyze.setStyleSheet("border: 1px solid black; padding: 0px")

        self.textLine_windowing_gray = QLabel("Windowing w/ Background Elimination")
        self.pushButton_windowing_gray = QPushButton("Enter")
        self.pushButton_windowing_gray.setFixedWidth(160)
        self.pushButton_windowing_gray.clicked.connect(self.pushButtonWindowingGrayClicked)
        self.pushButton_windowing_gray.setAutoDefault(True)

        self.textLine_feature_extraction = QLabel("Feature Extraction")
        self.pushButton_feature_extraction = QPushButton("Enter")
        self.pushButton_feature_extraction.setFixedWidth(160)
        self.pushButton_feature_extraction.clicked.connect(self.pushButtonFeatureExtractionClicked)
        self.pushButton_feature_extraction.setAutoDefault(True)

        self.textLine_landmark = QLabel("Semantic Segmentation (FCN)")
        self.pushButton_landmark = QPushButton("Enter")
        self.pushButton_landmark.setFixedWidth(160)
        self.pushButton_landmark.clicked.connect(self.pushButtonLandmarkClicked)
        self.pushButton_landmark.setAutoDefault(True)

        # setup anatomical imaging range buttons
        self.textLine_coord = QLabel("# 해부학적 스캔 범위")
        self.textLine_coord.setFixedHeight(60)
        self.textLine_coord.setStyleSheet("border: 1px solid black; padding: 0px")

        self.textLine_coord_windowing = QLabel("Windowing w/ Background Elimination")
        self.pushButton_coord_windowing = QPushButton("Enter")
        self.pushButton_coord_windowing.setFixedWidth(160)
        self.pushButton_coord_windowing.clicked.connect(self.pushButtonCoordWindowingClicked)
        self.pushButton_coord_windowing.setAutoDefault(True)

        self.textLine_coord_feature_extraction = QLabel("Feature Extraction")
        self.pushButton_coord_feature_extraction = QPushButton("Enter")
        self.pushButton_coord_feature_extraction.setFixedWidth(160)
        self.pushButton_coord_feature_extraction.clicked.connect(self.pushButtonCoordFEClicked)
        self.pushButton_coord_feature_extraction.setAutoDefault(True)

        self.textLine_coord_fcn = QLabel("Semantic Segmentation (FCN)")
        self.pushButton_coord_fcn = QPushButton("Enter")
        self.pushButton_coord_fcn.setFixedWidth(160)
        self.pushButton_coord_fcn.clicked.connect(self.pushButtonCoordFCNClicked)
        self.pushButton_coord_fcn.setAutoDefault(True)

        # stack layouts
        self.layout = QVBoxLayout()
        layout_main = QHBoxLayout()
        layout_file_io = QVBoxLayout()
        layout_pics = QHBoxLayout()
        layout_pics.addWidget(self.lbl_picture_origin)
        layout_pics.addWidget(self.lbl_picture_predict)
        layout_buttons = QVBoxLayout()
        layout_multiple = QVBoxLayout()
        layout_info = QHBoxLayout()
        layout_histogram_equalize = QHBoxLayout()
        layout_enhancer = QHBoxLayout()
        layout_windowing = QHBoxLayout()
        layout_windowing_wl = QVBoxLayout()
        layout_windowing_wl_value = QHBoxLayout()
        layout_windowing_ww = QVBoxLayout()
        layout_windowing_ww_value = QHBoxLayout()
        layout_windowing_value = QHBoxLayout()
        layout_analyze = QHBoxLayout()
        layout_windowing_gray = QHBoxLayout()
        layout_feature_extraction = QHBoxLayout()
        layout_landmark = QHBoxLayout()
        layout_coord = QHBoxLayout()
        layout_coord_windowing = QHBoxLayout()
        layout_coord_fe = QHBoxLayout()
        layout_coord_fcn = QHBoxLayout()

        layout_multiple.addWidget(self.textLine_multiple_image)
        layout_multiple.addWidget(self.pushButton_open_directory)
        layout_multiple.addWidget(self.pushButton_analyze)
        layout_multiple.addWidget(self.pushButton_download_csv)
        layout_buttons.addLayout(layout_multiple)

        layout_info.addWidget(self.textLine_info)
        layout_buttons.addLayout(layout_info)

        layout_histogram_equalize.addWidget(self.textLine_histogram_equalize)
        layout_histogram_equalize.addWidget(self.pushButton_histogram_equalize)
        layout_buttons.addLayout(layout_histogram_equalize)

        layout_enhancer.addWidget(self.textLine_enhancer_1)
        layout_enhancer.addWidget(self.textLine_enhancer_2)
        layout_enhancer.addWidget(self.textLine_enhancer_3)
        layout_enhancer.addWidget(self.textLine_enhancer_4)
        layout_enhancer.addWidget(self.pushButton_enhancer)
        layout_buttons.addLayout(layout_enhancer)

        layout_windowing.addWidget(self.textLine_windowing)
        layout_windowing.addWidget(self.pushButton_windowing)
        layout_buttons.addLayout(layout_windowing)

        layout_windowing_wl.addWidget(self.slider_windowing_wl)
        layout_windowing_wl_value.addWidget(self.label_windwoing_wl_min, Qt.AlignLeft)
        layout_windowing_wl_value.addWidget(self.label_windwoing_wl_max, Qt.AlignRight)
        layout_windowing_wl.addLayout(layout_windowing_wl_value)
        layout_windowing_ww.addWidget(self.slider_windowing_ww)
        layout_windowing_ww_value.addWidget(self.label_windwoing_ww_min, Qt.AlignLeft)
        layout_windowing_ww_value.addWidget(self.label_windwoing_ww_max, Qt.AlignRight)
        layout_windowing_ww.addLayout(layout_windowing_ww_value)
        layout_windowing_value.addWidget(self.textLine_windowing_wl, Qt.AlignHCenter)
        layout_windowing_value.addLayout(layout_windowing_wl)
        layout_windowing_value.addWidget(self.textLine_windowing_ww, Qt.AlignHCenter)
        layout_windowing_value.addLayout(layout_windowing_ww)
        layout_buttons.addLayout(layout_windowing_value)

        layout_analyze.addWidget(self.textLine_analyze)
        layout_buttons.addLayout(layout_analyze)

        layout_windowing_gray.addWidget(self.textLine_windowing_gray)
        layout_windowing_gray.addWidget(self.pushButton_windowing_gray)
        layout_buttons.addLayout(layout_windowing_gray)

        layout_feature_extraction.addWidget(self.textLine_feature_extraction)
        layout_feature_extraction.addWidget(self.pushButton_feature_extraction)
        layout_buttons.addLayout(layout_feature_extraction)

        layout_landmark.addWidget(self.textLine_landmark)
        layout_landmark.addWidget(self.pushButton_landmark)
        layout_buttons.addLayout(layout_landmark)

        layout_coord.addWidget(self.textLine_coord)
        layout_buttons.addLayout(layout_coord)

        layout_coord_windowing.addWidget(self.textLine_coord_windowing)
        layout_coord_windowing.addWidget(self.pushButton_coord_windowing)
        layout_buttons.addLayout(layout_coord_windowing)

        layout_coord_fe.addWidget(self.textLine_coord_feature_extraction)
        layout_coord_fe.addWidget(self.pushButton_coord_feature_extraction)
        layout_buttons.addLayout(layout_coord_fe)

        layout_coord_fcn.addWidget(self.textLine_coord_fcn)
        layout_coord_fcn.addWidget(self.pushButton_coord_fcn)
        layout_buttons.addLayout(layout_coord_fcn)

        layout_file_io.addWidget(self.pushButton)
        layout_file_io.addWidget(self.comboBox_recent_files)
        layout_file_io.addLayout(layout_pics)
        layout_main.addLayout(layout_file_io)
        layout_main.addLayout(layout_buttons)
        self.layout.addLayout(layout_main)

        self.setLayout(self.layout)

        self.show()

    def pushButtonClicked(self):
        fname = QFileDialog.getOpenFileName(self)
        # open dicom file if valid
        if len(fname[0]) >= 1 and fname[0][-4:] == '.dcm':
            # change combo box list
            idx = self.comboBox_recent_files.findText(fname[0])
            if idx > 0:
                self.comboBox_recent_files.removeItem(idx)
            self.comboBox_recent_files.insertItem(1, fname[0])
            self.dcmfile = pydicom.dcmread(fname[0])
            self.pixel_array = self.dcmfile.pixel_array

            # show original input image with metadata
            img, info = display_metadata.metadata(fname[0], copy.deepcopy(self.pixel_array))
            self.imageUpdate(img, 'o')

            self.lbl_picture_predict.setText("result image")

    def pushButtonOpenDirectoryClicked(self):
        fname = QFileDialog.getExistingDirectory(self, "Select Directory")
        if os.path.exists(fname):
            # read all dicom file in directory
            self.dicom_file_list = os.listdir(fname)
            self.dicom_file_list = [os.path.join(fname, item) for item in self.dicom_file_list if item[-4:] == '.dcm']
            self.lbl_picture_origin.setText(str(len(self.dicom_file_list)) + " DICOM files")
            self.lbl_picture_predict.setText("multiple image processing mode")

    def pushButtonAnalyzeClicked(self):
        # make progress bar
        self.progressBar = QProgressBar(self)
        self.layout.addWidget(self.progressBar)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(len(self.dicom_file_list) - 1)
        self.result_list = [['filename', 'filesize', 'y0', 'y1', 'y2', 'y3']]
        for i, item in enumerate(self.dicom_file_list):
            dcmfile = pydicom.dcmread(item)
            if dcmfile.BodyPartExamined == 'CHEST':
                # save predict result in list
                pred_img = resize(dcmfile.pixel_array, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
                preds = self.loaded_model.predict([[pred_img]])
                y1, y2 = fcn.get_only_coord(img_as_ubyte(preds[0].squeeze()), len(dcmfile.pixel_array), len(dcmfile.pixel_array[0]))
                self.result_list.append([os.path.basename(item), os.path.getsize(item), 0, y1, y2, len(dcmfile.pixel_array) - 1])
            self.progressBar.setValue(i)
            QApplication.processEvents()
        # remove progress bar all done
        self.layout.removeWidget(self.progressBar)
        self.progressBar.deleteLater()

    def pushButtonDownloadCSVClicked(self):
        # write csv file
        fname = QFileDialog.getSaveFileName(self, "Save file", "", "CSV Files (*.csv)")
        w = open(fname[0], 'w', newline='')
        wr = csv.writer(w)
        for item in self.result_list:
            wr.writerow(item)
        w.close()

    def comboBoxRecentFilesClicked(self):
        idx = self.comboBox_recent_files.currentIndex()
        # change current dicom file if valid
        if idx > 0:
            # change combo box list
            fname = self.comboBox_recent_files.currentText()
            self.comboBox_recent_files.removeItem(idx)
            self.comboBox_recent_files.insertItem(1, fname)
            self.comboBox_recent_files.setCurrentIndex(0)
            self.dcmfile = pydicom.dcmread(fname)
            self.pixel_array = self.dcmfile.pixel_array

            # load new dicom image
            img, info = display_metadata.metadata(fname, copy.deepcopy(self.pixel_array))
            self.imageUpdate(img, 'o')

            self.lbl_picture_predict.setText("result image")

    def pushButtonHistEqualizeClicked(self):
        # do histogram equalization
        img = copy.deepcopy(self.pixel_array)
        img = histogram_equalization.hist_equalize(img)
        self.imageUpdate(img, 'p')

    def pushButtonEnhancerClicked(self):
        # get brightness value and contrast value and update image
        br = self.textLine_enhancer_2.text()
        br = float(br) if br else 1.0
        self.textLine_enhancer_2.setText("")
        ct = self.textLine_enhancer_4.text()
        ct = float(ct) if ct else 1.0
        self.textLine_enhancer_4.setText("")
        img = copy.deepcopy(self.pixel_array)
        img = brightness_contrast_control.bc_control(img, br, ct)
        self.imageUpdate(img, 'p')

    def pushButtonWindowingClicked(self):
        # reset window level and window center
        self.slider_windowing_wl.setValue(0)
        self.slider_windowing_wl.setValue(127)
        self.slider_windowing_ww.setValue(255)

    def sliderWindowingChanged(self):
        # apply windowing to original input image
        wl = self.slider_windowing_wl.value()
        ww = self.slider_windowing_ww.value()
        img = copy.deepcopy(self.pixel_array)
        self.dcmfile.WindowCenter = wl
        self.dcmfile.WindowWidth = ww
        img = windowing.windowing(img, self.dcmfile)
        self.imageUpdate(img, 'p')

    def pushButtonWindowingGrayClicked(self):
        # do gray scale windowing to input image
        img = copy.deepcopy(self.pixel_array)
        img = cv2.equalizeHist(img)
        img = black_white_windowing.black_white_windowing(img)
        self.imageUpdate(img, 'p')

    def pushButtonFeatureExtractionClicked(self):
        img = copy.deepcopy(self.pixel_array)
        # do canny edge detection if current dicom body part examined is chest
        if self.dcmfile.BodyPartExamined == 'CHEST':
            img = Image.fromarray(img)
            enhancer_ct = ImageEnhance.Contrast(img)
            img = np.array(enhancer_ct.enhance(2.5))
            img = canny_edge_detection.plot_canny(img)
            self.imageUpdate(img, 'p')
        # do super pixel clustering if current dicom body part examined is head
        if self.dcmfile.BodyPartExamined == 'HEAD':
            img = skull_super_pixel_clustering.clustering(img)
            self.imageUpdate(img, 'p')

    def pushButtonLandmarkClicked(self):
        # popup error message if current dicom body part examined is not chest
        if not self.dcmfile.BodyPartExamined == 'CHEST':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Not Implemented!!")
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        # predict landmark image and update input image with predict image
        pred_img = resize(self.pixel_array, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
        preds = self.loaded_model.predict([[pred_img]])
        img = resize(preds[0].squeeze(), output_shape=(len(self.pixel_array), len(self.pixel_array[0])), preserve_range=True)
        img = img_as_ubyte(img)
        img = fcn.landmark(img, copy.deepcopy(self.pixel_array))
        self.imageUpdate(img, 'p')

    def pushButtonCoordWindowingClicked(self):
        # apply histogram equalization to input image
        img = copy.deepcopy(self.pixel_array)
        img = cv2.equalizeHist(img)
        # find coordinate of landmark and update output image
        if self.dcmfile.BodyPartExamined == 'CHEST':
            img = black_white_windowing.coord_windowing(img)
            self.imageUpdate(img, 'p')
        elif self.dcmfile.BodyPartExamined == 'HEAD':
            img = skull.coord_windowing(img)
            self.imageUpdate(img, 'p')
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Not Implemented!!")
            msg.setWindowTitle("Error")
            msg.exec_()
            return

    def pushButtonCoordFEClicked(self):
        # popup error message if current dicom body part examined is not head
        if not self.dcmfile.BodyPartExamined == 'HEAD':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Not Implemented!!")
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        # find coordinate using super pixel clustering
        img = copy.deepcopy(self.pixel_array)
        img = skull_super_pixel_clustering.clustering_coord(img)
        self.imageUpdate(img, 'p')

    def pushButtonCoordFCNClicked(self):
        # popup error message if current dicom body part examined is not chest
        if not self.dcmfile.BodyPartExamined == 'CHEST':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Not Implemented!!")
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        # predict landmark image and update input image with predict image
        pred_img = resize(self.pixel_array, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
        preds = self.loaded_model.predict([[pred_img]])
        img = resize(preds[0].squeeze(), output_shape=(len(self.pixel_array), len(self.pixel_array[0])), preserve_range=True)
        img = img_as_ubyte(img)
        img, (start, end) = fcn.coord(img, copy.deepcopy(self.pixel_array))
        self.imageUpdate(img, 'p')

    def imageUpdate(self, img, update='p'):
        # update image window of demo program
        img = QImage(img, len(img), len(img[0]), QImage.Format_RGB888)
        qPixmapFileVar = QPixmap(img)
        qPixmapFileVar = qPixmapFileVar.scaledToHeight(600)
        qPixmapFileVar = qPixmapFileVar.scaledToWidth(600)
        if update == 'p':
            self.lbl_picture_predict.setPixmap(qPixmapFileVar)
        if update == 'o':
            self.lbl_picture_origin.setPixmap(qPixmapFileVar)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont("NanumGothicBold", 12, QFont.Bold))
    ex = MyApp()
    sys.exit(app.exec_())
