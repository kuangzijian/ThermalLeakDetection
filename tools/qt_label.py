import sys
import os
import cv2
import csv
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPixmap, QCursor

class ImageLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def enterEvent(self, event):
        super().enterEvent(event)
        current_label = self.parent().currentLabel
        if current_label in ['positive', 'negative', 'shadow']:
            self.setCursor(self.parent().cursorZoo[current_label])
        else:
            self.unsetCursor()

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)


class LabelingApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setMouseTracking(True)
        self.imageIndex = 0
        self.images = []
        self.directory = ''
        self.currentLabel = None
        self.currentImage = None
        self.currentdiff_map = None
        self.labeledBoxes = {}
        self.imageStates = {}
        self.history = []
        self.future = []
        self.saveStatus = {}
        self.showdiff_mapImage = True
        self.positiveCursor = QCursor(QPixmap("positive_green_cursor.png").scaled(32, 32), -1, -1)
        self.negativeCursor = QCursor(QPixmap("negative_yellow_cursor.png").scaled(32, 32), -1, -1)
        self.shadowCursor = QCursor(QPixmap("shadow_grey_cursor.png").scaled(32, 32), -1, -1)
        self.defaultCursor = self.cursor()  # Save the default cursor
        self.cursorZoo = {"positive": self.positiveCursor,
                          "negative": self.negativeCursor,
                          "shadow": self.shadowCursor,
                          None: self.defaultCursor}

    def initUI(self):
        self.setMouseTracking(True)
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('LeakAI patch data Labeler')
        layout = QtWidgets.QVBoxLayout()

        self.openButton = QtWidgets.QPushButton('Open Directory', self)
        self.openButton.clicked.connect(self.openDirectory)
        layout.addWidget(self.openButton)

        imageLayout = QtWidgets.QHBoxLayout()

        self.imageLabel = ImageLabel(self)
        self.imageLabel.setFixedSize(640, 480)  # Set the fixed size for imageLabel
        self.imageLabel.mousePressEvent = self.mousePressEvent
        self.imageLabel.mouseMoveEvent = self.mouseMoveEvent
        imageLayout.addWidget(self.imageLabel)

        self.diff_mapImageLabel = QtWidgets.QLabel(self)
        self.diff_mapImageLabel.setAlignment(QtCore.Qt.AlignTop)
        self.diff_mapImageLabel.setFixedSize(640, 480)  # Set the same fixed size for diff_mapImageLabel
        imageLayout.addWidget(self.diff_mapImageLabel)

        layout.addLayout(imageLayout)

        # self.togglediff_mapButton = QtWidgets.QPushButton('Show/Hide Difference Map', self)
        # self.togglediff_mapButton.clicked.connect(self.togglediff_mapImage)
        # layout.addWidget(self.togglediff_mapButton)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.valueChanged.connect(self.sliderMoved)
        layout.addWidget(self.slider)

        indexAndSaveLayout = QtWidgets.QHBoxLayout()

        indexLayout = QtWidgets.QHBoxLayout()
        indexLayout.setAlignment(QtCore.Qt.AlignLeft)

        self.currentIndexEdit = QtWidgets.QLineEdit(self)
        self.currentIndexEdit.setText("1")
        self.currentIndexEdit.returnPressed.connect(self.indexEntered)
        self.currentIndexEdit.setFixedWidth(50)
        self.currentIndexEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        indexLayout.addWidget(self.currentIndexEdit)

        self.currentIndexEdit.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.totalLabel = QtWidgets.QLabel(" / 1", self)
        indexLayout.addWidget(self.totalLabel)

        indexAndSaveLayout.addLayout(indexLayout)

        # Add a spacer to push the saveStatusLabel to the right
        indexAndSaveLayout.addStretch()

        self.saveStatusLabel = QtWidgets.QLabel("Not Saved", self)
        self.saveStatusLabel.setStyleSheet("QLabel { color: red; font-size: 18pt; }")
        indexAndSaveLayout.addWidget(self.saveStatusLabel)

        layout.addLayout(indexAndSaveLayout)

        buttonLayout = QtWidgets.QHBoxLayout()
        
        self.positiveButton = QtWidgets.QPushButton('Label Positive (1)', self)
        self.positiveButton.setStyleSheet("background-color: rgb(0,255,0)")
        self.positiveButton.clicked.connect(lambda: self.setCurrentLabel('positive'))
        buttonLayout.addWidget(self.positiveButton)

        self.negativeButton = QtWidgets.QPushButton('Label Negative (2)', self)
        self.negativeButton.setStyleSheet("background-color: rgb(255,255,0)")
        self.negativeButton.clicked.connect(lambda: self.setCurrentLabel('negative'))
        buttonLayout.addWidget(self.negativeButton)

        self.shadowButton = QtWidgets.QPushButton('Label Shadow (3)', self)
        self.shadowButton.setStyleSheet("background-color: rgb(128,128,128)")
        self.shadowButton.clicked.connect(lambda: self.setCurrentLabel('shadow'))
        buttonLayout.addWidget(self.shadowButton)

        self.undoButton = QtWidgets.QPushButton('Undo (Ctrl+Z)', self)
        self.undoButton.clicked.connect(self.undoAction)
        buttonLayout.addWidget(self.undoButton)

        self.redoButton = QtWidgets.QPushButton('Redo (Ctrl+Y)', self)
        self.redoButton.clicked.connect(self.redoAction)
        buttonLayout.addWidget(self.redoButton)

        self.nextButton = QtWidgets.QPushButton('Next Image (D)', self)
        self.nextButton.clicked.connect(self.nextImage)
        buttonLayout.addWidget(self.nextButton)

        self.prevButton = QtWidgets.QPushButton('Previous Image (A)', self)
        self.prevButton.clicked.connect(self.prevImage)
        buttonLayout.addWidget(self.prevButton)

        self.saveButton = QtWidgets.QPushButton('Save Data (Cmd/Ctrl + S)', self)
        self.saveButton.clicked.connect(self.saveData)
        buttonLayout.addWidget(self.saveButton)

        layout.addLayout(buttonLayout)
        self.setLayout(layout)



    def checkAndUpdateLabels(self):
        positive_path = os.path.join(self.directory, 'Positive')
        negative_path = os.path.join(self.directory, 'Negative')
        shadow_path = os.path.join(self.directory, 'Shadow')
        
        if not os.path.exists(positive_path) and not os.path.exists(negative_path) and not os.path.exists(shadow_path):
            return  # No Positive/Negative/Shadow folders to check

        for imagePath in self.images:
            baseName = os.path.basename(imagePath)
            annotationPath = os.path.join(self.directory, 'labels', baseName.replace('.jpg', '.txt'))
            if not os.path.exists(annotationPath):
                continue
            
            # Load the current image if not already loaded
            self.currentImage = cv2.imread(imagePath)
            if self.currentImage is None:
                continue  # Skip if the image cannot be loaded
            
            updated_boxes = []
            update_needed = False
            with open(annotationPath, 'r') as file:
                for line in file:
                    data = line.split()
                    if len(data) == 6:
                        updated_boxes.append(line.strip())  # Already labeled, keep as is
                        continue
                    
                    x_center, y_center, width, height = map(float, data[1:5])
                    x = int((x_center - width / 2) * self.currentImage.shape[1])
                    y = int((y_center - height / 2) * self.currentImage.shape[0])
                    w = int(width * self.currentImage.shape[1])
                    h = int(height * self.currentImage.shape[0])
                    
                    # Determine the label by checking Positive/Negative/Shadow folders
                    label = None
                    image_name = os.path.splitext(baseName)[0]
                    filename = f"{image_name}_{x}_{y}_{w}_{h}.jpg"
                    if os.path.exists(os.path.join(positive_path, filename)):
                        label = '1'
                    elif os.path.exists(os.path.join(negative_path, filename)):
                        label = '0'
                    elif os.path.exists(os.path.join(shadow_path, filename)):
                        label = '2'
                    
                    if label is not None:
                        updated_boxes.append(f"0 {x_center} {y_center} {width} {height} {label}")
                        update_needed = True
                    else:
                        updated_boxes.append(f"0 {x_center} {y_center} {width} {height}")
            
            if update_needed:
                # Write updated annotations back to file
                with open(annotationPath, 'w') as file:
                    for box in updated_boxes:
                        file.write(box + '\n')

    def openDirectory(self):
        self.directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        self.images = sorted([os.path.join(self.directory, 'images', f.replace("txt", "jpg")) for f in os.listdir(os.path.join(self.directory, 'labels')) if f.endswith('.txt')])
        self.loadStats()
        self.checkAndUpdateLabels()  # Check and update labels based on Positive/Negative/Shadow folders
        self.imageIndex = 0
        self.loadImageAndAnnotations()
        self.slider.setMaximum(len(self.images))
        self.updateIndexDisplay()
        self.checkdiff_mapImages()

    def checkdiff_mapImages(self):
        diff_map_dir = os.path.join(self.directory, 'diff_maps')
        if not os.path.exists(diff_map_dir) or not any(fname.endswith('.jpg') or fname.endswith('.png') for fname in os.listdir(diff_map_dir)):
            # self.togglediff_mapButton.setDisabled(True)
            QtWidgets.QMessageBox.information(self, 'Information', 'No diff_map images found in diff_maps folder.')
        # else:
        #     self.togglediff_mapButton.setDisabled(False)

    # def togglediff_mapImage(self):
    #     self.showdiff_mapImage = not self.showdiff_mapImage
    #     self.updateImage()

    def sliderMoved(self, value):
        if 0 <= value - 1 < len(self.images):
            self.imageIndex = value - 1
            self.loadImageAndAnnotations()
            self.updateIndexDisplay()

    def indexEntered(self):
        index_str = self.currentIndexEdit.text()
        try:
            index = int(index_str) - 1
            if 0 <= index < len(self.images):
                self.slider.setValue(index + 1)
                self.imageIndex = index
                self.loadImageAndAnnotations()
                self.currentIndexEdit.clearFocus()
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Invalid index entered.")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter a valid number.")

    def loadImageAndAnnotations(self):
        imagePath = self.images[self.imageIndex]
        baseName = os.path.basename(imagePath)
        diff_mapPath = os.path.join(self.directory, 'diff_maps', baseName)
        annotationPath = os.path.join(self.directory, 'labels', baseName.replace('.jpg', '.txt'))
        
        print(f"Loading image: {imagePath}")
        print(f"Loading diff_map image: {diff_mapPath}")

        self.currentImage = cv2.imread(imagePath)
        self.currentdiff_map = cv2.imread(diff_mapPath)

        if self.currentImage is not None:
            self.currentImage = cv2.resize(self.currentImage, (640, 480))
        else:
            print(f"Failed to load image: {imagePath}")
        if self.currentdiff_map is not None:
            self.currentdiff_map = cv2.resize(self.currentdiff_map, (640, 480))
        else:
            print(f"Failed to load diff_map image: {diff_mapPath}")

        saved = self.saveStatus.get(self.imageIndex, False)
        self.updateSaveStatusLabel(saved)

        if imagePath not in self.imageStates:
            self.labeledBoxes = []
            try:
                with open(annotationPath, 'r') as file:
                    for line in file:
                        data = line.split()
                        x_center, y_center, width, height = map(float, data[1:5])
                        label = data[5] if len(data) > 5 else None
                        x = int((x_center - width / 2) * self.currentImage.shape[1])
                        y = int((y_center - height / 2) * self.currentImage.shape[0])
                        w = int(width * self.currentImage.shape[1])
                        h = int(height * self.currentImage.shape[0])
                        box_label = 'positive' if label == '1' else 'negative' if label == '0' else 'shadow' if label == '2' else None
                        self.labeledBoxes.append({'coordinates': (x, y, w, h), 'label': box_label})
                        print(f"Loaded box: {x, y, w, h} with label: {box_label}")  # Debug print
            except FileNotFoundError:
                print(f"No annotation file found for {imagePath}")
            self.imageStates[imagePath] = self.labeledBoxes
        else:
            self.labeledBoxes = self.imageStates[imagePath]

        self.updateImage()
        self.scale_w = self.imageLabel.width() / self.currentImage.shape[1]
        self.scale_h = self.imageLabel.height() / self.currentImage.shape[0]
        self.updateIndexDisplay()

    def updateImage(self):
        if self.currentImage is None:
            return

        tempImage = self.currentImage.copy()
        tempdiff_map = self.currentdiff_map.copy() if self.currentdiff_map is not None else None

        for box in self.labeledBoxes:
            color = (255, 255, 255)
            if box['label'] == 'positive':
                color = (0, 255, 0)
            elif box['label'] == 'negative':
                color = (0, 255, 255)
            elif box['label'] == 'shadow':
                color = (128, 128, 128)
            x, y, w, h = box['coordinates']
            cv2.rectangle(tempImage, (x, y), (x + w, y + h), color, 2)
            if tempdiff_map is not None:
                cv2.rectangle(tempdiff_map, (x, y), (x + w, y + h), color, 2)
            print(f"Drawing box: {x, y, w, h} with color: {color}")  # Debug print

        print("Updating image with labeled boxes")

        self.displayImage = self.convertCvQt(tempImage)
        self.imageLabel.setPixmap(self.displayImage)

        if self.showdiff_mapImage and tempdiff_map is not None:
            self.displaydiff_map = self.convertCvQt(tempdiff_map)
            self.diff_mapImageLabel.setPixmap(self.displaydiff_map.scaled(self.diff_mapImageLabel.size(), QtCore.Qt.KeepAspectRatio))
        else:
            self.diff_mapImageLabel.clear()
        
        self.imageLabel.repaint()

    def updateIndexDisplay(self):
        self.currentIndexEdit.setText(str(self.imageIndex + 1))
        self.totalLabel.setText(f"/ {len(self.images)}")

    def setCurrentLabel(self, label):
        self.currentLabel = label
        if label in ['positive', 'negative', 'shadow']:
            self.imageLabel.setCursor(self.cursorZoo[label])
        else:
            self.imageLabel.unsetCursor()
        print(f"Label set to {label}")

    def mousePressEvent(self, event):
        if not self.currentIndexEdit.underMouse():
            self.currentIndexEdit.clearFocus()
        if self.currentImage is None:
            return

        x, y = event.pos().x(), event.pos().y()

        labeled = False
        for box in self.labeledBoxes:
            bx, by, bw, bh = box['coordinates']
            if bx <= x / self.scale_w <= bx + bw and by <= y / self.scale_h <= by + bh:
                self.history.append((self.imageIndex, box.copy()))
                box['label'] = self.currentLabel
                labeled = True

        if labeled:
            self.updateImage()
            self.future.clear()

    def mouseMoveEvent(self, event):
        if self.currentImage is None:
            return

        x, y = event.pos().x(), event.pos().y()
        found = False

        for box in self.labeledBoxes:
            bx, by, bw, bh = box['coordinates']
            scaled_x = x / self.scale_w
            scaled_y = y / self.scale_h

            if (bx <= scaled_x <= bx + bw) and (by <= scaled_y <= by + bh):
                self.imageLabel.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                found = True
                break

        if not found:
            if 0 < x < 640 and 0 < y < 480:
                self.imageLabel.setCursor(self.cursorZoo[self.currentLabel])
            else:
                self.imageLabel.unsetCursor()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if key == QtCore.Qt.Key_1:
            self.setCurrentLabel('positive')
        elif key == QtCore.Qt.Key_2:
            self.setCurrentLabel('negative')
        elif key == QtCore.Qt.Key_3:
            self.setCurrentLabel('shadow')
        elif key == QtCore.Qt.Key_D:
            self.nextImage()
        elif key == QtCore.Qt.Key_A:
            self.prevImage()
        elif (modifiers == QtCore.Qt.ControlModifier or modifiers == QtCore.Qt.MetaModifier) and key == QtCore.Qt.Key_S:
            self.saveData()
        elif (modifiers == QtCore.Qt.ControlModifier or modifiers == QtCore.Qt.MetaModifier) and key == QtCore.Qt.Key_Z:
            self.undoAction()
        elif (modifiers == QtCore.Qt.ControlModifier or modifiers == QtCore.Qt.MetaModifier) and key == QtCore.Qt.Key_Y:
            self.redoAction()

    def updateSaveStatusLabel(self, saved):
        if saved:
            self.saveStatusLabel.setText("Saved")
            self.saveStatusLabel.setStyleSheet("QLabel { color: green; font-size: 18pt; }")
        else:
            self.saveStatusLabel.setText("Not Saved")
            self.saveStatusLabel.setStyleSheet("QLabel { color: red; font-size: 18pt; }")

    def saveData(self):
        positive_path = os.path.join(self.directory, 'Positive')
        negative_path = os.path.join(self.directory, 'Negative')
        shadow_path = os.path.join(self.directory, 'Shadow')
        os.makedirs(positive_path, exist_ok=True)
        os.makedirs(negative_path, exist_ok=True)
        os.makedirs(shadow_path, exist_ok=True)
        annotationPath = os.path.join(self.directory, 'labels', os.path.basename(self.images[self.imageIndex]).replace('.jpg', '.txt'))
        with open(annotationPath, 'w') as file:
            for box in self.labeledBoxes:
                if box['label']:
                    bx, by, bw, bh = box['coordinates']
                    label = '1' if box['label'] == 'positive' else '0' if box['label'] == 'negative' else '2'
                    x_center = (bx + bw / 2) / self.currentImage.shape[1]
                    y_center = (by + bh / 2) / self.currentImage.shape[0]
                    width = bw / self.currentImage.shape[1]
                    height = bh / self.currentImage.shape[0]
                    self.deleteSavedData(box)  # Ensure previous labels are deleted
                    file.write(f"0 {x_center} {y_center} {width} {height} {label}\n")
                    label_dir = positive_path if box['label'] == 'positive' else negative_path if box['label'] == 'negative' else shadow_path
                    patch = self.currentImage[by:by+bh, bx:bx+bw]
                    image_name = os.path.splitext(os.path.basename(self.images[self.imageIndex]))[0]
                    filename = f"{image_name}_{bx}_{by}_{bw}_{bh}.jpg"
                    cv2.imwrite(os.path.join(label_dir, filename), patch)
                    print(f"Data saved for image {image_name}!")
        self.saveStatus[self.imageIndex] = True
        self.updateSaveStatusLabel(True)
        self.saveStats()

    def convertCvQt(self, cvImg):
        rgbImage = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        
        print(f"Converting image: {cvImg.shape} to QPixmap")

        return QPixmap.fromImage(convertToQtFormat)

    def nextImage(self):
        if self.imageIndex + 1 < len(self.images):
            self.imageIndex += 1
            self.slider.setValue(self.imageIndex + 1)
            self.loadImageAndAnnotations()
        else:
            QtWidgets.QMessageBox.information(self, 'Information', 'No more images to display.')

    def prevImage(self):
        if self.imageIndex > 0:
            self.imageIndex -= 1
            self.slider.setValue(self.imageIndex + 1)
            self.loadImageAndAnnotations()
        else:
            QtWidgets.QMessageBox.information(self, 'Information', 'This is the first image.')

    def loadStats(self):
        self.saveStatus = {}
        try:
            with open(os.path.join(self.directory, 'stats.csv'), mode='r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    index = int(row['index'])
                    is_saved = row['is_saved'] == 'True'
                    self.saveStatus[index] = is_saved
        except FileNotFoundError:
            self.saveStatus = {i: False for i in range(len(self.images))}
            self.saveStats()

    def saveStats(self):
        with open(os.path.join(self.directory, 'stats.csv'), mode='w', newline='') as file:
            fieldnames = ['image_name', 'index', 'is_saved']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for index, saved in self.saveStatus.items():
                image_name = self.images[index].split("/")[-1] if index < len(self.images) else "Unknown"
                writer.writerow({'image_name': image_name, 'index': index, 'is_saved': saved})

    def undoAction(self):
        if self.history:
            last_action = self.history.pop()
            image_index, box_state = last_action
            if image_index == self.imageIndex:
                for box in self.labeledBoxes:
                    if box['coordinates'] == box_state['coordinates']:
                        self.deleteSavedData(box)  # Delete the saved data first
                        self.future.append((self.imageIndex, box.copy()))
                        box['label'] = box_state['label']
                        break
                self.updateImage()
                self.updateSaveStatusLabel(False)

    def redoAction(self):
        if self.future:
            next_action = self.future.pop()
            image_index, box_state = next_action
            if image_index == self.imageIndex:
                for box in self.labeledBoxes:
                    if box['coordinates'] == box_state['coordinates']:
                        self.deleteSavedData(box)  # Delete the saved data first
                        self.history.append((self.imageIndex, box.copy()))
                        box['label'] = box_state['label']
                        break
                self.updateImage()
                self.updateSaveStatusLabel(False)

    def deleteSavedData(self, box):
        bx, by, bw, bh = box['coordinates']
        label_dirs = ['Positive', 'Negative', 'Shadow']
        image_name = os.path.splitext(os.path.basename(self.images[self.imageIndex]))[0]
        filename = f"{image_name}_{bx}_{by}_{bw}_{bh}.jpg"
        for label_dir in label_dirs:
            file_path = os.path.join(self.directory, label_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f'Deleted {file_path}')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = LabelingApp()
    ex.show()
    sys.exit(app.exec_())