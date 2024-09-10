import cv2
import subprocess
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
#from PyQt5.QtCore import QThread, pyqtSignal, QImage
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtGui import QPixmap, QImage
import torch
import easyocr
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import os
from ultralytics import YOLO
from DBHelper import BDDeManager
from datetime import datetime


# Initialiser le lecteur OCR
reader = easyocr.Reader(['en'], gpu=False)

# Dictionnaire de mappage pour la conversion des caractères
dict_char_to_int = {
    'O': '0',
    'I': '1',
    'J': '3',
    'A': '4',
    'G': '6',
    'S': '5'
}

# Ajout de chemins vers les plugins de Qt
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms'
os.environ['QT_QPA_PLATFORM'] = 'xcb'


class FrameGrabber(QtCore.QThread):
    signal = QtCore.pyqtSignal(QtGui.QImage, list)
    alertSignal = QtCore.pyqtSignal(str)

    def __init__(self, use_ip_camera=False, ip_address=None, port_number=None, stream_url=None, parent=None):
        super(FrameGrabber, self).__init__(parent)
        self.use_ip_camera = use_ip_camera
        self.ip_address = ip_address
        self.port_number = port_number
        self.stream_url = stream_url
        self.detected_plates = set()

        # Initialize YOLOv9 model from Ultralytics
        self.model = YOLO("best.pt")
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'])

    def capture_video(self):
        if self.use_ip_camera and self.stream_url:
            yield from self._capture_ip_camera()
        else:
            yield from self._capture_local_camera()

    def _capture_ip_camera(self):
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open IP camera stream: {self.stream_url}")
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from IP camera stream")
            yield frame

    def _capture_local_camera(self):
        command = [
            "libcamera-vid",
            "--codec", "mjpeg",
            "--width", "640",
            "--height", "480",
            "--framerate", "30",
            "-o", "-",  # Output to stdout
            "--timeout", "0"  # Capture indefinitely
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
        if process:
            stream = process.stdout
            buffer = b''
            while True:
                buffer += stream.read(4096)
                start_marker = buffer.find(b'\xff\xd8')
                end_marker = buffer.find(b'\xff\xd9')
                if start_marker != -1 and end_marker != -1:
                    jpeg_data = buffer[start_marker:end_marker+2]
                    buffer = buffer[end_marker+2:]
                    frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        yield frame

    def run(self):
        try:
            for frame in self.capture_video():
                ocr_results = self.process_frame(frame)
                qImg = self.convert_to_qimage(frame)
                self.signal.emit(qImg, ocr_results)
        except RuntimeError as e:
            self.alertSignal.emit(str(e))

    def process_frame(self, frame):
        results = self.model.predict(frame, show=False)
        label = 'licence plate'
        ocr_results = []
        for box in results[0].boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            plate_region = frame[ymin:ymax, xmin:xmax]
            plate_text = self.getOCR(plate_region)
            self.detected_plates.add(plate_text)
            ocr_results.append(plate_text.strip())
            self.draw_label(frame, xmin, ymin, xmax, ymax, plate_text, confidence)
        return ocr_results

    def convert_to_qimage(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytesPerLine = ch * w
        return QtGui.QImage(image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)

    def draw_label(self, frame, xmin, ymin, xmax, ymax, plate_text, confidence):
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        text = f"{plate_text} License plate {confidence:.2f}"
        cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def getOCR(self, image):
        preprocessed_image = preprocess_image(image)
        plate_text, _ = read_license_plate(preprocessed_image, self.reader)
        return plate_text if plate_text else ""

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 5)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def read_license_plate(license_plate_crop, reader):
    detections = reader.readtext(license_plate_crop)
    best_text = None
    max_score = 0
    for detection in detections:
        bbox, text, score = detection
        if score > max_score:
            max_score = score
            best_text = text.upper().replace(' ', '')
    if best_text and license_complies_format(best_text):
        return format_license(best_text), max_score
    else:
        return None, None

def license_complies_format(text):
    if len(text) != 10:
        return False
    for char in text:
        if char.isdigit() or char in dict_char_to_int.keys():
            continue
        else:
            return False
    return True

def format_license(text):
    license_plate = ''
    for j in range(len(text)):
        if text[j] in dict_char_to_int.keys():
            license_plate += dict_char_to_int[text[j]]
        else:
            license_plate += text[j]
    return license_plate


        
class AddPlateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ajouter une plaque d'immatriculation")
        self.resize(300, 150)
       
        layout = QVBoxLayout(self)
       
        self.label = QLabel("Numéro de plaque:")
        layout.addWidget(self.label)
       
        self.lineEdit = QLineEdit()
        layout.addWidget(self.lineEdit)
       
        self.button = QPushButton("Ajouter")
        layout.addWidget(self.button)
        self.button.clicked.connect(self.add_plate)
       
    def add_plate(self):
        plate_text = self.lineEdit.text().strip()
        if plate_text:
            # Ajouter le numéro de plaque à la base de données ou à toute autre logique de traitement
            print(f"Ajout de la plaque: {plate_text}")
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez entrer un numéro de plaque valide.")


class Ui_MainWindow(QtWidgets.QMainWindow):
   
    # Déclaration du signal personnalisé avec une liste de paramètres en fonction de vos besoins
    plateDetectedInDB = pyqtSignal(str, str)

    def __init__(self, MainWindow):
        super().__init__()
        self.MainWindow = MainWindow
        self.setupUi(self.MainWindow)
        self.grabber = FrameGrabber()
        self.grabber.signal.connect(self.updateFrame)
        self.grabber.start()
        self.db_manager = BDDeManager()  # Initialisation de la bdd
        # Initialize row counter
        self.current_row = 0
        self.db_manager.create_table()  # creation de table
        # chargement de la data
        self.load_data_from_db()

    def load_data_from_db(self):
        data = self.db_manager.recup()
        if data:
            self.populate_table(data)

    def populate_table(self, data):
        self.tableWidget.setRowCount(len(data))
        for row_index, row_data in enumerate(data):
            if len(row_data) == 3:
                id, numbers, date_time = row_data
                plate_text = numbers
                # Affichage de plate_text dans le tableau
                self.tableWidget.setItem(row_index, 0, QtWidgets.QTableWidgetItem(plate_text))
                # Affichage de date_time dans le tableau
                self.tableWidget.setItem(row_index, 1, QtWidgets.QTableWidgetItem(date_time))
            else:
                print(f"Unexpected data format in row {row_index}: {row_data}")



    def closeEvent(self, event):
        # fermuture de la connexion de la bdd
        self.db_manager.close_connection()
        event.accept()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(840, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 640, 480))
        self.label.setObjectName("label")
        
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(640, 0, 200, 450))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setRowCount(15)
        self.tableWidget.setHorizontalHeaderLabels(['plate', 'time'])
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        
        self.addPlateButton = QtWidgets.QPushButton(self.centralwidget)
        self.addPlateButton.setGeometry(QtCore.QRect(640, 400, 200, 25))
        self.addPlateButton.setObjectName("addPlateButton")
        self.addPlateButton.setText("Add License Plate")
        
        self.removePlateButton = QtWidgets.QPushButton(self.centralwidget)
        self.removePlateButton.setGeometry(QtCore.QRect(640, 430, 200, 25))
        self.removePlateButton.setObjectName("removePlateButton")
        self.removePlateButton.setText("Remove Selected Plate")
        
        # self.changeCameraButton = QtWidgets.QPushButton(self.centralwidget)
        # self.changeCameraButton.setGeometry(QtCore.QRect(640, 370, 200, 25))
        # self.changeCameraButton.setObjectName("changeCameraButton")
        # self.changeCameraButton.setText("Change to IP Camera")
        
        self.autoAdd = QtWidgets.QPushButton(self.centralwidget)
        self.autoAdd.setGeometry(QtCore.QRect(640, 320, 200, 25))
        self.autoAdd.setObjectName("autoAdd")
        self.autoAdd.setText("Auto add")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.plateDetectedInDB.connect(self.handle_plate_detected_in_db)
        self.addPlateButton.clicked.connect(self.openAddPlateDialog)
        self.removePlateButton.clicked.connect(self.removeSelectedPlate)
        #self.changeCameraButton.clicked.connect(self.changeToIPCamera)
        self.autoAdd.clicked.connect(self.toggleAutoAdd)

        # Connect the itemChanged signal to a slot for handling changes
        self.tableWidget.itemChanged.connect(self.handleItemChanged)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Initial state for auto add
        self.autoAddEnabled = True
        
    
    # def changeToIPCamera(self):
    #     dialog = IPConfigDialog(self)
    #     if dialog.exec_() == QtWidgets.QDialog.Accepted:
    #         ip_address = dialog.ipLineEdit.text()
    #         port_number = dialog.portLineEdit.text()
    #         stream_url = dialog.streamLineEdit.text()

    #         # Stop the current camera grabber
    #         self.grabber.terminate()

    #         # Initialize a new camera grabber with the IP camera stream
    #         self.grabber = FrameGrabber(ip_address, port_number, stream_url)
    #         self.grabber.signal.connect(self.updateFrame)
    #         self.grabber.start()

    def showAlertMessage(self, message):
        # Méthode pour afficher l'alerte dans votre interface utilisateur
        QtWidgets.QMessageBox.warning(self, "Alert", message)
    
        
    def handleItemChanged(self, item):
        row = item.row()
        column = item.column()
        new_text = item.text()
        
        if column == 0:  # Only handle changes in the 'plate' column
            # Update the database with the new plate text
            old_plate_text = self.tableWidget.item(row, 0).text()
            self.db_manager.update_plate(old_plate_text, new_text)
            

    def removeSelectedPlate(self):
        selected_row = self.tableWidget.currentRow()
        if selected_row >= 0:
            plate_text_item = self.tableWidget.item(selected_row, 0)
            plate_text = plate_text_item.text()
            self.db_manager.delete_plate(plate_text)
            self.tableWidget.removeRow(selected_row)

        
    
    def toggleAutoAdd(self):
        self.autoAddEnabled = not self.autoAddEnabled
        if self.autoAddEnabled:
            self.autoAdd.setText("Disable Auto Add")
        else:
            self.autoAdd.setText("Enable Auto Add")


    
    
    def openAddPlateDialog(self):
        #je doit la modifier pour la verification de length de text saisie 
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Add License Plate")
        dialog.resize(300, 200)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Numéro de série
        self.serialInput = QtWidgets.QLineEdit(dialog)
        self.serialInput.setPlaceholderText("Enter 5-digit serial number")
        layout.addWidget(self.serialInput)
        
        # Année
        self.yearInput = QtWidgets.QLineEdit(dialog)
        self.yearInput.setPlaceholderText("Enter 3-digit year")
        layout.addWidget(self.yearInput)
        
        # Wilaya
        self.wilayaInput = QtWidgets.QLineEdit(dialog)
        self.wilayaInput.setPlaceholderText("Enter 2-digit wilaya code")
        layout.addWidget(self.wilayaInput)
        
        # Boutons OK et Annuler
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(self.addLicensePlate)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.exec_()
        
    def addLicensePlate(self):
        serial = self.serialInput.text()
        year = self.yearInput.text()
        wilaya = self.wilayaInput.text()
        
        if len(serial) != 5 or not serial.isdigit():
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Serial number must be 5 digits.")
            return
        
        if len(year) != 3 or not year.isdigit():
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Year must be 3 digits.")
            return
        
        if len(wilaya) != 2 or not wilaya.isdigit():
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Wilaya must be 2 digits.")
            return
        
        plate_text = f"{serial}{year}{wilaya}"
        
        # Ajoutez la nouvelle plaque à la table et à la base de données
        self.db_manager.insertion(plate_text)
        current_time = QtCore.QDateTime.currentDateTime()
        time_str = current_time.toString(QtCore.Qt.DefaultLocaleLongDate)
        
        new_row = 0
        if self.current_row >= self.tableWidget.rowCount():
            self.tableWidget.removeRow(self.tableWidget.rowCount() - 1)
            self.current_row -= 1
        
        for row in range(self.current_row, 0, -1):
            for col in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row - 1, col)
                if item:
                    self.tableWidget.setItem(row, col, QtWidgets.QTableWidgetItem(item.text()))
                else:
                    self.tableWidget.setItem(row, col, QtWidgets.QTableWidgetItem(""))
        
        self.tableWidget.setItem(new_row, 0, QtWidgets.QTableWidgetItem(plate_text))
        self.tableWidget.setItem(new_row, 1, QtWidgets.QTableWidgetItem(time_str))
        
        if self.current_row < self.tableWidget.rowCount() - 1:
            self.current_row += 1

        QtWidgets.QMessageBox.information(self, "Success", "License plate added successfully.")



        
    def check_plate_in_db(self, plate_text):
        # Vérifie si la plaque spécifiée existe dans la base de données
        data = self.db_manager.recup()
        for row_data in data:
            id, numbers, date_time = row_data
            if numbers == plate_text:
                return True
        return False

    @QtCore.pyqtSlot(QtGui.QImage, list)
    def updateFrame(self, image, ocr_results):
        self.label.setPixmap(QPixmap.fromImage(image))
        # Update table with new OCR results
        for plate_text in ocr_results:
            if plate_text:
                # verifier si la plaque est detecter dans la base de donnes 
                if self.check_plate_in_db(plate_text):
                    now = datetime.now()
                    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    self.plateDetectedInDB.emit(plate_text, date_time)
                elif self.autoAddEnabled:
                    new_row = 0
                    if self.current_row >= self.tableWidget.rowCount():
                        self.tableWidget.removeRow(self.tableWidget.rowCount() - 1)
                        self.current_row -= 1
                    for row in range(self.current_row, 0, -1):
                        for col in range(self.tableWidget.columnCount()):
                            item = self.tableWidget.item(row - 1, col)
                            if item:
                                self.tableWidget.setItem(row, col, QtWidgets.QTableWidgetItem(item.text()))
                            else:
                                self.tableWidget.setItem(row, col, QtWidgets.QTableWidgetItem(""))
                    self.tableWidget.setItem(new_row, 0, QtWidgets.QTableWidgetItem(plate_text))
                    current_time = QtCore.QDateTime.currentDateTime()
                    time_str = current_time.toString(QtCore.Qt.DefaultLocaleLongDate)
                    self.tableWidget.setItem(new_row, 1, QtWidgets.QTableWidgetItem(time_str))
                    if self.current_row < self.tableWidget.rowCount() - 1:
                        self.current_row += 1
                    self.db_manager.insertion(plate_text)


    def handle_plate_detected_in_db(self, plate_text, date_time):
            
        # voire si la plaque est détectée dans la base de données
        QtWidgets.QMessageBox.warning(self, "Plaque déjà enregistrée",
            f"La plaque d'immatriculation {plate_text} a été détectée précédemment à la date {date_time}.")
   
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ANPR"))


class IPConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        #self.frame_grabber.alertSignal.connect(self.showAlertMessage)
        self.setWindowTitle("Configure IP Camera")
        self.resize(300, 150)

        layout = QtWidgets.QVBoxLayout(self)

        self.ipLabel = QtWidgets.QLabel("IP Address:")
        self.ipLineEdit = QtWidgets.QLineEdit()
        self.ipLineEdit.setText("192.168.224.186")

        self.portLabel = QtWidgets.QLabel("Port Number:")
        self.portLineEdit = QtWidgets.QLineEdit()
        self.portLineEdit.setText("81")

        self.streamLabel = QtWidgets.QLabel("Stream URL:")
        self.streamLineEdit = QtWidgets.QLineEdit()
        self.streamLineEdit.setText("/stream")

        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.startCamera)  
        buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.ipLabel)
        layout.addWidget(self.ipLineEdit)
        layout.addWidget(self.portLabel)
        layout.addWidget(self.portLineEdit)
        layout.addWidget(self.streamLabel)
        layout.addWidget(self.streamLineEdit)
        layout.addWidget(buttonBox)
        
    

    def startCamera(self):
        ip_address = self.ipLineEdit.text()
        port_number = self.portLineEdit.text()
        stream_url = f"http://{ip_address}:{port_number}{self.streamLineEdit.text()}"
        
        # Instancier FrameGrabber avec la caméra IP configurée
        self.parent().frame_grabber = FrameGrabber(use_ip_camera=True, ip_address=ip_address, port_number=port_number, stream_url=stream_url)
        
        # Connecter le signal d'alerte à une méthode dans l'interface utilisateur principale
        self.parent().frame_grabber.alertSignal.connect(self.parent().showAlertMessage)

        # Fermer la boîte de dialogue
        self.accept()



        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
