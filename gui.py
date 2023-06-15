# -*- coding: utf-8 -*-

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from form_control import Control
import pyttsx3  


class my_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.control = Control()
        self.tts = pyttsx3.init()

    def setupUi(self):
        self.setWindowTitle("Heritage Finder")
        self.pushButton = QPushButton(self)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(10, 10, 101, 31))
        self.pushButton_2 = QPushButton(self)
        self.pushButton_2.setObjectName(u"sound")
        self.pushButton_2.setGeometry(QRect(131, 10, 101, 31))
        self.groupBox = QGroupBox(self)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(10, 50, 381, 371))
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(160, 170, 100, 30))
        self.groupBox_2 = QGroupBox(self)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(400, 50, 331, 51))
        self.plainTextEdit_2 = QPlainTextEdit(self.groupBox_2)
        self.plainTextEdit_2.setObjectName(u"plainTextEdit_2")
        self.plainTextEdit_2.setGeometry(QRect(0, 20, 331, 31))
        self.plainTextEdit_2.setFrameShape(QFrame.NoFrame)
        self.groupBox_3 = QGroupBox(self)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(400, 100, 331, 321))
        self.plainTextEdit = QPlainTextEdit(self.groupBox_3)
        self.plainTextEdit.setObjectName(u"plainTextEdit")
        self.plainTextEdit.setGeometry(QRect(0, 20, 331, 301))
        self.plainTextEdit.setAutoFillBackground(False)
        self.plainTextEdit.setFrameShape(QFrame.NoFrame)
        self.plainTextEdit.setTabChangesFocus(False)
        self.plainTextEdit.setReadOnly(True)
        self.statusbar = QStatusBar()
        self.statusbar.setObjectName(u"statusbar")
        self.groupBox_4 = QGroupBox(self)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(400, 370, 331, 51))
        self.radioButton = QRadioButton(self.groupBox_4)
        self.radioButton.setObjectName(u"radioButton")
        self.radioButton.setGeometry(QRect(10, 20, 89, 16))
        self.radioButton.setChecked(True)
        self.radioButton_2 = QRadioButton(self.groupBox_4)
        self.radioButton_2.setObjectName(u"radioButton_2")
        self.radioButton_2.setGeometry(QRect(80, 20, 89, 16))
        self.radioButton_3 = QRadioButton(self.groupBox_4)
        self.radioButton_3.setObjectName(u"radioButton_3")
        self.radioButton_3.setGeometry(QRect(150, 20, 89, 16))
        self.radioButton_4 = QRadioButton(self.groupBox_4)
        self.radioButton_4.setObjectName(u"radioButton_4")
        self.radioButton_4.setGeometry(QRect(230, 20, 89, 16))

        self.retranslateUi()
        self.set_event()
        
    # setupUi

    def retranslateUi(self):
        self.setWindowTitle("Heritage Finder")
        self.pushButton.setText(u"file upload")
        self.pushButton_2.setText(u"sound")
        self.groupBox.setTitle(u"Image")
        self.label.setText(u"NoImage")
        self.groupBox_2.setTitle(u"Heritage")
        self.groupBox_3.setTitle(u"Description")
        self.groupBox_4.setTitle(u"Language")
        self.radioButton.setText(u"Korean")
        self.radioButton_2.setText(u"English")
        self.radioButton_3.setText(u"Japanes")
        self.radioButton_4.setText(u"Chines")
        
    # retranslateUi

    def set_event(self):
        self.pushButton.clicked.connect(self.button_1_clicked)
        self.pushButton_2.clicked.connect(self.button_2_clicked)
        self.radioButton.toggled.connect(self.set_desc)
        self.radioButton_2.toggled.connect(self.set_desc)
        self.radioButton_3.toggled.connect(self.set_desc)
        self.radioButton_3.toggled.connect(self.set_desc)

    def button_2_clicked(self):
        self.tts.say(self.plainTextEdit_2.toPlainText())
        self.tts.runAndWait()
        self.tts.say(self.plainTextEdit.toPlainText())
        self.tts.runAndWait()

    def button_1_clicked(self):
        fname = QFileDialog.getOpenFileName(self)
        if fname[0]== "": return

        self.control.image_path = fname[0]
        print(fname[0])
        self.label.resize(371, 371)
        pixmap = QPixmap()
        pixmap.load(fname[0])
        self.label.setPixmap(QPixmap(pixmap.scaled(371,371, Qt.KeepAspectRatio)))


        self.control.evaluate_model()    
        self.set_desc()


    def get_filename_byValue(self):
        if self.control.desc == "": return ""

        if self.radioButton.isChecked():
            return self.control.title_ko, self.control.desc + ".txt"
        
        elif self.radioButton_2.isChecked():
            return self.control.title_en, self.control.desc + "_en" + ".txt"
        
        elif self.radioButton_3.isChecked():
            return self.control.title_ja, self.control.desc + "_ja" + ".txt"
        
        else:
            return self.control.title_cn, self.control.desc + "_cn" + ".txt"
            

    def get_text(self):
        title, file = self.get_filename_byValue()
        if file == "": return ""
        with open(file, encoding="UTF8", mode="r") as f:
            text = f.read()

            return title, text

    def set_desc(self):
        title, text = self.get_text()
        
        self.plainTextEdit.setPlainText(text)
        self.plainTextEdit_2.setPlainText(title)
