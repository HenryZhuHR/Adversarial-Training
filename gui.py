import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import  QWidget, QApplication,QDesktopWidget


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Robust Model demo')
        self.show()

    def center(self):
        '''move to window to center of screen'''
        screen:QtCore.QRect = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width()-size.width())/2,(screen.height()-size.height())/2)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.center()
    sys.exit(app.exec_())
