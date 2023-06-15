import sys
from gui import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = my_Form()
    ex.show()
    sys.exit(app.exec_())