#include "QtGuiApplication.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QtGuiApplication w;
	w.setWindowTitle(QStringLiteral("TensorRT����������"));
	//qRegisterMetaType<EngineGenerator>("EngineGenerator");
	w.show();
	return a.exec();
}
