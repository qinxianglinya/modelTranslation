#include "QtGuiApplication.h"
#include "qpushbutton.h"

QtGuiApplication::QtGuiApplication(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	mTfRetinanetWindow = new TfRetinanetWindow();
	mDetectronRetinanetWindow = new DetectronRetinanetWindow();
	mDetectronYolov3Window = new DetectronYolov3Window();
	QObject::connect(ui.pushButton, SIGNAL(clicked()), this, SLOT(onPushButtonClicked()));
	QObject::connect(mTfRetinanetWindow, SIGNAL(shiftWindow()), this, SLOT(returnHomeFromTf()));
	QObject::connect(mDetectronRetinanetWindow, SIGNAL(shiftWindow()), this, SLOT(returnHomeFromDet()));
	QObject::connect(mDetectronYolov3Window, SIGNAL(shiftWindow()), this, SLOT(returnHomeFromYolov3()));

	//ui.pushButton
}

void QtGuiApplication::onPushButtonClicked()
{
	if (ui.comboBox->currentText() == QString("Tf-RetinaNet"))
	{
		this->hide();
		mTfRetinanetWindow->show();
	}
	if (ui.comboBox->currentText() == QString("D2-RetinaNet"))
	{
		this->hide();
		mDetectronRetinanetWindow->show();
	}
	if (ui.comboBox->currentText() == QString("D2-Yolov3"))
	{
		this->hide();
		mDetectronYolov3Window->show();
	}
}

void QtGuiApplication::returnHomeFromTf()
{
	this->show();
	mTfRetinanetWindow->hide();
}

void QtGuiApplication::returnHomeFromDet()
{
	this->show();
	mDetectronRetinanetWindow->hide();
}

void QtGuiApplication::returnHomeFromYolov3()
{
	this->show();
	mDetectronYolov3Window->hide();
}