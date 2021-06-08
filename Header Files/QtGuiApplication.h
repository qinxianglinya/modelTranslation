#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtGuiApplication.h"
#include "qlabel.h"
#include <iostream>
#include "TfRetinanetWindow.h"
#include "DetectronRetinanetWindow.h"
#include "DetectronYolov3Window.h"
class QtGuiApplication : public QMainWindow
{
	Q_OBJECT

public:
	QtGuiApplication(QWidget *parent = Q_NULLPTR);

private:
	Ui::QtGuiApplicationClass ui;

private slots:
	void onPushButtonClicked();
	void returnHomeFromTf();
	void returnHomeFromDet();
	void returnHomeFromYolov3();

private:
	QWidget* mTfRetinanetWindow;
	QWidget* mDetectronRetinanetWindow;
	QWidget* mDetectronYolov3Window;
	QLabel* mModelChooseLabel;
};
