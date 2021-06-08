#pragma once
#include <qwidget.h>
#include "ui_DetectronYolov3.h"
#include "EngineGenerator.h"
#include <qthread.h>
#include "EngineDetYolov3.h"
#include <qprogressdialog.h>

class DetectronYolov3Window :
	public QWidget
{
	Q_OBJECT

public:
	DetectronYolov3Window();
	~DetectronYolov3Window();


private slots:
	void returnHome();
	void shiftModel();
	void chooseYolov3Model();
	void chooseTrtSaveModel();
	void updateDlg(int&);

signals:
	void shiftWindow();

signals:
	void startRequest(Engine* engine, std::string& trtSavePath, int& i);

private:
	Ui_Yolov3_Form ui;
	EngineDetYolov3* mEngineDetYolov3;
	EngineGenerator* mEngineGenerator;
	Yolov3Params mParams;
	QThread* mChildThread;
	QProgressDialog* mProgressDlg;
};

