#pragma once
#include <qwidget.h>
#include "ui_DetectronRetinanet.h"
#include "struct.h"
#include "EngineGenerator.h"
#include <qthread.h>
#include "EngineDetRetinaNet.h"
#include <qprogressdialog.h>

class DetectronRetinanetWindow :  public QWidget
{
	Q_OBJECT

public:
	DetectronRetinanetWindow();
	~DetectronRetinanetWindow();

private slots:
	void returnHome();
	void chooseDetectronModel();
	void chooseTrtSaveModel();
	void shiftModel();
	void updateDlg(int&);

signals:
	void shiftWindow();
signals:
	void startRequest(Engine* engine, std::string& trtSavePath, int& i);

private:
	Ui::FormDetRetina ui;
	DetRetinaNetParams mDetParams;
	EngineGenerator* mEngineGenerator;
	QThread* mChildThread;
	EngineDetRetinaNet* mEngineDetRetinanet;
	QProgressDialog* mProgressDlg;
};

