#pragma once
#include <qwidget.h>
#include "ui_TfRetinanet.h"
#include "struct.h"	
#include "EngineGenerator.h"
#include <qthread.h>
#include <qmessagebox.h>
#include <qprogressdialog.h>
#include "EngineTfRetinaNet.h"

class TfRetinanetWindow:public QWidget
{
	Q_OBJECT
public:
	TfRetinanetWindow();
	~TfRetinanetWindow();
private:
	Ui::FormTfRetina ui;
	//QtGuiApplication* mHomeWindow;

private slots:
	void returnHome();
	void shiftModel();
	void chooseUffModel();
	void chooseTrtSaveModel();
	void updateDlg(int&);

signals:
	void shiftWindow();
signals:
	void startRequest(Engine* engine, std::string& trtSavePath, int& i);

private:
	EngineTfRetinaNet* mEngineTfRetinanet;
	EngineGenerator *mEngineGenerator;
	UffParams mUffParams;
	QThread *mChildThread;
	QProgressDialog* mProgressDlg;
};

