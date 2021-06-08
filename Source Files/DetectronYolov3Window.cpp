#include "DetectronYolov3Window.h"
#include <algorithm>
#include <qfiledialog.h>
#include <qmessagebox.h>

DetectronYolov3Window::DetectronYolov3Window()
{
	ui.setupUi(this);
	mEngineGenerator = new EngineGenerator;
	mChildThread = new QThread;

	QObject::connect(ui.pushButton_2, SIGNAL(clicked()), this, SLOT(returnHome()));
	QObject::connect(ui.shiftPushButton, SIGNAL(clicked()), this, SLOT(shiftModel()));
	QObject::connect(ui.pushButton, SIGNAL(clicked()), this, SLOT(chooseYolov3Model()));
	QObject::connect(ui.pushButton_4, SIGNAL(clicked()), this, SLOT(chooseTrtSaveModel()));//选择trt保存路径
	QObject::connect(mEngineGenerator, SIGNAL(statusChanged(int &)), this, SLOT(updateDlg(int &)));
}

DetectronYolov3Window::~DetectronYolov3Window()
{
}


void DetectronYolov3Window::chooseYolov3Model()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("文件对话框"), "D:/trtModelGenerator2.0/model/", tr("模型文件(*wts)"));
	ui.lineEdit->setText(fileName);
	std::string modelPath = fileName.toStdString();
	mParams.weightsPath = modelPath;
	std::cout << "wts path:" << modelPath << std::endl;
}

void DetectronYolov3Window::chooseTrtSaveModel()
{
	QString fileName = QFileDialog::getSaveFileName(this,
		tr("文件对话框"), "D:/trtModelGenerator2.0/model/d2-yolov3.trt", tr("*.trt"));
	ui.lineEdit_5->setText(fileName);
	std::string trtPath = fileName.toStdString();
	mParams.trtSavePath = trtPath;
}

void DetectronYolov3Window::returnHome()
{
	std::cout << "det2 return home" << std::endl;
	emit shiftWindow();
}

void DetectronYolov3Window::shiftModel()
{
	mEngineGenerator->moveToThread(mChildThread);
	mChildThread->start();//启动子线程

	QString trtName = ui.lineEdit_5->text();
	QFileInfo fileInfo(trtName);
	QString fileName = fileInfo.fileName();


	if (ui.lineEdit_2->text().isEmpty() || ui.lineEdit_3->text().isEmpty() || ui.lineEdit_4->text().isEmpty() || ui.lineEdit_6->text().isEmpty() || 
		ui.lineEdit->text().isEmpty() || ui.lineEdit_5->text().isEmpty() || ui.lineEdit_8->text().isEmpty() || ui.lineEdit_9->text().isEmpty())
	{
		QMessageBox msg(this);
		msg.setWindowTitle(QStringLiteral("提示"));
		msg.setText(QStringLiteral("参数输入不完整，请检查参数输入是否正确！"));
		msg.setStandardButtons(QMessageBox::Ok);
		msg.exec();
	}
	else if(trtName != QString("d2-yolov3.trt"))
	{
		QMessageBox msg(this);
		msg.setWindowTitle(QStringLiteral("提示"));
		msg.setText(QStringLiteral("请将trt文件名称设置为：d2-yolov3.trt"));
		msg.setStandardButtons(QMessageBox::Ok);
		msg.exec();
	}
	else
	{
		QString precision = ui.comboBox->currentText();
		if (precision == "FP16")
		{
			mParams.FP16 = true;
		}
		else
		{
			mParams.FP32 = true;
		}

		int srcW = ui.lineEdit_2->text().toInt();
		int srcH = ui.lineEdit_3->text().toInt();
		int targetW = ui.lineEdit_4->text().toInt();
		int targetH = ui.lineEdit_6->text().toInt();
		int max_long_edge = std::max(targetW, targetH);
		int max_short_edge = std::min(targetW, targetH);
		float scale_factor = std::min((float)max_long_edge / std::max(srcW, srcH), (float)max_short_edge / std::min(srcW, srcH));
		int new_w = (int)(srcW * float(scale_factor) + 0.5);
		int new_h = (int)(srcH * float(scale_factor) + 0.5);
		new_w = std::ceil(new_w / 32) * 32;
		new_h = std::ceil(new_h / 32) * 32;
		mParams.imgH = new_h;
		mParams.imgW = new_w;
		mParams.srcW = srcW;
		mParams.srcH = srcH;

		mParams.maxBatchSize = ui.comboBox_2->currentText().toInt();
		mParams.nbCls = ui.lineEdit_8->text().toInt();
		mParams.scoreThr = ui.lineEdit_9->text().toFloat();

		int i = 0;
		mProgressDlg = new QProgressDialog(QStringLiteral("正在生成TensorRT引擎...\n 预计耗时6分钟，请勿关闭此对话框！"), QStringLiteral("取消"), 0, 100, this);
		mProgressDlg->setCancelButton(0);
		QObject::connect(mProgressDlg, SIGNAL(canceled()), this, SLOT(closeProgressDlg()));
		mProgressDlg->setWindowModality(Qt::WindowModal);
		mProgressDlg->setMinimumDuration(0);

		//设置标题，可以不设置默认继承父窗口标题                        
		mProgressDlg->setWindowTitle(QStringLiteral("请稍候"));
		mProgressDlg->setValue(i);
		mEngineDetYolov3 = new EngineDetYolov3(mParams);
		emit startRequest(mEngineDetYolov3, mParams.trtSavePath, i);//发送信号

	}	
}

void DetectronYolov3Window::updateDlg(int & i)
{
	mProgressDlg->setValue(i);
	if (i == 100)
	{
		std::cout << "finish" << std::endl;
		QMessageBox msg(this);
		msg.setWindowTitle(QStringLiteral("提示"));
		msg.setText(QStringLiteral("TensorRT引擎转换完成！"));
		msg.setStandardButtons(QMessageBox::Ok);
		msg.exec();
	}
}