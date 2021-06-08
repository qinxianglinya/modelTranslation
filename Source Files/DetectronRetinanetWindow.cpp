#include "DetectronRetinanetWindow.h"
#include <iostream>
#include <qfiledialog.h>
#include <qmessagebox.h>
#include <algorithm>

DetectronRetinanetWindow::DetectronRetinanetWindow()
{
	ui.setupUi(this);
	mEngineGenerator = new EngineGenerator;
	mChildThread = new QThread;
	QObject::connect(ui.pushButton_2, SIGNAL(clicked()), this, SLOT(returnHome()));
	QObject::connect(ui.pushButton, SIGNAL(clicked()), this, SLOT(chooseDetectronModel()));//ѡ��uffģ��
	QObject::connect(ui.pushButton_4, SIGNAL(clicked()), this, SLOT(chooseTrtSaveModel()));//ѡ��trt����·��
	QObject::connect(ui.shiftPushButton, SIGNAL(clicked()), this, SLOT(shiftModel()));
	qRegisterMetaType<EngineDetRetinaNet>("EngineDetRetinaNet");
	qRegisterMetaType<Engine>("Engine");
	qRegisterMetaType<std::string>("std::string&");
	qRegisterMetaType<int>("int&");
	QObject::connect(this, SIGNAL(startRequest(Engine*, std::string &, int&)), mEngineGenerator, SLOT(startGenerate(Engine*, std::string &, int &)));
	QObject::connect(mEngineGenerator, SIGNAL(statusChanged(int &)), this, SLOT(updateDlg(int &)));
}

DetectronRetinanetWindow::~DetectronRetinanetWindow()
{
}

void DetectronRetinanetWindow::chooseDetectronModel()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("�ļ��Ի���"), "D:/trtModelGenerator2.0/model/", tr("ģ���ļ�(*wts)"));
	ui.lineEdit->setText(fileName);
	std::string modelPath = fileName.toStdString();
	mDetParams.weightsPath = modelPath;
	std::cout << "wts path:" << modelPath << std::endl;
}

void DetectronRetinanetWindow::chooseTrtSaveModel()
{
	QString fileName = QFileDialog::getSaveFileName(this,
		tr("�ļ��Ի���"), "D:/trtModelGenerator2.0/model/d2-retinanet.trt", tr("*.trt"));
	ui.lineEdit_5->setText(fileName);
	std::string trtPath = fileName.toStdString();
	mDetParams.trtSavePath = trtPath;
}

void DetectronRetinanetWindow::shiftModel()
{
	mEngineGenerator->moveToThread(mChildThread);
	mChildThread->start();//�������߳�


	//mDetParams.srcW = imgW;
	//mDetParams.srcH = imgH;

	//int resizedShortEdg = rShortEdg.toInt();

	//int maxSize = 1333;
	//int shortLen = std::min(imgW, imgH);
	//float scale = float(resizedShortEdg) / shortLen;
	//std::cout << "scale:" << scale << std::endl;
	//float newH, newW;
	//if (imgH < imgW)
	//{
	//	newH = resizedShortEdg;
	//	newW = scale * imgW;
	//}
	//else
	//{
	//	newH = scale * imgH;
	//	newW = resizedShortEdg;
	//}
	/*if (std::max(newH, newW) > maxSize)
	{
		scale = float(maxSize) / std::max(newH, newW);
		newH = newH * scale;
		newW = newW * scale;
	}
	int targetW, targetH;
	targetW = int(newW + 0.5);
	targetH = int(newH + 0.5);

	if ((targetH % 32) != 0)
	{
		targetH = (1 + int(targetH / 32)) * 32;
	}
	if ((targetW % 32) != 0)
	{
		targetW = (1 + int(targetW / 32)) * 32;
	}*/

	QString trtName = ui.lineEdit_5->text();
	QFileInfo fileInfo(trtName);
	QString fileName = fileInfo.fileName();

	QString width = ui.lineEdit_2->text();//resize���
	QString height = ui.lineEdit_3->text();//resize���
	int imgW = width.toInt();
	int imgH = height.toInt();

	int minLayer = ui.comboBox->currentText().toInt();
	int maxLayer = ui.comboBox_4->currentText().toInt();
	if (ui.lineEdit_2->text().isEmpty() || ui.lineEdit_3->text().isEmpty() || ui.lineEdit->text().isEmpty() || ui.lineEdit_5->text().isEmpty() || ui.lineEdit_6->text().isEmpty()|| ui.lineEdit_7->text().isEmpty())
	{
		QMessageBox msg(this);
		msg.setWindowTitle(QStringLiteral("��ʾ"));
		msg.setText(QStringLiteral("�������벻������������������Ƿ���ȷ��"));
		msg.setStandardButtons(QMessageBox::Ok);
		msg.exec();
	}
	else if (fileName != QString("d2-retinanet.trt"))
	{
		QMessageBox msg(this);
		msg.setWindowTitle(QStringLiteral("��ʾ"));
		msg.setText(QStringLiteral("�뽫trt�ļ���������Ϊ��d2-retinanet.trt"));
		msg.setStandardButtons(QMessageBox::Ok);
		msg.exec();
	}
	else if (maxLayer <= minLayer)
	{
		QMessageBox msg(this);
		msg.setWindowTitle(QStringLiteral("��ʾ"));
		msg.setText(QStringLiteral("��������ѡ������"));
		msg.setStandardButtons(QMessageBox::Ok);
		msg.exec();
	}
	else if ((imgW % 32 != 0) || (imgH % 32 != 0 ))
	{
		QMessageBox msg(this);
		msg.setWindowTitle(QStringLiteral("��ʾ"));
		msg.setText(QStringLiteral("resize��Ŀ��߱���Ϊ32���������������µ���������"));
		msg.setStandardButtons(QMessageBox::Ok);
		msg.exec();
	}
	else
	{
		//QString rShortEdg = ui.lineEdit_4->text();
		QString precision = ui.comboBox_3->currentText();
		QString rScoreThr = ui.lineEdit_7->text();
		if (precision == "FP16")
		{
			mDetParams.FP16 = true;
		}
		else
		{
			mDetParams.FP32 = true;
		}
		int i = 0;

		mProgressDlg = new QProgressDialog(QStringLiteral("��������TensorRT����...\n Ԥ�ƺ�ʱ6���ӣ�����رմ˶Ի���"), QStringLiteral("ȡ��"), 0, 100, this);
		mProgressDlg->setCancelButton(0);
		QObject::connect(mProgressDlg, SIGNAL(canceled()), this, SLOT(closeProgressDlg()));
		mProgressDlg->setWindowModality(Qt::WindowModal);
		mProgressDlg->setMinimumDuration(0);

		//���ñ��⣬���Բ�����Ĭ�ϼ̳и����ڱ���                        
		mProgressDlg->setWindowTitle(QStringLiteral("���Ժ�"));
		mProgressDlg->setValue(i);

		mDetParams.nbCls = ui.lineEdit_6->text().toInt();
		mDetParams.weightsPath = ui.lineEdit->text().toStdString();
		mDetParams.trtSavePath = ui.lineEdit_5->text().toStdString();
		mDetParams.imgW = imgW;
		mDetParams.imgH = imgH;
		mDetParams.maxBatchSize = ui.comboBox_2->currentText().toInt();
		mDetParams.minLayer = minLayer;
		mDetParams.maxLayer = maxLayer;
		mDetParams.scoreThr = rScoreThr.toFloat();
		mDetParams.keepTopK = 900;
		mEngineDetRetinanet = new EngineDetRetinaNet(mDetParams);
		emit startRequest(mEngineDetRetinanet, mDetParams.trtSavePath, i);//�����ź�
	}
}

void DetectronRetinanetWindow::updateDlg(int & i)
{
	mProgressDlg->setValue(i);
	if (i == 100)
	{
		std::cout << "finish" << std::endl;
		QMessageBox msg(this);
		msg.setWindowTitle(QStringLiteral("��ʾ"));
		msg.setText(QStringLiteral("TensorRT����ת����ɣ�"));
		msg.setStandardButtons(QMessageBox::Ok);
		msg.exec();
	}
}


void DetectronRetinanetWindow::returnHome()
{
	emit shiftWindow();
	std::cout << "det2 return home" << std::endl;
}


