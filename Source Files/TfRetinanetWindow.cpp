#include "TfRetinanetWindow.h"
#include <qfiledialog.h>

TfRetinanetWindow::TfRetinanetWindow()
{
	ui.setupUi(this);
	mEngineGenerator = new EngineGenerator;
	mChildThread = new QThread;
	//mHomeWindow = new QtGuiApplication();
	QObject::connect(ui.pushButton_2, SIGNAL(clicked()), this, SLOT(returnHome()));
	QObject::connect(ui.pushButton, SIGNAL(clicked()), this, SLOT(chooseUffModel()));//选择uff模型
	QObject::connect(ui.pushButton_4, SIGNAL(clicked()), this, SLOT(chooseTrtSaveModel()));//选择trt保存路径
	QObject::connect(ui.shiftPushButton, SIGNAL(clicked()), this, SLOT(shiftModel()));
	qRegisterMetaType<Engine>("Engine");
	QObject::connect(this, SIGNAL(startRequest(Engine*, std::string &, int&)), mEngineGenerator, SLOT(startGenerate(Engine*, std::string &, int &)));
	QObject::connect(mEngineGenerator, SIGNAL(statusChanged(int &)), this, SLOT(updateDlg(int &)));

}

TfRetinanetWindow::~TfRetinanetWindow()
{
	delete mEngineTfRetinanet;
	delete mEngineGenerator;
	delete mChildThread;
	delete mProgressDlg;
}

void TfRetinanetWindow::shiftModel()
{
	mEngineGenerator->moveToThread(mChildThread);
	mChildThread->start();//启动子线程

	QString width = ui.lineEdit_2->text();
	QString height = ui.lineEdit_3->text();
	QString precision = ui.comboBox_3->currentText();
	if (precision == "FP16")
	{
		mUffParams.FP16 = true;
	}
	else
	{
		mUffParams.FP32 = true;
	}
	QString trtName = ui.lineEdit_5->text();
	QFileInfo fileInfo(trtName);
	QString fileName = fileInfo.fileName();
	int imgW = width.toInt();
	int imgH = height.toInt();
	if (width.isEmpty() || height.isEmpty() || ui.lineEdit->text().isEmpty() || ui.lineEdit_5->text().isEmpty())
	{
		QMessageBox msg(this);
		msg.setWindowTitle(QStringLiteral("提示"));
		msg.setText(QStringLiteral("参数输入不完整，请检查参数输入是否正确！"));
		msg.setStandardButtons(QMessageBox::Ok);
		msg.exec();
	}
	else if (fileName != QString("tf-retinanet.trt"))
	{
		QMessageBox msg(this);
		msg.setWindowTitle(QStringLiteral("提示"));
		msg.setText(QStringLiteral("请将trt文件名称设置为：tf-retinanet.trt"));
		msg.setStandardButtons(QMessageBox::Ok);
		msg.exec();
	}
	else
	{
		int i = 0;

		mProgressDlg = new QProgressDialog(QStringLiteral("正在生成TensorRT引擎...\n 预计耗时6分钟，请勿关闭此对话框！"), QStringLiteral("取消"), 0, 100, this);
		mProgressDlg->setCancelButton(0);
		QObject::connect(mProgressDlg, SIGNAL(canceled()), this, SLOT(closeProgressDlg()));
		mProgressDlg->setWindowModality(Qt::WindowModal);
		mProgressDlg->setMinimumDuration(0);

		//设置标题，可以不设置默认继承父窗口标题                        
		mProgressDlg->setWindowTitle(QStringLiteral("请稍候"));
		mProgressDlg->setValue(i);

		mUffParams.inputDims = { {3} };
		mUffParams.inputTensorNames = { "Input" };
		mUffParams.outputTensorNames = { "NMS" };
		mUffParams.uffModelPath = ui.lineEdit->text().toStdString();
		mUffParams.trtSavePath = ui.lineEdit_5->text().toStdString();
		mUffParams.imgW = imgW;
		mUffParams.imgH = imgH;
		mUffParams.maxBatchSize = ui.comboBox_2->currentText().toInt();
		//std::cout << "batch:" << ui.comboBox_2->currentText().toInt() << std::endl;
		mEngineTfRetinanet = new EngineTfRetinaNet(mUffParams);
		emit startRequest(mEngineTfRetinanet, mUffParams.trtSavePath, i);//发送信号
	}
}

void TfRetinanetWindow::chooseUffModel()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("文件对话框"), "D:/trtModelGenerator2.0/model/", tr("模型文件(*uff *onnx *weights *pth)"));
	ui.lineEdit->setText(fileName);
	std::string uffPath = fileName.toStdString();
	mUffParams.uffModelPath = uffPath;
	std::cout << "uff path:" << uffPath << std::endl;
}

void TfRetinanetWindow::chooseTrtSaveModel()
{
	QString fileName = QFileDialog::getSaveFileName(this,
		tr("文件对话框"), "D:/trtModelGenerator2.0/model/tf-retinanet.trt", tr("*.trt"));
	ui.lineEdit_5->setText(fileName);
	std::string trtPath = fileName.toStdString();
	mUffParams.trtSavePath = trtPath;
}

void TfRetinanetWindow::updateDlg(int &i)
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

void TfRetinanetWindow::returnHome()
{
	emit shiftWindow();
}


