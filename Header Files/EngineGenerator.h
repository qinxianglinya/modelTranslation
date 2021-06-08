#pragma once
#include <qobject.h>
#include "struct.h"
#include "Engine.h"
#include "EngineDetRetinaNet.h"

class EngineGenerator :
	public QObject
{
	Q_OBJECT
public:
	explicit EngineGenerator(QObject *parent = 0) {};

signals:
	void statusChanged(int &i);

public slots:
	void startGenerate(Engine* engine, std::string& trtSavePath, int &);
};

