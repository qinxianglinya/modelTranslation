#include "EngineGenerator.h"




void EngineGenerator::startGenerate(Engine* engine, std::string& trtSavePath, int& i)
{
	std::cout << "engine generator start" << std::endl;
	i = 30;
	emit statusChanged(i);
	engine->initEngine();
	i = 80;
	emit statusChanged(i);
	engine->saveEngine(trtSavePath);
	i = 100;
	emit statusChanged(i);
}
