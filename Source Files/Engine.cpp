#include "Engine.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <NvInferPlugin.h>
#include<time.h>
#include "buffer.h"
//using namespace buffer;

Engine::Engine()
{
}

Engine::~Engine()
{
}

void Engine::setConfig(EngineUniquePtr<IBuilder>& builder, EngineUniquePtr<INetworkDefinition>& network)
{
}

void Engine::initEngine()
{

}

void Engine::saveEngine(std::string trtSavePath)
{
	std::string engineFile = trtSavePath;
	//std::cout << "uff模型正在序列化为trt模型" << std::endl;
	std::cout << "transform start!!!!!!" << std::endl;
	double   duration;
	clock_t start = clock();
	nvinfer1::IHostMemory* data = mEngine->serialize();
	clock_t end = clock();
	duration = (double)(end - start) / CLOCKS_PER_SEC;
	printf("%f seconds\n", duration);
	//std::cout << "uff模型序列化为trt模型完成" << std::endl;
	std::cout << "transform finish!!!!!!!!" << std::endl;
	std::ofstream file;
	file.open(engineFile, std::ios::binary | std::ios::out);
	//std::cout << "trt模型写入本地" << std::endl;
	std::cout << "saving!!!!!!!!" << std::endl;
	file.write((const char*)data->data(), data->size());

	//std::cout << "trt模型保存地址为：" << engineFile << std::endl;
	std::cout << "save address:" << engineFile << std::endl;
	file.close();
}
