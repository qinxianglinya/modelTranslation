#pragma once
#include <string>
#include <assert.h>
#include "struct.h"
#include "NvUffParser.h"
#include "NvInferRuntime.h"
#include <iostream>
#include <NvInferPlugin.h>
#include "resizeNearestPlugin.h"

using namespace nvinfer1;
template <typename T>
using EngineUniquePtr = std::unique_ptr<T, InferDeleter>;

class Engine
{
public:
	Engine();
	virtual ~Engine();

	void setConfig(EngineUniquePtr<IBuilder> &builder, EngineUniquePtr<INetworkDefinition> &network);
	virtual void initEngine();
	//void initEngine();
	void saveEngine(std::string trtSavePath);

protected:
	//std::shared_ptr<ICudaEngine> mEngine;
	ICudaEngine* mEngine;
	TrtLogger mLogger;
};

