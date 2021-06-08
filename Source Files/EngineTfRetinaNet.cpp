#include "EngineTfRetinaNet.h"
#include <assert.h>
#include "common.h"

EngineTfRetinaNet::EngineTfRetinaNet(UffParams params):mParams(params)
{

}

EngineTfRetinaNet::~EngineTfRetinaNet()
{
}

void EngineTfRetinaNet::initEngine()
{
	initLibNvInferPlugins(&mLogger, "");
	auto builder = EngineUniquePtr<IBuilder>(createInferBuilder(mLogger));
	assert(builder != nullptr);
	auto network = EngineUniquePtr<INetworkDefinition>(builder->createNetworkV2(0));
	assert(network != nullptr);
	auto parser = EngineUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
	assert(parser != nullptr);
	std::cout << "================parse1===============" << std::endl;
	assert(mParams.inputTensorNames.size() == mParams.inputDims.size());
	std::cout << "================parse2===============" << std::endl;
	//parse input
	std::cout << "inputTensorNames.size():" << mParams.inputTensorNames.size() << std::endl;
	for (size_t i = 0; i < mParams.inputTensorNames.size(); i++) {
		nvinfer1::Dims dim;
		dim.nbDims = mParams.inputDims[i].size();
		for (int j = 0; j < dim.nbDims; j++) {
			dim.d[j] = mParams.inputDims[i][j];
		}
		std::cout << "input register: " << mParams.inputTensorNames[i].c_str() << std::endl;
		parser->registerInput(mParams.inputTensorNames[i].c_str(), DimsCHW(3, mParams.imgH, mParams.imgW), nvuffparser::UffInputOrder::kNCHW);
	}
	//parse output
	std::cout << "outputTensorNames.size():" << mParams.outputTensorNames.size() << std::endl;
	for (size_t i = 0; i < mParams.outputTensorNames.size(); i++) {
		std::cout << "input register: " << mParams.outputTensorNames[i].c_str() << std::endl;
		parser->registerOutput(mParams.outputTensorNames[i].c_str());
	}
	std::cout << "start paser" << std::endl;
	if (!parser->parse(mParams.uffModelPath.c_str(), *network, nvinfer1::DataType::kFLOAT)) {
		std::cout << "error: parse model failed" << std::endl;
	}

	setConfig(builder, network);
}

void EngineTfRetinaNet::setConfig(EngineUniquePtr<IBuilder>& builder, EngineUniquePtr<INetworkDefinition>& network)
{
	auto config = EngineUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	//使用FP16精度
	if (mParams.FP16)
	{
		if (!builder->platformHasFastFp16())
		{
			std::cout << "platform do not support FP16" << std::endl;
		}
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}

	builder->setMaxBatchSize(mParams.maxBatchSize);
	config->setMaxWorkspaceSize(3_GiB);

	mEngine = (builder->buildEngineWithConfig(*network, *config));
}
