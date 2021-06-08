#pragma once
#include "Engine.h"
#include <map>

#include <QMetaType>

class EngineDetRetinaNet :
	public Engine
{
public:
	EngineDetRetinaNet();
	EngineDetRetinaNet(DetRetinaNetParams params);
	~EngineDetRetinaNet();

	void initEngine();
	void setConfig(nvinfer1::IBuilder*, nvinfer1::INetworkDefinition* network);

	std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file);

	bool constructNetwork(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network);

	bool constructResnet(ILayer* input, nvinfer1::INetworkDefinition* network, int minLayer, int maxLayer, std::vector<ILayer*>&output);

	bool constructFpn(std::vector<ILayer*>&input, nvinfer1::INetworkDefinition* network, std::vector<ILayer*>&output, int minLayer, int maxLayer);

	bool constructHead(std::vector<ILayer*>&inFeature, nvinfer1::INetworkDefinition* network,
		std::vector<ILayer*>&classHead, std::vector<ILayer*>&locHead, int minLayer, int maxLayer);

	IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
		std::map<std::string, Weights>& weightMap,
		ITensor& input,
		std::string lname,
		float eps);

	IActivationLayer* bottleneck(nvinfer1::INetworkDefinition* network,
		std::map<std::string, Weights>& weightMap,
		ITensor& input,
		int inch,
		int outch,
		int stride,
		std::string lname);
	IActivationLayer* bottleneck_test(nvinfer1::INetworkDefinition* network,
		std::map<std::string, Weights>& weightMap,
		ITensor& input,
		int inch,
		int outch,
		int stride,
		std::string lname);
	IConvolutionLayer *addHead(nvinfer1::INetworkDefinition* network,
		std::map<std::string, Weights>& weightMap,
		ITensor& input,
		const std::string &lname,
		const int& numClass,
		const int& anchorNum,
		const int& index);

private:
	DetRetinaNetParams mParams;
	std::map<std::string, nvinfer1::Weights> mWeightMap;
	nvinfer1::INetworkDefinition* m_Network;
	nvinfer1::IBuilder* m_Builder;
	nvinfer1::IBuilderConfig* m_Config;
};

