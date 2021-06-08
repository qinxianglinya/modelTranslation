#pragma once
#include "Engine.h"
#include <map>
#include "struct.h"
using namespace myPlugin;

class EngineDetYolov3 : public Engine
{
public:
	EngineDetYolov3(Yolov3Params params);

	~EngineDetYolov3();

	void initEngine();

	void setConfig(nvinfer1::IBuilder*, nvinfer1::INetworkDefinition* network);

	std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file);

	IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor & input, const std::string& lname, float eps);	

private:
	ILayer* convBnLeaky(INetworkDefinition *network, ITensor& input, int outch, int ksize, int s, int p, const std::string& commonStr, const enum NetworkPart& networkPart);

	ILayer* resBlock(INetworkDefinition* network, ITensor& input, int outch1, int ksize1, int s1, int p1, int ksize2, int s2, int p2, const std::string& commonStr);

	ILayer* convResBlock1(INetworkDefinition *network, ITensor& input);
	ILayer* convResBlock2(INetworkDefinition *network, ITensor& input);
	ILayer* convResBlock3(INetworkDefinition *network, ITensor& input);
	ILayer* convResBlock4(INetworkDefinition *network, ITensor& input);
	ILayer* convResBlock5(INetworkDefinition *network, ITensor& input);

private:	
	ILayer* DetectionBlock1(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network, ILayer* input);
	ILayer* DetectionBlock2(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network, ILayer* input);
	ILayer* DetectionBlock3(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network, ILayer* input);

private:

	void constructNetwork(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network);

	//构建backbone
	void constructDarknet(INetworkDefinition *network, ITensor& input, std::vector<ILayer*>& darknetOut);

	//构建neck
	void constructNeck(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network, const std::vector<ILayer*>& input, std::vector<ILayer*>& output);

	//构建head
	void constructHead(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network, const std::vector<ILayer*>& input, std::vector<ILayer*>& output);

	//anchor plugin
	ILayer* constructAnchor(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network);

	//nms plugin
	ILayer* constructNms(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network, std::vector<ILayer*>& input, ILayer* anchor);

private:
	Yolov3Params mParams;
	std::map<std::string, nvinfer1::Weights> mWeightMap;
	nvinfer1::INetworkDefinition* m_Network;
	nvinfer1::IBuilder* m_Builder;
	nvinfer1::IBuilderConfig* m_Config;

};

