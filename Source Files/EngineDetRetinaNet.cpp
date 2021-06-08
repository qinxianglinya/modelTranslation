#include "EngineDetRetinaNet.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "anchorGeneratorTorch.h"
#include "nmsPluginTorch.h"
#include "common.h"
#include "buffer.h"
#include <exception>
using namespace buffer;

EngineDetRetinaNet::EngineDetRetinaNet()
{
}

EngineDetRetinaNet::EngineDetRetinaNet(DetRetinaNetParams  params) :mParams(params)
{
	//qRegisterMetaType<EngineDetRetinaNet>("EngineDetRetinaNet");
	std::cout << "-----------print det params----------------" << std::endl;
	std::cout << "img srcw:" << params.srcW << std::endl;
	std::cout << "img srch:" << params.srcH << std::endl;
	std::cout << "img targetw:" << params.imgW << std::endl;
	std::cout << "img targeth:" << params.imgH << std::endl;
	std::cout << "fp32:" << params.FP32 << std::endl;
	std::cout << "maxBatchSize:" << params.maxBatchSize << std::endl;
	std::cout << "trt save path:" << params.trtSavePath << std::endl;
	std::cout << "min layer:" << params.minLayer << std::endl;
	std::cout << "max layer:" << params.maxLayer << std::endl;
	std::cout << "nbPrior:" << params.nbPrior << std::endl;
	std::cout << "nbcls:" << params.nbCls << std::endl;
	std::cout << "topk:" << params.topK << std::endl;
	std::cout << "keepTopK:" << params.keepTopK << std::endl;
}

EngineDetRetinaNet::~EngineDetRetinaNet()
{
}

void EngineDetRetinaNet::initEngine()
{
	mWeightMap = loadWeights(mParams.weightsPath);

	initLibNvInferPlugins(&mLogger, "");
	m_Builder = (nvinfer1::createInferBuilder(mLogger));
	//auto builder = EngineUniquePtr<IBuilder>(createInferBuilder(mLogger));
	assert(m_Builder != nullptr);
	m_Network = m_Builder->createNetwork();
	assert(m_Network != nullptr);
	constructNetwork(m_Builder, m_Network);
	std::cout << "construct network finished!!!!!!!!" << std::endl;
	m_Config = m_Builder->createBuilderConfig();
	assert(m_Config);
	setConfig(m_Builder, m_Network);
}

void EngineDetRetinaNet::setConfig(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network)
{
	//auto config = builder->createBuilderConfig();
   //使用FP16精度
	if (mParams.FP16)
	{
		if (!builder->platformHasFastFp16())
		{
			std::cout << "platform does not support FP16" << std::endl;
		}
		m_Config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}

	builder->setMaxBatchSize(mParams.maxBatchSize);
	m_Config->setMaxWorkspaceSize(2_GiB);


	mEngine = builder->buildEngineWithConfig(*network, *m_Config);

}

std::map<std::string, nvinfer1::Weights> EngineDetRetinaNet::loadWeights(const std::string & file)
{
	std::cout << file << std::endl;
	std::map<std::string, nvinfer1::Weights> weights;
	std::cout << "Opening `" << file << "` TensorRT weights..." << std::endl;
	// Open weights file
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file.");

	// Read number of weight blobs
	int32_t count;
	input >> count;
	assert(count > 0 && "Invalid weight map file.");

	while (count--) {
		nvinfer1::Weights wt{ DataType::kFLOAT, nullptr, 0 };
		uint32_t size;

		// Read name and type of blob
		std::string name;
		int type;
		input >> name >> std::dec >> size;

		wt.type = DataType::kFLOAT;

		// Load blob
		uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));

		for (uint32_t x = 0, y = size; x < y; ++x)
		{
			input >> std::hex >> val[x];
		}

		wt.values = val;
		wt.count = size;
		weights[name] = wt;
	}

	return weights;
}

bool EngineDetRetinaNet::constructNetwork(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network)
{
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	ITensor* data = network->addInput(
		"data", DataType::kFLOAT, Dims3{ 3, mParams.imgH, mParams.imgW });
	assert(data);

	int minLayer = mParams.minLayer;
	int maxLayer = mParams.maxLayer;

	//stem:7*7 conv
	//该处固定
	IConvolutionLayer *conv1 = network->addConvolution(*data, 64, DimsHW{ 7,7 }, mWeightMap["backbone.bottom_up.stem.conv1.weight"], emptywts);
	conv1->setStride(DimsHW{ 2,2 });
	conv1->setPadding(DimsHW{ 3,3 });
	IScaleLayer* bn1 = addBatchNorm2d(network, mWeightMap, *conv1->getOutput(0), "backbone.bottom_up.stem.conv1.norm", 1e-5);
	IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	assert(relu1);
	IPoolingLayer* pool1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });
	assert(pool1);
	pool1->setStride(DimsHW{ 2, 2 });
	pool1->setPadding(DimsHW{ 1, 1 });
	//构建resnet50
	std::vector<ILayer*> resnetOutput;
	constructResnet(pool1, network, minLayer, maxLayer, resnetOutput);
	//构建FPN
	std::vector<ILayer*> fpnOutput;
	constructFpn(resnetOutput, network, fpnOutput, minLayer, maxLayer);

	//resnetOutput[0]->getOutput(0)->setName("Loc");
	//resnetOutput[1]->getOutput(0)->setName("Loc1");
	//resnetOutput[2]->getOutput(0)->setName("Loc2");
	//network->markOutput(*resnetOutput[0]->getOutput(0));
	//network->markOutput(*resnetOutput[1]->getOutput(0));
	//network->markOutput(*resnetOutput[2]->getOutput(0));


	/*

	************************************************************************************
										 HEAD
	************************************************************************************
	*/
	//test params
	const int nbCls = mParams.nbCls;
	const int nbPrior = mParams.nbPrior;
	const int nbLayer = maxLayer - minLayer + 1;
	//构建head
	std::vector<ILayer*> clsHead;
	std::vector<ILayer*> locHead;
	constructHead(fpnOutput, network, clsHead, locHead, nbCls, nbPrior);

	///*
	//************************************************************************************
	//								 ANCHOR-GENERATOR
	//************************************************************************************
	//*/
	//prepare anchorGenerator params
	std::vector<int> baseSize;
	std::vector<int> layerIndex;
	float anchorScale = 4.0;
	for (int i = minLayer; i <= maxLayer; i++)
	{
		baseSize.push_back(anchorScale * pow(2, i));
		std::cout << baseSize.back() << std::endl;
		layerIndex.push_back(i);
	}

	std::vector<AnchorParamsTorch> params(maxLayer - minLayer + 1);
	for (int i = 0; i < (maxLayer - minLayer + 1); i++)
	{
		params[i].nStride = pow(2, layerIndex[i]);
		params[i].nOffset = 0;
		params[i].nSize[0] = baseSize[i];
		params[i].nSize[1] = baseSize[i] * pow(2, 1.0 / 3);
		params[i].nSize[2] = baseSize[i] * pow(2, 2.0 / 3);
		for (int j = 0; j < 3; j++)
		{
			float area = pow(params[i].nSize[j], 2);
			for (int k = 0; k < 3; k++)
			{
				Coordinate coordinate;
				float w = sqrt(area / params[i].fAspectRatio[k]);
				float h = params[i].fAspectRatio[k] * w;
				coordinate.x0 = -(w / 2.0);
				coordinate.y0 = -(h / 2.0);
				coordinate.x1 = w / 2.0;
				coordinate.y1 = h / 2.0;
				params[i].fBaseAnchor[3 * j + k] = coordinate;
			}
		}
		FeatureSize fs;
		int d = pow(2, layerIndex[i]);
		if (mParams.imgW % d != 0)
		{
			fs.nWidth = int(mParams.imgW / d) + 1;
		}
		else
		{
			fs.nWidth = mParams.imgW / d;
		}

		if (mParams.imgH % d != 0)
		{
			fs.nHeight = int(mParams.imgH / d) + 1;
		}
		else
		{
			fs.nHeight = mParams.imgH / d;
		}
		params[i].featureSize = fs;
		std::cout << "f s:" << params[i].featureSize.nHeight << " " << params[i].featureSize.nWidth << std::endl;
	}

	detectronPlugin::AnchorGeneratorTorch anchorGenerator(params.data(), maxLayer - minLayer + 1);
	auto anchor = network->addPluginV2(nullptr, 0, anchorGenerator);


	//从前端获取

	DetectionOutputParametersTorch detectionParams;
	detectionParams.nbCls = nbCls;
	detectionParams.topK = mParams.topK;
	detectionParams.keepTopK = mParams.keepTopK;
	detectionParams.nbLayer = nbLayer;
	detectionParams.nbPriorbox = nbPrior;
	detectionParams.srcW = mParams.srcW;
	detectionParams.srcH = mParams.srcH;
	detectionParams.targetW = mParams.imgW;
	detectionParams.targetH = mParams.imgH;
	detectionParams.scoreThreshold = mParams.scoreThr;
	detectionParams.iouThreshold = 0.5;
	std::vector<int> featureSize(maxLayer - minLayer + 1);
	std::vector<int> topkCandidates(maxLayer - minLayer + 1);

	for (int i = 0; i < (maxLayer - minLayer + 1); i++)
	{
		featureSize[i] = (anchor->getOutput(i)->getDimensions().d[0]) / nbPrior / 4;
		int minNum = std::min(featureSize[i] * nbPrior, detectionParams.keepTopK);
		topkCandidates[i] = minNum;
	}

	detectronPlugin::DetectionOutput detectionOutput(detectionParams, featureSize.data(), topkCandidates.data());

	//ITensor **pAnchor = new ITensor*[nbLayer];
	std::vector<ITensor*> pAnchor(nbLayer);
	std::vector<ITensor*> pConf(nbLayer);
	std::vector<ITensor*> pLoc(nbLayer);

	for (int i = 0; i < nbLayer; i++)
	{
		pAnchor[i] = anchor->getOutput(i);
		pConf[i] = clsHead[i]->getOutput(0);
		pLoc[i] = locHead[i]->getOutput(0);
	}

	std::vector<ITensor*> inputNms(3 * nbLayer);
	for (int i = 0; i < nbLayer; i++)
	{
		inputNms[i] = pConf[i];
		inputNms[i + nbLayer] = pLoc[i];
		inputNms[i + 2 * nbLayer] = pAnchor[i];
	}

	auto result = network->addPluginV2(inputNms.data(), 3 * nbLayer, detectionOutput);

	//set output
	result->getOutput(0)->setName("Loc");
	result->getOutput(1)->setName("Score");
	result->getOutput(2)->setName("Cls");
	network->markOutput(*result->getOutput(0));
	network->markOutput(*result->getOutput(1));
	network->markOutput(*result->getOutput(2));

	return true;
}

bool EngineDetRetinaNet::constructResnet(ILayer * input, nvinfer1::INetworkDefinition* network, int minLayer, int maxLayer, std::vector<ILayer*>& output)
{
	/*
   ************************************************************************************
									   block_2
   ************************************************************************************
   stride_per_block:[1,1,1], in_channels:64, out_channels:256,  num_blocks:3
   if in_channels!= out_channels, shortCut=conv2d(in_channels, out_channels)
   ************************************************************************************
   */
   //int i = 0;
	int nbLayer = maxLayer - minLayer + 1;
	IActivationLayer* x2 = bottleneck(network, mWeightMap, *input->getOutput(0), 64, 64, 1, "backbone.bottom_up.res2.0");
	x2 = bottleneck(network, mWeightMap, *x2->getOutput(0), 256, 64, 1, "backbone.bottom_up.res2.1");
	x2 = bottleneck(network, mWeightMap, *x2->getOutput(0), 256, 64, 1, "backbone.bottom_up.res2.2");
	if (minLayer == 2)
	{
		output.push_back(x2);
	}
	//x2->getOutput(0)->setName("Loc");
	//resnetOutput[1]->getOutput(0)->setName("Loc1");
	//resnetOutput[2]->getOutput(0)->setName("Loc2");
	//network->markOutput(*x2->getOutput(0));
	//network->markOutput(*resnetOutput[1]->getOutput(0));
	//network->markOutput(*resnetOutput[2]->getOutput(0));
	/*
	************************************************************************************
										block_3
	************************************************************************************
	stride_per_block:[2,1,1,1], in_channels:256, out_channels:512, num_blocks:4
	************************************************************************************
	*/

	IActivationLayer* x3 = bottleneck(network, mWeightMap, *x2->getOutput(0), 256, 128, 2, "backbone.bottom_up.res3.0");
	x3 = bottleneck(network, mWeightMap, *x3->getOutput(0), 512, 128, 1, "backbone.bottom_up.res3.1");
	x3 = bottleneck(network, mWeightMap, *x3->getOutput(0), 512, 128, 1, "backbone.bottom_up.res3.2");
	x3 = bottleneck(network, mWeightMap, *x3->getOutput(0), 512, 128, 1, "backbone.bottom_up.res3.3");
	if (minLayer <= 3 && maxLayer >= 3)
	{
		output.push_back(x3);

	}
	/*
	************************************************************************************
										block_4
	************************************************************************************
	stride_per_block:[2,1,1,1,1,1], in_channels:512, out_channels:1024, num_blocks:6
	************************************************************************************
	*/

	IActivationLayer* x4 = bottleneck(network, mWeightMap, *x3->getOutput(0), 512, 256, 2, "backbone.bottom_up.res4.0");
	x4 = bottleneck(network, mWeightMap, *x4->getOutput(0), 1024, 256, 1, "backbone.bottom_up.res4.1");
	x4 = bottleneck(network, mWeightMap, *x4->getOutput(0), 1024, 256, 1, "backbone.bottom_up.res4.2");
	x4 = bottleneck(network, mWeightMap, *x4->getOutput(0), 1024, 256, 1, "backbone.bottom_up.res4.3");
	x4 = bottleneck(network, mWeightMap, *x4->getOutput(0), 1024, 256, 1, "backbone.bottom_up.res4.4");
	x4 = bottleneck(network, mWeightMap, *x4->getOutput(0), 1024, 256, 1, "backbone.bottom_up.res4.5");
	if (minLayer <= 4 && maxLayer >= 4)
	{
		output.push_back(x4);

	}
	/*
	************************************************************************************
										block_5
	************************************************************************************
	stride_per_block:[2,1,1,1,1,1], in_channels:512, out_channels:1024, num_blocks:6
	************************************************************************************
	*/
	if (maxLayer < 5)
		return true;
	IActivationLayer* x5 = bottleneck(network, mWeightMap, *x4->getOutput(0), 1024, 512, 2, "backbone.bottom_up.res5.0");
	x5 = bottleneck(network, mWeightMap, *x5->getOutput(0), 2048, 512, 1, "backbone.bottom_up.res5.1");
	x5 = bottleneck(network, mWeightMap, *x5->getOutput(0), 2048, 512, 1, "backbone.bottom_up.res5.2");
	if (minLayer <= 5 && maxLayer >= 5)
	{
		output.push_back(x5);

	}

	return true;
}

bool EngineDetRetinaNet::constructFpn(std::vector<ILayer*>& input, nvinfer1::INetworkDefinition* network, std::vector<ILayer*>& output, int minLayer, int maxLayer)
{
	ILayer* previous = input.back();
	ResizeNearest upSampling(2);

	if (maxLayer == 7)
	{
		IConvolutionLayer *x6 = network->addConvolution(*previous->getOutput(0), 256, DimsHW{ 3,3 }, mWeightMap["backbone.top_block.p6.weight"], mWeightMap["backbone.top_block.p6.bias"]);
		x6->setStride(DimsHW{ 2,2 });
		x6->setPadding(DimsHW{ 1,1 });

		IActivationLayer* x6_relu = network->addActivation(*x6->getOutput(0), ActivationType::kRELU);
		IConvolutionLayer *x7 = network->addConvolution(*x6_relu->getOutput(0), 256, DimsHW{ 3,3 }, mWeightMap["backbone.top_block.p7.weight"], mWeightMap["backbone.top_block.p7.bias"]);
		x7->setStride(DimsHW{ 2,2 });
		x7->setPadding(DimsHW{ 1,1 });
		std::cout << "d0:"<< x7->getOutput(0)->getDimensions().d[0] << std::endl;
		std::cout << "d1:" << x7->getOutput(0)->getDimensions().d[1] << std::endl;
		std::cout << "d2:" << x7->getOutput(0)->getDimensions().d[2] << std::endl;
		output.push_back(x7);
		output.push_back(x6);
	}

	if (maxLayer == 6)
	{
		IConvolutionLayer *x6 = network->addConvolution(*previous->getOutput(0), 256, DimsHW{ 3,3 }, mWeightMap["backbone.top_block.p6.weight"], mWeightMap["backbone.top_block.p6.bias"]);
		x6->setStride(DimsHW{ 2,2 });
		x6->setPadding(DimsHW{ 1,1 });
		output.push_back(x6);
	}

	if (maxLayer >= 5 && 5 >= minLayer)
	{
		/*
			************************************************************************************
									 FPN-P5
			************************************************************************************
		*/
		IConvolutionLayer *lateralConv5 = network->addConvolution(*previous->getOutput(0), 256, DimsHW{ 1,1 }, mWeightMap["backbone.fpn_lateral5.weight"], mWeightMap["backbone.fpn_lateral5.bias"]);
		IConvolutionLayer *outputConv5 = network->addConvolution(*lateralConv5->getOutput(0), 256, DimsHW{ 3,3 }, mWeightMap["backbone.fpn_output5.weight"], mWeightMap["backbone.fpn_output5.bias"]);
		outputConv5->setPadding(DimsHW{ 1,1 });
		output.push_back(outputConv5);

		ITensor *in[] = { &(*lateralConv5->getOutput(0)) };
		ILayer* topToDownFeature1 = network->addPluginV2(in, 1, upSampling);
		previous = topToDownFeature1;
	}



	if (maxLayer == 4)
	{
		IConvolutionLayer *lateralConv4 = network->addConvolution(*previous->getOutput(0), 256, DimsHW{ 1,1 }, mWeightMap["backbone.fpn_lateral4.weight"], mWeightMap["backbone.fpn_lateral4.bias"]);
		IConvolutionLayer *outputConv4 = network->addConvolution(*lateralConv4->getOutput(0), 256, DimsHW{ 3,3 }, mWeightMap["backbone.fpn_output4.weight"], mWeightMap["backbone.fpn_output4.bias"]);
		outputConv4->setPadding(DimsHW{ 1,1 });
		output.push_back(outputConv4);

		ITensor *in1[] = { &(*lateralConv4->getOutput(0)) };
		ILayer* topTodownFeature2 = network->addPluginV2(in1, 1, upSampling);
		previous = topTodownFeature2;
	}
	else
	{
		IConvolutionLayer *lateralConv4 = network->addConvolution(*input[4 - minLayer]->getOutput(0), 256, DimsHW{ 1,1 }, mWeightMap["backbone.fpn_lateral4.weight"], mWeightMap["backbone.fpn_lateral4.bias"]);
		IElementWiseLayer *ew1 = network->addElementWise(*previous->getOutput(0), *lateralConv4->getOutput(0), ElementWiseOperation::kSUM);
		IConvolutionLayer *outputConv4 = network->addConvolution(*ew1->getOutput(0), 256, DimsHW{ 3,3 }, mWeightMap["backbone.fpn_output4.weight"], mWeightMap["backbone.fpn_output4.bias"]);
		outputConv4->setPadding(DimsHW{ 1,1 });
		output.push_back(outputConv4);

		if (minLayer == 4)
		{
			return true;
		}

		ITensor *in1[] = { &(*ew1->getOutput(0)) };
		ILayer* topTodownFeature2 = network->addPluginV2(in1, 1, upSampling);
		previous = topTodownFeature2;
	}

	if (maxLayer == 3)
	{
		IConvolutionLayer *lateralConv3 = network->addConvolution(*previous->getOutput(0), 256, DimsHW{ 1,1 }, mWeightMap["backbone.fpn_lateral3.weight"], mWeightMap["backbone.fpn_lateral3.bias"]);
		IConvolutionLayer *outputConv3 = network->addConvolution(*lateralConv3->getOutput(0), 256, DimsHW{ 3,3 }, mWeightMap["backbone.fpn_output3.weight"], mWeightMap["backbone.fpn_output3.bias"]);
		outputConv3->setPadding(DimsHW{ 1,1 });
		output.push_back(outputConv3);

		ITensor *in2[] = { &(*lateralConv3->getOutput(0)) };
		ILayer* topTodownFeature3 = network->addPluginV2(in2, 1, upSampling);
		previous = topTodownFeature3;
	}
	else
	{
		IConvolutionLayer *lateralConv3 = network->addConvolution(*input[3 - minLayer]->getOutput(0), 256, DimsHW{ 1,1 }, mWeightMap["backbone.fpn_lateral3.weight"], mWeightMap["backbone.fpn_lateral3.bias"]);
		IElementWiseLayer *ew2 = network->addElementWise(*previous->getOutput(0), *lateralConv3->getOutput(0), ElementWiseOperation::kSUM);
		IConvolutionLayer *outputConv3 = network->addConvolution(*ew2->getOutput(0), 256, DimsHW{ 3,3 }, mWeightMap["backbone.fpn_output3.weight"], mWeightMap["backbone.fpn_output3.bias"]);
		outputConv3->setPadding(DimsHW{ 1,1 });
		output.push_back(outputConv3);

		if (minLayer == 3)
		{
			return true;
		}

		ITensor *in2[] = { &(*ew2->getOutput(0)) };
		ILayer* topTodownFeature3 = network->addPluginV2(in2, 1, upSampling);
		previous = topTodownFeature3;
	}

	if (minLayer == 2)
	{
		IConvolutionLayer *lateralConv2 = network->addConvolution(*input[2 - minLayer]->getOutput(0), 256, DimsHW{ 1,1 }, mWeightMap["backbone.fpn_lateral2.weight"], mWeightMap["backbone.fpn_lateral2.bias"]);
		IElementWiseLayer *ew3 = network->addElementWise(*previous->getOutput(0), *lateralConv2->getOutput(0), ElementWiseOperation::kSUM);
		IConvolutionLayer *outputConv2 = network->addConvolution(*ew3->getOutput(0), 256, DimsHW{ 3,3 }, mWeightMap["backbone.fpn_output2.weight"], mWeightMap["backbone.fpn_output2.bias"]);
		outputConv2->setPadding(DimsHW{ 1,1 });
		output.push_back(outputConv2);
	}

	return true;
}

bool EngineDetRetinaNet::constructHead(std::vector<ILayer*>& inFeature, nvinfer1::INetworkDefinition* network, std::vector<ILayer*>& classHead, std::vector<ILayer*>& locHead, int nbCls, int nbPrior)
{
	int nbLayer = inFeature.size();

	for (int i = nbLayer - 1; i >= 0; i--)
	{
		IConvolutionLayer* cls = addHead(network, mWeightMap, *inFeature[i]->getOutput(0), "head.cls_subnet", nbCls, nbPrior, 0);
		IConvolutionLayer* loc = addHead(network, mWeightMap, *inFeature[i]->getOutput(0), "head.bbox_subnet", nbCls, nbPrior, 1);
		classHead.push_back(cls);
		locHead.push_back(loc);
	}

	return true;
}

IScaleLayer * EngineDetRetinaNet::addBatchNorm2d(nvinfer1::INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor & input, std::string lname, float eps)
{
	float *gamma = (float*)weightMap[lname + ".weight"].values;
	float *beta = (float*)weightMap[lname + ".bias"].values;
	float *mean = (float*)weightMap[lname + ".running_mean"].values;
	float *var = (float*)weightMap[lname + ".running_var"].values;
	int len = weightMap[lname + ".running_var"].count;

	float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval[i] = gamma[i] / sqrt(var[i] + eps);
	}
	Weights scale{ DataType::kFLOAT, scval, len };

	float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
	}
	Weights shift{ DataType::kFLOAT, shval, len };

	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	Weights power{ DataType::kFLOAT, pval, len };

	//weightMap[lname + ".scale"] = scale;
	//weightMap[lname + ".shift"] = shift;
	//weightMap[lname + ".power"] = power;
	IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
	//assert(scale_1);
	return scale_1;
}

IActivationLayer * EngineDetRetinaNet::bottleneck(nvinfer1::INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor & input, int inch, int outch, int stride, std::string lname)
{
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ 1, 1 }, weightMap[lname + ".conv1.weight"], emptywts);
	conv1->setStride(DimsHW{ stride, stride });
	assert(conv1);

	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".conv1.norm", 1e-5);

	IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	assert(relu1);

	IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], emptywts);
	assert(conv2);
	//conv2->setStride(DimsHW{ stride, stride });
	conv2->setPadding(DimsHW{ 1, 1 });

	IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".conv2.norm", 1e-5);

	IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
	assert(relu2);

	IConvolutionLayer* conv3 = network->addConvolution(*relu2->getOutput(0), outch * 4, DimsHW{ 1, 1 }, weightMap[lname + ".conv3.weight"], emptywts);
	assert(conv3);

	IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".conv3.norm", 1e-5);

	IElementWiseLayer* ew1;
	if (stride != 1 || inch != outch * 4) {
		IConvolutionLayer* conv4 = network->addConvolution(input, outch * 4, DimsHW{ 1, 1 }, weightMap[lname + ".shortcut.weight"], emptywts);
		assert(conv4);
		conv4->setStride(DimsHW{ stride, stride });

		IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + ".shortcut.norm", 1e-5);
		ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
	}
	else {
		ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
	}
	IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
	assert(relu3);
	return relu3;
}

IActivationLayer * EngineDetRetinaNet::bottleneck_test(nvinfer1::INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor & input, int inch, int outch, int stride, std::string lname)
{
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ 1, 1 }, weightMap[lname + ".conv1.weight"], emptywts);
	conv1->setStride(DimsHW{ stride, stride });
	assert(conv1);

	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".conv1.norm", 1e-5);

	IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	assert(relu1);

	IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], emptywts);
	assert(conv2);
	//conv2->setStride(DimsHW{ stride, stride });
	conv2->setPadding(DimsHW{ 1, 1 });

	IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".conv2.norm", 1e-5);

	IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
	assert(relu2);

	IConvolutionLayer* conv3 = network->addConvolution(*relu2->getOutput(0), outch * 4, DimsHW{ 1, 1 }, weightMap[lname + ".conv3.weight"], emptywts);
	assert(conv3);

	IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".conv3.norm", 1e-5);

	bn3->getOutput(0)->setName("Loc");
	network->markOutput(*bn3->getOutput(0));

	IElementWiseLayer* ew1;
	if (stride != 1 || inch != outch * 4) {
		IConvolutionLayer* conv4 = network->addConvolution(input, outch * 4, DimsHW{ 1, 1 }, weightMap[lname + ".shortcut.weight"], emptywts);
		assert(conv4);
		conv4->setStride(DimsHW{ stride, stride });

		IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + ".shortcut.norm", 1e-5);
		ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
	}
	else {
		ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
	}
	IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
	assert(relu3);


	return relu3;
}

IConvolutionLayer * EngineDetRetinaNet::addHead(nvinfer1::INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor & input, const std::string & lname, const int & nbCls, const int & anchorNum, const int & index)
{
	std::cout << "lname:" << lname << std::endl;
	IConvolutionLayer *conv1 = network->addConvolution(input, 256, DimsHW{ 3,3 }, weightMap[lname + ".0.weight"], weightMap[lname + ".0.bias"]);
	conv1->setStride(DimsHW{ 1,1 });
	conv1->setPadding(DimsHW{ 1,1 });
	IActivationLayer *relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
	assert(relu1);

	IConvolutionLayer *conv2 = network->addConvolution(*relu1->getOutput(0), 256, DimsHW{ 3,3 }, weightMap[lname + ".2.weight"], weightMap[lname + ".2.bias"]);
	conv2->setStride(DimsHW{ 1,1 });
	conv2->setPadding(DimsHW{ 1,1 });
	IActivationLayer *relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
	assert(relu2);

	IConvolutionLayer *conv3 = network->addConvolution(*relu2->getOutput(0), 256, DimsHW{ 3,3 }, weightMap[lname + ".4.weight"], weightMap[lname + ".4.bias"]);
	conv3->setStride(DimsHW{ 1,1 });
	conv3->setPadding(DimsHW{ 1,1 });
	IActivationLayer *relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
	assert(relu3);

	IConvolutionLayer *conv4 = network->addConvolution(*relu3->getOutput(0), 256, DimsHW{ 3,3 }, weightMap[lname + ".6.weight"], weightMap[lname + ".6.bias"]);
	conv4->setStride(DimsHW{ 1,1 });
	conv4->setPadding(DimsHW{ 1,1 });
	IActivationLayer *relu4 = network->addActivation(*conv4->getOutput(0), ActivationType::kRELU);
	assert(relu4);

	//return relu4;

	if (index == 0)
	{
		IConvolutionLayer *conv5 = network->addConvolution(*relu4->getOutput(0), anchorNum * nbCls, DimsHW{ 3,3 }, weightMap["head.cls_score.weight"], weightMap["head.cls_score.bias"]);
		conv5->setStride(DimsHW{ 1,1 });
		conv5->setPadding(DimsHW{ 1,1 });
		return conv5;
	}
	else if (index == 1)
	{
		IConvolutionLayer *conv5 = network->addConvolution(*relu4->getOutput(0), anchorNum * 4, DimsHW{ 3,3 }, weightMap["head.bbox_pred.weight"], weightMap["head.bbox_pred.bias"]);
		conv5->setStride(DimsHW{ 1,1 });
		conv5->setPadding(DimsHW{ 1,1 });
		return conv5;
	}
}
