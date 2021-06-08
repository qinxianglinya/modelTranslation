#include "EngineDetYolov3.h"
#include "common.h"
#include "Yolov3AnchorPlugin.h"
#include "Yolov3NmsPlugin.h"

using namespace detectron2;

EngineDetYolov3::EngineDetYolov3(Yolov3Params params) :mParams(params)
{
}

EngineDetYolov3::~EngineDetYolov3()
{
}

void EngineDetYolov3::initEngine()
{
	mWeightMap = loadWeights(mParams.weightsPath);

	initLibNvInferPlugins(&mLogger, "");
	m_Builder = nvinfer1::createInferBuilder(mLogger);
	assert(m_Builder != nullptr);
	m_Network = m_Builder->createNetwork();
	assert(m_Network != nullptr);
	constructNetwork(m_Builder, m_Network);
	m_Config = m_Builder->createBuilderConfig();
	assert(m_Config);
	setConfig(m_Builder, m_Network);
}

void EngineDetYolov3::setConfig(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network)
{
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

std::map<std::string, nvinfer1::Weights> EngineDetYolov3::loadWeights(const std::string & file)
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
		input >> name >> std::dec >> size;
		wt.type = DataType::kFLOAT;

		// Load blob
		auto *val = reinterpret_cast<uint32_t *>(malloc(sizeof(uint32_t) * size));
		//std::cout << "i:" << count << std::endl;
		//clock_t start = clock();
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

IScaleLayer* EngineDetYolov3::addBatchNorm2d(nvinfer1::INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor & input, const std::string& lname, float eps)
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

	IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
	assert(scale_1);
	return scale_1;
}

ILayer* EngineDetYolov3::convBnLeaky(INetworkDefinition *network, ITensor& input, int outch, int ksize, int s, int p, const std::string& commonStr, const enum NetworkPart& networkPart)
{
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	//"backbone."
	std::string convName, bnName;
	if (networkPart == 1)
	{
		convName = "backbone." + commonStr + ".conv.weight";
		bnName = "backbone." + commonStr + ".bn";
	}
	//neck
	else if (networkPart == 2)
	{
		convName = "neck." + commonStr + ".conv.weight";
		bnName = "neck." + commonStr + ".bn";
	}
	else if (networkPart == 3)
	{
		convName = "bbox_head.convs_bridge." + commonStr + ".conv.weight";
		bnName = "bbox_head.convs_bridge." + commonStr + ".bn";
	}

	IConvolutionLayer *conv = network->addConvolutionNd(input, outch, DimsHW(ksize, ksize), mWeightMap[convName], emptywts);
	conv->setStride(DimsHW{ s,s });
	conv->setPadding(DimsHW{ p,p });

	IScaleLayer* bn = addBatchNorm2d(network, mWeightMap, *conv->getOutput(0), bnName, 1e-5);

	IActivationLayer* relu = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
	relu->setAlpha(0.1);

	return relu;
}

ILayer * EngineDetYolov3::resBlock(INetworkDefinition * network, ITensor & input, int outch1, int ksize1, int s1, int p1, int ksize2, int s2, int p2, const std::string & commonStr)
{
	enum NetworkPart networkPart = BACKBONE;
	std::string convStr1 = commonStr + ".conv1";
	std::string convStr2 = commonStr + ".conv2";

	auto cbl1 = convBnLeaky(network, input, outch1, ksize1, s1, p1, convStr1, networkPart);
	auto cbl2 = convBnLeaky(network, *(cbl1->getOutput(0)), outch1 * 2, ksize2, s2, p2, convStr2, networkPart);

	auto out = network->addElementWise(input, *cbl2->getOutput(0), ElementWiseOperation::kSUM);
	return out;
}

ILayer * EngineDetYolov3::convResBlock1(INetworkDefinition * network, ITensor & input)
{
	enum NetworkPart networkPart = BACKBONE;
	const std::string convStr = "conv_res_block1.conv";
	const std::string resStr = "conv_res_block1.res0";

	auto conv1 = convBnLeaky(network, input, 64, 3, 2, 1, convStr, networkPart);
	auto crb = resBlock(network, *conv1->getOutput(0), 32, 1, 1, 0, 3, 1, 1, resStr);

	return crb;
}

ILayer * EngineDetYolov3::convResBlock2(INetworkDefinition * network, ITensor & input)
{
	enum NetworkPart networkPart = BACKBONE;
	const std::string convStr = "conv_res_block2.conv";

	auto conv1 = convBnLeaky(network, input, 128, 3, 2, 1, convStr, networkPart);

	const std::string resStr0 = "conv_res_block2.res0";
	const std::string resStr1 = "conv_res_block2.res1";

	auto rb0 = resBlock(network, *conv1->getOutput(0), 64, 1, 1, 0, 3, 1, 1, resStr0);

	auto rb1 = resBlock(network, *rb0->getOutput(0), 64, 1, 1, 0, 3, 1, 1, resStr1);

	return rb1;
}

ILayer * EngineDetYolov3::convResBlock3(INetworkDefinition * network, ITensor & input)
{
	enum NetworkPart networkPart = BACKBONE;

	const std::string convStr = "conv_res_block3.conv";
	const std::string resStr0 = "conv_res_block3.res0";
	const std::string resStr1 = "conv_res_block3.res1";
	const std::string resStr2 = "conv_res_block3.res2";
	const std::string resStr3 = "conv_res_block3.res3";
	const std::string resStr4 = "conv_res_block3.res4";
	const std::string resStr5 = "conv_res_block3.res5";
	const std::string resStr6 = "conv_res_block3.res6";
	const std::string resStr7 = "conv_res_block3.res7";

	auto conv1 = convBnLeaky(network, input, 256, 3, 2, 1, convStr, networkPart);

	auto rb0 = resBlock(network, *conv1->getOutput(0), 128, 1, 1, 0, 3, 1, 1, resStr0);
	auto rb1 = resBlock(network, *rb0->getOutput(0), 128, 1, 1, 0, 3, 1, 1, resStr1);
	auto rb2 = resBlock(network, *rb1->getOutput(0), 128, 1, 1, 0, 3, 1, 1, resStr2);
	auto rb3 = resBlock(network, *rb2->getOutput(0), 128, 1, 1, 0, 3, 1, 1, resStr3);
	auto rb4 = resBlock(network, *rb3->getOutput(0), 128, 1, 1, 0, 3, 1, 1, resStr4);
	auto rb5 = resBlock(network, *rb4->getOutput(0), 128, 1, 1, 0, 3, 1, 1, resStr5);
	auto rb6 = resBlock(network, *rb5->getOutput(0), 128, 1, 1, 0, 3, 1, 1, resStr6);
	auto rb7 = resBlock(network, *rb6->getOutput(0), 128, 1, 1, 0, 3, 1, 1, resStr7);

	return rb7;
}

ILayer * EngineDetYolov3::convResBlock4(INetworkDefinition * network, ITensor & input)
{
	enum NetworkPart networkPart = BACKBONE;

	const std::string convStr = "conv_res_block4.conv";
	const std::string resStr0 = "conv_res_block4.res0";
	const std::string resStr1 = "conv_res_block4.res1";
	const std::string resStr2 = "conv_res_block4.res2";
	const std::string resStr3 = "conv_res_block4.res3";
	const std::string resStr4 = "conv_res_block4.res4";
	const std::string resStr5 = "conv_res_block4.res5";
	const std::string resStr6 = "conv_res_block4.res6";
	const std::string resStr7 = "conv_res_block4.res7";

	auto conv1 = convBnLeaky(network, input, 512, 3, 2, 1, convStr, networkPart);

	auto rb0 = resBlock(network, *conv1->getOutput(0), 256, 1, 1, 0, 3, 1, 1, resStr0);
	auto rb1 = resBlock(network, *rb0->getOutput(0), 256, 1, 1, 0, 3, 1, 1, resStr1);
	auto rb2 = resBlock(network, *rb1->getOutput(0), 256, 1, 1, 0, 3, 1, 1, resStr2);
	auto rb3 = resBlock(network, *rb2->getOutput(0), 256, 1, 1, 0, 3, 1, 1, resStr3);
	auto rb4 = resBlock(network, *rb3->getOutput(0), 256, 1, 1, 0, 3, 1, 1, resStr4);
	auto rb5 = resBlock(network, *rb4->getOutput(0), 256, 1, 1, 0, 3, 1, 1, resStr5);
	auto rb6 = resBlock(network, *rb5->getOutput(0), 256, 1, 1, 0, 3, 1, 1, resStr6);
	auto rb7 = resBlock(network, *rb6->getOutput(0), 256, 1, 1, 0, 3, 1, 1, resStr7);

	return rb7;
}

ILayer * EngineDetYolov3::convResBlock5(INetworkDefinition * network, ITensor & input)
{
	enum NetworkPart networkPart = BACKBONE;

	const std::string convStr = "conv_res_block5.conv";
	const std::string resStr0 = "conv_res_block5.res0";
	const std::string resStr1 = "conv_res_block5.res1";
	const std::string resStr2 = "conv_res_block5.res2";
	const std::string resStr3 = "conv_res_block5.res3";

	auto conv1 = convBnLeaky(network, input, 1024, 3, 2, 1, convStr, networkPart);

	auto rb0 = resBlock(network, *conv1->getOutput(0), 512, 1, 1, 0, 3, 1, 1, resStr0);
	auto rb1 = resBlock(network, *rb0->getOutput(0), 512, 1, 1, 0, 3, 1, 1, resStr1);
	auto rb2 = resBlock(network, *rb1->getOutput(0), 512, 1, 1, 0, 3, 1, 1, resStr2);
	auto rb3 = resBlock(network, *rb2->getOutput(0), 512, 1, 1, 0, 3, 1, 1, resStr3);

	return rb3;
}

void EngineDetYolov3::constructDarknet(INetworkDefinition * network, ITensor & input, std::vector<ILayer*>& darknetOut)
{
	enum NetworkPart networkPart = BACKBONE;
	auto conv1 = convBnLeaky(network, input, 32, 3, 1, 1, "conv1", networkPart);

	//搭建darknet的5个conv_res_block模块
	auto crb1 = convResBlock1(network, *conv1->getOutput(0));

	auto crb2 = convResBlock2(network, *crb1->getOutput(0));

	auto crb3 = convResBlock3(network, *crb2->getOutput(0));

	auto crb4 = convResBlock4(network, *crb3->getOutput(0));

	auto crb5 = convResBlock5(network, *crb4->getOutput(0));

	darknetOut.push_back(crb3);
	darknetOut.push_back(crb4);
	darknetOut.push_back(crb5);
	//测试模块输出
	//conv1->getOutput(0)->setName("conv1");
	//network->markOutput(*conv1->getOutput(0));

	//crb1->getOutput(0)->setName("1");
	//network->markOutput(*crb1->getOutput(0));

	//crb2->getOutput(0)->setName("2");
	//network->markOutput(*crb2->getOutput(0));



}

ILayer * EngineDetYolov3::DetectionBlock3(nvinfer1::IBuilder * builder, nvinfer1::INetworkDefinition * network, ILayer * input)
{
	enum NetworkPart networkPart = NECK;

	auto conv1 = convBnLeaky(network, *input->getOutput(0), 128, 1, 1, 0, "detect3.conv1", networkPart);
	auto conv2 = convBnLeaky(network, *conv1->getOutput(0), 256, 3, 1, 1, "detect3.conv2", networkPart);
	auto conv3 = convBnLeaky(network, *conv2->getOutput(0), 128, 1, 1, 0, "detect3.conv3", networkPart);
	auto conv4 = convBnLeaky(network, *conv3->getOutput(0), 256, 3, 1, 1, "detect3.conv4", networkPart);
	auto conv5 = convBnLeaky(network, *conv4->getOutput(0), 128, 1, 1, 0, "detect3.conv5", networkPart);

	return conv5;
}

ILayer * EngineDetYolov3::DetectionBlock2(nvinfer1::IBuilder * builder, nvinfer1::INetworkDefinition * network, ILayer * input)
{
	enum NetworkPart networkPart = NECK;

	auto conv1 = convBnLeaky(network, *input->getOutput(0), 256, 1, 1, 0, "detect2.conv1", networkPart);
	auto conv2 = convBnLeaky(network, *conv1->getOutput(0), 512, 3, 1, 1, "detect2.conv2", networkPart);
	auto conv3 = convBnLeaky(network, *conv2->getOutput(0), 256, 1, 1, 0, "detect2.conv3", networkPart);
	auto conv4 = convBnLeaky(network, *conv3->getOutput(0), 512, 3, 1, 1, "detect2.conv4", networkPart);
	auto conv5 = convBnLeaky(network, *conv4->getOutput(0), 256, 1, 1, 0, "detect2.conv5", networkPart);

	return conv5;
}

ILayer * EngineDetYolov3::DetectionBlock1(nvinfer1::IBuilder * builder, nvinfer1::INetworkDefinition * network, ILayer * input)
{
	enum NetworkPart networkPart = NECK;

	std::cout << input->getOutput(0)->getDimensions().d[0] << std::endl;
	std::cout << input->getOutput(0)->getDimensions().d[1] << std::endl;
	std::cout << input->getOutput(0)->getDimensions().d[2] << std::endl;

	auto conv1 = convBnLeaky(network, *input->getOutput(0), 512, 1, 1, 0, "detect1.conv1", networkPart);
	auto conv2 = convBnLeaky(network, *conv1->getOutput(0), 1024, 3, 1, 1, "detect1.conv2", networkPart);
	auto conv3 = convBnLeaky(network, *conv2->getOutput(0), 512, 1, 1, 0, "detect1.conv3", networkPart);
	auto conv4 = convBnLeaky(network, *conv3->getOutput(0), 1024, 3, 1, 1, "detect1.conv4", networkPart);
	auto conv5 = convBnLeaky(network, *conv4->getOutput(0), 512, 1, 1, 0, "detect1.conv5", networkPart);

	return conv5;
}

void EngineDetYolov3::constructNeck(nvinfer1::IBuilder * builder, nvinfer1::INetworkDefinition * network, const std::vector<ILayer*>& input, std::vector<ILayer*>& output)
{
	enum NetworkPart networkPart = NECK;

	auto dblOut1 = DetectionBlock1(builder, network, input[2]);

	auto conv1 = convBnLeaky(network, *dblOut1->getOutput(0), 256, 1, 1, 0, "conv1", networkPart);

	//上采样
	ResizeNearest upSampling(2);
	ITensor *in[] = { conv1->getOutput(0) };
	ILayer* topToDownFeature1 = network->addPluginV2(in, 1, upSampling);

	ITensor* inputTensors[] = { topToDownFeature1->getOutput(0), input[1]->getOutput(0) };
	auto concat1 = network->addConcatenation(inputTensors, 2);

	auto dblOut2 = DetectionBlock2(builder, network, concat1);

	auto conv2 = convBnLeaky(network, *dblOut2->getOutput(0), 128, 1, 1, 0, "conv2", networkPart);

	ResizeNearest upSampling1(2);
	ITensor *in1[] = { conv2->getOutput(0) };
	ILayer* topToDownFeature2 = network->addPluginV2(in1, 1, upSampling1);

	ITensor* inputTensors1[] = { topToDownFeature2->getOutput(0), input[0]->getOutput(0) };
	auto concat2 = network->addConcatenation(inputTensors1, 2);

	auto dblOut3 = DetectionBlock3(builder, network, concat2);

	//dblOut1:最上一层
	output.push_back(dblOut1);
	output.push_back(dblOut2);
	output.push_back(dblOut3);

	//dblOut1->getOutput(0)->setName("3");
	//network->markOutput(*dblOut1->getOutput(0));

	//dblOut2->getOutput(0)->setName("4");
	//network->markOutput(*dblOut2->getOutput(0));

	//dblOut3->getOutput(0)->setName("5");
	//network->markOutput(*dblOut3->getOutput(0));
}

void EngineDetYolov3::constructHead(nvinfer1::IBuilder * builder, nvinfer1::INetworkDefinition * network, const std::vector<ILayer*>& input, std::vector<ILayer*>& output)
{
	int nbCls = mParams.nbCls;
	int outDims = (4 + 1 + nbCls) * 3;
	std::cout << "outDims:" << outDims << std::endl;

	enum NetworkPart networkPart = HEAD;
	auto cbl0 = convBnLeaky(network, *input[0]->getOutput(0), 1024, 3, 1, 1, "0", networkPart);
	auto pre0 = network->addConvolutionNd(*cbl0->getOutput(0), outDims, DimsHW(1, 1), mWeightMap["bbox_head.convs_pred.0.weight"], mWeightMap["bbox_head.convs_pred.0.bias"]);

	auto cbl1 = convBnLeaky(network, *input[1]->getOutput(0), 512, 3, 1, 1, "1", networkPart);
	auto pre1 = network->addConvolutionNd(*cbl1->getOutput(0), outDims, DimsHW(1, 1), mWeightMap["bbox_head.convs_pred.1.weight"], mWeightMap["bbox_head.convs_pred.1.bias"]);

	auto cbl2 = convBnLeaky(network, *input[2]->getOutput(0), 256, 3, 1, 1, "2", networkPart);
	auto pre2 = network->addConvolutionNd(*cbl2->getOutput(0), outDims, DimsHW(1, 1), mWeightMap["bbox_head.convs_pred.2.weight"], mWeightMap["bbox_head.convs_pred.2.bias"]);

	output.push_back(pre0);
	output.push_back(pre1);
	output.push_back(pre2);

	//pre0->getOutput(0)->setName("3");
	//network->markOutput(*pre0->getOutput(0));

	//pre1->getOutput(0)->setName("4");
	//network->markOutput(*pre1->getOutput(0));

	//pre2->getOutput(0)->setName("5");
	//network->markOutput(*pre2->getOutput(0));
}

ILayer* EngineDetYolov3::constructAnchor(nvinfer1::IBuilder * builder, nvinfer1::INetworkDefinition * network)
{
	std::vector<AnchorParamsYolov3> anchorParamsVec(3);

	/*AnchorParamsYolov3 anchorParamsYolov3;*/
	//锚框的baseSize，人为设定，无计算规律。
	anchorParamsVec[0].fBaseSize[0].x = 116, anchorParamsVec[0].fBaseSize[0].y = 90;
	anchorParamsVec[0].fBaseSize[1].x = 156, anchorParamsVec[0].fBaseSize[1].y = 198;
	anchorParamsVec[0].fBaseSize[2].x = 373, anchorParamsVec[0].fBaseSize[2].y = 326;
	anchorParamsVec[0].nStride = 32;
	anchorParamsVec[0].featureSize.nWidth = (mParams.imgW / 32);
	anchorParamsVec[0].featureSize.nHeight = (mParams.imgH / 32);

	anchorParamsVec[1].fBaseSize[0].x = 30, anchorParamsVec[1].fBaseSize[0].y = 61;
	anchorParamsVec[1].fBaseSize[1].x = 62, anchorParamsVec[1].fBaseSize[1].y = 45;
	anchorParamsVec[1].fBaseSize[2].x = 59, anchorParamsVec[1].fBaseSize[2].y = 119;
	anchorParamsVec[1].nStride = 16;
	anchorParamsVec[1].featureSize.nWidth = (mParams.imgW / 16);
	anchorParamsVec[1].featureSize.nHeight = (mParams.imgH / 16);

	anchorParamsVec[2].fBaseSize[0].x = 10, anchorParamsVec[2].fBaseSize[0].y = 13;
	anchorParamsVec[2].fBaseSize[1].x = 16, anchorParamsVec[2].fBaseSize[1].y = 30;
	anchorParamsVec[2].fBaseSize[2].x = 33, anchorParamsVec[2].fBaseSize[2].y = 23;
	anchorParamsVec[2].nStride = 8;
	anchorParamsVec[2].featureSize.nWidth = (mParams.imgW / 8);
	anchorParamsVec[2].featureSize.nHeight = (mParams.imgH / 8);

	for (int i = 0; i < 3; i++)
	{
		float center = anchorParamsVec[i].nStride / 2.0;
		for (int j = 0; j < 3; j++)
		{

			anchorParamsVec[i].fBaseAnchor[j].x0 = center - 0.5 * anchorParamsVec[i].fBaseSize[j].x;
			anchorParamsVec[i].fBaseAnchor[j].y0 = center - 0.5 * anchorParamsVec[i].fBaseSize[j].y;
			anchorParamsVec[i].fBaseAnchor[j].x1 = center + 0.5 * anchorParamsVec[i].fBaseSize[j].x;
			anchorParamsVec[i].fBaseAnchor[j].y1 = center + 0.5 * anchorParamsVec[i].fBaseSize[j].y;
		}
	}

	//
	Yolov3AnchorPlugin yolov3AnchorPlugin(anchorParamsVec.data(), 3);
	auto anchorLayer = network->addPluginV2(nullptr, 0, yolov3AnchorPlugin);
	return anchorLayer;
}

ILayer * EngineDetYolov3::constructNms(nvinfer1::IBuilder * builder, nvinfer1::INetworkDefinition * network, std::vector<ILayer*>& input, ILayer * anchor)
{
	Yolov3NmsParams yolo3nmsParams;
	yolo3nmsParams.nbCls = mParams.nbCls;
	yolo3nmsParams.conf_thr = 0.005;
	yolo3nmsParams.score_thr = mParams.scoreThr;
	yolo3nmsParams.iou_thr = 0.5;

	yolo3nmsParams.stride[0] = 32;
	yolo3nmsParams.stride[1] = 16;
	yolo3nmsParams.stride[2] = 8;

	yolo3nmsParams.factor_scales[0] = (float)mParams.imgW / mParams.srcW;
	yolo3nmsParams.factor_scales[1] = (float)mParams.imgH / mParams.srcH;
	yolo3nmsParams.factor_scales[2] = (float)mParams.imgW / mParams.srcW;
	yolo3nmsParams.factor_scales[3] = (float)mParams.imgH / mParams.srcH;

	std::vector<int> featureSize(3);
	for (int i = 0; i < 3; i++)
	{
		featureSize[i] = (anchor->getOutput(i)->getDimensions().d[0]) / 3 / 4;
	}


	ITensor *in1[] = { input[0]->getOutput(0), anchor->getOutput(0), input[1]->getOutput(0), anchor->getOutput(1), input[2]->getOutput(0), anchor->getOutput(2) };

	Yolov3NmsPlugin yolov3NmsPlugin(yolo3nmsParams, featureSize.data());
	auto nmsLayer = network->addPluginV2(in1, 6, yolov3NmsPlugin);

	nmsLayer->getOutput(0)->setName("box");
	nmsLayer->getOutput(1)->setName("cls");
	nmsLayer->getOutput(2)->setName("conf");

	network->markOutput(*nmsLayer->getOutput(0));
	network->markOutput(*nmsLayer->getOutput(1));
	network->markOutput(*nmsLayer->getOutput(2));
	return nullptr;
}



void EngineDetYolov3::constructNetwork(nvinfer1::IBuilder * builder, nvinfer1::INetworkDefinition * network)
{
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	ITensor* data = network->addInput(
		"data", DataType::kFLOAT, Dims3{ 3, mParams.imgH, mParams.imgW });


	std::vector<ILayer*> darknetOut;
	constructDarknet(network, *data, darknetOut);


	std::vector<ILayer*> neckOut;
	constructNeck(builder, network, darknetOut, neckOut);

	std::vector<ILayer*> headOut;
	constructHead(builder, network, neckOut, headOut);

	auto anchorOut = constructAnchor(builder, network);

	constructNms(builder, network, headOut, anchorOut);


	//darknetOut[0]->getOutput(0)->setName("sss");
	//network->markOutput(*darknetOut[0]->getOutput(0));



}

