#ifndef STRUCT_H
#define STRUCT_H
#include <string>
#include <vector>
#include <iostream>
#include "NvinferRuntimeCommon.h"

struct InferDeleter
{
	template <typename T>
	void operator()(T* obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}
};

class TrtLogger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kVERBOSE)
			std::cout << msg << std::endl;
	}
};

struct EngineParams
{
	int imgW = 640;
	int imgH = 640;
	int imgC = 3;

	std::vector<std::string> inputTensorNames = {};
	std::vector<std::string> outputTensorNames = {};
	std::vector<std::vector<int>> inputDims = { {} };

	bool FP32 = false;
	bool FP16 = false;
	int maxBatchSize = 16;
	std::string trtSavePath;

};

struct UffParams : public EngineParams
{
	//resize∫ÛÕº∆¨–≈œ¢
	std::string uffModelPath = "";
};

struct DetRetinaNetParams : public EngineParams
{
	std::string weightsPath = "";
	float scoreThr = 0.3;
	int minLayer = 3;
	int maxLayer = 5;
	int nbPrior = 9;
	int nbCls = 5;
	int topK = 100;
	int keepTopK = 1000;
	int srcW = 320;
	int srcH = 320;
};

namespace myPlugin
{
	struct GridAnchorParametersTf
	{
		int level, scalesPerOctave;
		// float anchorScale;
		float* aspectRatios;
		int numAspectRatios;
		int imgH, imgW;
		int W, H;
		float variance[4];
	};

	struct Coordinate
	{
		float x0;
		float y0;
		float x1;
		float y1;
	};

	struct FeatureSize
	{
		int nWidth;
		int nHeight;
	};

	struct AnchorParamsTorch
	{
		int nOffset;
		int nStride;
		float nSize[3];
		float fAspectRatio[3] = { 0.5, 1, 2 };
		Coordinate fBaseAnchor[9];
		FeatureSize featureSize;
	};

	struct DetectionOutputParametersTorch
	{
		int nbCls, topK, keepTopK;
		int nbLayer;
		int nbPriorbox;
		int srcW, srcH, targetW, targetH;
		float scoreThreshold, iouThreshold;
	};

	struct Yolov3NmsParams
	{
		int nbCls;
		float conf_thr, score_thr, iou_thr;
		float factor_scales[4];
		int stride[3];
	};

	struct Yolov3Params : public EngineParams
	{
		int srcW = 320;
		int srcH = 320;
		int nbCls = 3;
		std::string weightsPath = "";
		float scoreThr = 0.3;
	};

	enum NetworkPart
	{
		BACKBONE = 1,
		NECK,
		HEAD
	};

	struct Coordinate2d
	{
		float x, y;
	};

	struct AnchorParamsYolov3
	{
		Coordinate fBaseAnchor[3];
		Coordinate2d fBaseSize[3];
		FeatureSize featureSize;
		int nStride;
	};
}

#endif // !STRUCT_H
