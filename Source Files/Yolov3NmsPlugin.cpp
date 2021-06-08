#include "Yolov3NmsPlugin.h"
#include "common.h"
#include "bboxUtils.h"
#include "kernel.h"
#include "nmsUtils.h"
namespace
{
	const char* NMS_YOLOV3_PLUGIN_VERSION{ "1" };
	const char* NMS_YOLOV3_PLUGIN_NAME{ "NMS_YOLOV3_TRT" };
} // namespace

detectron2::Yolov3NmsPlugin::Yolov3NmsPlugin(Yolov3NmsParams params, const int* featureSizeIn)
{
	mParams.nbCls = params.nbCls;
	mParams.conf_thr = params.conf_thr;
	mParams.score_thr = params.score_thr;
	mParams.iou_thr = params.iou_thr;
	for (int i = 0; i < 4; i++)
	{
		mParams.factor_scales[i] = params.factor_scales[i];
	}
	mFeatureSize.resize(3);
	for (int i = 0; i < 3; i++)
	{
		mFeatureSize[i] = featureSizeIn[i];
		mParams.stride[i] = params.stride[i];
	}
}

detectron2::Yolov3NmsPlugin::Yolov3NmsPlugin(const void * data, size_t length)
{
	const char *d = reinterpret_cast<const char*>(data), *a = d;
	mParams.nbCls = read<int>(d);
	mParams.conf_thr = read<float>(d);
	mParams.score_thr = read<float>(d);
	mParams.iou_thr = read<float>(d);
	for (int i = 0; i < 4; i++)
	{
		mParams.factor_scales[i] = read<float>(d);
	}
	mFeatureSize.resize(3);
	for (int i = 0; i < 3; i++)
	{
		mFeatureSize[i] = read<int>(d);
	}
	for (int i = 0; i < 3; i++)
	{
		mParams.stride[i] = read<int>(d);
	}
}

int detectron2::Yolov3NmsPlugin::getNbOutputs() const
{
	return 3;
}

Dims detectron2::Yolov3NmsPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
{
	if (index == 0)
	{
		return DimsCHW(1, 100 * 4, 1);
	}
	else if (index == 1)
	{
		return DimsCHW(1, 100, 1);
	}
	else
	{
		return DimsCHW(1, 100, 1);
	}

}

int detectron2::Yolov3NmsPlugin::initialize()
{
	return 0;
}

void detectron2::Yolov3NmsPlugin::terminate()
{
}

void detectron2::Yolov3NmsPlugin::destroy()
{
	delete this;
}

size_t detectron2::Yolov3NmsPlugin::getWorkspaceSize(int batchSize) const
{

	return yolov3NmsWorkspaceSize(batchSize, mParams, mFeatureSize[2]);
}

int detectron2::Yolov3NmsPlugin::enqueue(int batch_size, const void * const * inputs, void ** outputs, void * workspace, cudaStream_t stream)
{
	return 0;
}

size_t detectron2::Yolov3NmsPlugin::getSerializationSize() const
{
	return (sizeof(int) * 7 + sizeof(float) * 7);
}

void detectron2::Yolov3NmsPlugin::serialize(void * buffer) const
{
	char *d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, mParams.nbCls);
	write(d, mParams.conf_thr);
	write(d, mParams.score_thr);
	write(d, mParams.iou_thr);
	
	for (int i = 0; i < 4; i++)
	{
		write(d, mParams.factor_scales[i]);
	}
	for (int i = 0; i < 3; i++)
	{
		write(d, mFeatureSize[i]);
	}
	for (int i = 0; i < 3; i++)
	{
		write(d, mParams.stride[i]);
	}

	ASSERT(d == a + getSerializationSize());
}

bool detectron2::Yolov3NmsPlugin::supportsFormat(DataType type, PluginFormat format) const
{
	return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char * detectron2::Yolov3NmsPlugin::getPluginType() const
{
	return "NMS_YOLOV3_TRT";
}

const char * detectron2::Yolov3NmsPlugin::getPluginVersion() const
{
	return "1";
}

IPluginV2Ext * detectron2::Yolov3NmsPlugin::clone() const
{
	IPluginV2Ext* plugin = new Yolov3NmsPlugin(mParams, mFeatureSize.data());
	return plugin;
}

void detectron2::Yolov3NmsPlugin::setPluginNamespace(const char * pluginNamespace)
{
	mPluginNamespace = pluginNamespace;
}

const char * detectron2::Yolov3NmsPlugin::getPluginNamespace() const
{
	return mPluginNamespace.c_str();
}

DataType detectron2::Yolov3NmsPlugin::getOutputDataType(int index, const nvinfer1::DataType * inputTypes, int nbInputs) const
{
	return DataType::kFLOAT;
}

bool detectron2::Yolov3NmsPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool * inputIsBroadcasted, int nbInputs) const
{
	return false;
}

bool detectron2::Yolov3NmsPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
	return false;
}

void detectron2::Yolov3NmsPlugin::configurePlugin(const Dims * inputDims, int nbInputs, const Dims * outputDims, int nbOutputs, const DataType * inputTypes, const DataType * outputTypes, const bool * inputIsBroadcast, const bool * outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
}


//----------------------------------- creator-----------------------------------

detectron2::Yolov3NmsPluginCreatorTorch::Yolov3NmsPluginCreatorTorch()
{
}

const char * detectron2::Yolov3NmsPluginCreatorTorch::getPluginName() const
{
	return NMS_YOLOV3_PLUGIN_NAME;
}

const char * detectron2::Yolov3NmsPluginCreatorTorch::getPluginVersion() const
{
	return NMS_YOLOV3_PLUGIN_VERSION;
}

const PluginFieldCollection * detectron2::Yolov3NmsPluginCreatorTorch::getFieldNames()
{
	return nullptr;
}

IPluginV2Ext * detectron2::Yolov3NmsPluginCreatorTorch::createPlugin(const char * name, const PluginFieldCollection * fc)
{
	return nullptr;
}

IPluginV2Ext * detectron2::Yolov3NmsPluginCreatorTorch::deserializePlugin(const char * name, const void * serialData, size_t serialLength)
{
	Yolov3NmsPlugin* obj = new Yolov3NmsPlugin(serialData, serialLength);
	return obj;
}

