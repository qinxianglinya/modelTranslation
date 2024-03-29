/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_TORCH_NMS_PLUGIN_H
#define TRT_TORCH_NMS_PLUGIN_H
#include "kernel.h"
//#include "nmsUtils.h"
#include "plugin.h"
#include <string>
#include <vector>
#include "nmsUtils.h"

using namespace nvinfer1::plugin;

namespace detectronPlugin
{

class DetectionOutput : public IPluginV2Ext
{
public:

	DetectionOutput(DetectionOutputParametersTorch params, const int* featureSizeIn, const int* topkCandidateIn);

    //DetectionOutput(int keepTopk, int topK, int srcW, int srcH, int targetW, int targetH, float scoreThreshold, float iouThreshold);

	//DetectionOutput(DetectionOutputParametersTorch params, float scoreThreshold, float iouThreshold, int srcW, int srcH, int targetW, int targetH);

    DetectionOutput(const void* data, size_t length);

    ~DetectionOutput() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    void detachFromContext() override;

private:
    DetectionOutputParametersTorch mParam;
	std::vector<int> mFeatureSize;
	std::vector<int> mTopkCandidates;

	//float mScoreThreshold, mIouThreshold;
    //int mNumPriors;
    std::string mPluginNamespace;
};
class DetectionOutputCreator : public BaseCreator
{
public:
	DetectionOutputCreator();

	~DetectionOutputCreator() {};

	const char* getPluginName() const override;

	const char* getPluginVersion() const override;

	const PluginFieldCollection* getFieldNames() override;

	IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

	IPluginV2Ext* deserializePlugin(const char* name, const void* data, size_t length) override;

};

} // namespace plugin

#endif // TRT_NMS_PLUGIN_H
