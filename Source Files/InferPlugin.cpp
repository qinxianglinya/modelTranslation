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
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <stack>
#include <unordered_set>

#include "gridAnchorPlugin.h"
#include "nmsPlugin.h"
#include "resizeNearestPlugin.h"
#include "flattenConcat.h"

#include "anchorGeneratorTorch.h"
#include "nmsPluginTorch.h"
#include "Yolov3AnchorPlugin.h"
#include "Yolov3NmsPlugin.h"

namespace nvinfer1
{

namespace plugin
{
ILogger* gLogger{};

// This singleton ensures that each plugin is only registered once for a given
// namespace and type, and attempts of duplicate registration are ignored.
class PluginCreatorRegistry
{
public:
    static PluginCreatorRegistry& getInstance()
    {
        static PluginCreatorRegistry instance;
        return instance;
    }

    template <typename CreatorType>
    void addPluginCreator(void* logger, const char* libNamespace)
    {
        // Make accesses to the plugin creator registry thread safe
        std::lock_guard<std::mutex> lock(mRegistryLock);

        std::string errorMsg;
        std::string verboseMsg;

        std::unique_ptr<CreatorType> pluginCreator{new CreatorType{}};
        pluginCreator->setPluginNamespace(libNamespace);

        nvinfer1::plugin::gLogger = static_cast<nvinfer1::ILogger*>(logger);
        std::string pluginType
            = std::string(pluginCreator->getPluginNamespace()) + "::" + std::string(pluginCreator->getPluginName());

        if (mRegistryList.find(pluginType) == mRegistryList.end())
        {
            bool status = getPluginRegistry()->registerCreator(*pluginCreator, libNamespace);
            if (status)
            {
                mRegistry.push(std::move(pluginCreator));
                mRegistryList.insert(pluginType);
                verboseMsg = "Plugin creator registration succeeded - " + pluginType;
            }
            else
            {
                errorMsg = "Could not register plugin creator:  " + pluginType;
            }
        }
        else
        {
            verboseMsg = "Plugin creator already registered - " + pluginType;
        }

        if (logger)
        {
            if (!errorMsg.empty())
            {
                nvinfer1::plugin::gLogger->log(ILogger::Severity::kERROR, errorMsg.c_str());
            }
            if (!verboseMsg.empty())
            {
                nvinfer1::plugin::gLogger->log(ILogger::Severity::kVERBOSE, verboseMsg.c_str());
            }
        }
    }

    ~PluginCreatorRegistry()
    {
        std::lock_guard<std::mutex> lock(mRegistryLock);

        // Release pluginCreators in LIFO order of registration.
        while (!mRegistry.empty())
        {
            mRegistry.pop();
        }
        mRegistryList.clear();
    }

private:
    PluginCreatorRegistry() {}

    std::mutex mRegistryLock;
    std::stack<std::unique_ptr<IPluginCreator>> mRegistry;
    std::unordered_set<std::string> mRegistryList;

public:
    PluginCreatorRegistry(PluginCreatorRegistry const&) = delete;
    void operator=(PluginCreatorRegistry const&) = delete;
};

template <typename CreatorType>
void initializePlugin(void* logger, const char* libNamespace)
{
    PluginCreatorRegistry::getInstance().addPluginCreator<CreatorType>(logger, libNamespace);
}

} // namespace plugin
} // namespace nvinfer1

extern "C" {
bool initLibNvInferPlugins(void* logger, const char* libNamespace)
{

	initializePlugin<nvinfer1::plugin::GridAnchorPluginCreator>(logger, libNamespace);
	initializePlugin<nvinfer1::plugin::NMSPluginCreator>(logger, libNamespace);
	initializePlugin<nvinfer1::plugin::ResizeNearestPluginCreator>(logger, libNamespace);
	initializePlugin<nvinfer1::plugin::FlattenConcatPluginCreator>(logger, libNamespace);

	initializePlugin<detectronPlugin::GridAnchorBasePluginCreatorTorch>(logger, libNamespace);
	initializePlugin<detectronPlugin::DetectionOutputCreator>(logger, libNamespace);
	initializePlugin<detectron2::Yolov3AnchorPluginCreatorTorch>(logger, libNamespace);
	initializePlugin<detectron2::Yolov3NmsPluginCreatorTorch>(logger, libNamespace);
    return true;
}
} // extern "C"
