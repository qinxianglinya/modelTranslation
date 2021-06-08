#pragma once
#include "Engine.h"
class EngineTfRetinaNet :
	public Engine
{
public:
	EngineTfRetinaNet(UffParams params);
	~EngineTfRetinaNet();

	void initEngine();
	void setConfig(EngineUniquePtr<IBuilder> &builder, EngineUniquePtr<INetworkDefinition> &network);

private:
	UffParams mParams;
};

