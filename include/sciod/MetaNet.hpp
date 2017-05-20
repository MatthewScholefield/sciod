#pragma once

#include <vector>
#include "NeuralNet.hpp"


namespace sciod
{
	class NetState
	{
	public:
		NetState(const NeuralNet &net);
		FloatVec createInputs(float error);
		long getEpoch();
		
	private:
		const NeuralNet &net;
		
		constexpr static float resolutionGrowth = 2.f;
		long epoch = 0;
		float prevErr;
		float prevPrevErr;
		
		constexpr static float avErrResolution = 0.1f;
		FloatVec avErrs;
	};

	class MetaNet
	{
	public:

		enum Type
		{
			general
		};

		MetaNet(Type type);
		void train();
		BackPropResult backProp(NeuralNet &net, const std::vector<FloatVecIO> &data, float maxError, bool debug = false);

	private:
		static constexpr float metaError = 0.1f;
		static const size_t numTestData = 10;
		Type type;
		NeuralNet metaNet;
	};
}
