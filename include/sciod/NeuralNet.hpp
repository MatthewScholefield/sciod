#pragma once

#include <vector>
#include <string>
#include "sciod/Layer.hpp"

#include "sciod/FloatVec.hpp"

namespace sciod
{
	class MetaNet;
	float squash(float val);
	float unsquash(float val);
	float randFloat(float min, float max);
	
	struct BackPropResult
	{
		long epoch;
		float error;
	};
	
	class NeuralNet
	{
		friend class NetState;
		friend class MetaNet;
	public:
		NeuralNet() = default;
		NeuralNet(int numInputs, int numHidden, int numHidLayers, int numOutputs);
		void create(int numInputs, int numHidden, int numHidLayers, int numOutputs);
		std::string toString() const;
		size_t getNumInputs() const;
		size_t getNumOutputs() const;
		void randomize();
		BackPropResult backPropagate(const std::vector<FloatVecIO> &vals, float maxError = 0.001f, float learningRate = 0.5f, bool debug = false);
		FloatVec2D calcProbFull(const FloatVec &inputVals) const;
		FloatVec calcProb(const FloatVec &inputVals) const;

	private:
		static std::vector<FloatVecIO> resolveConflicts(std::vector<FloatVecIO> vals);
		float backPropagateStep(const FloatVecIO &vals, float learningRate);
		float calcNode(const Layer &prevRow, const FloatVec &prevVals, int id) const;
		FloatVec calcLayerOutputs(const Layer &prevRow, const FloatVec &prevVals) const;

		std::vector<Layer> layers;
		static MetaNet metaNet;
		static bool mustTrain;
	};
}
