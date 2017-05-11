#pragma once

#include <vector>
#include <string>
#include "sciod/Layer.hpp"

#include "sciod/FloatVec.hpp"

namespace sciod
{
	float squash(float val);
	
	struct BackPropResult
	{
		long epoch;
		float error;
	};
	
	class NeuralNet
	{
	public:
		NeuralNet() = default;
		NeuralNet(int numInputs, int numHidden, int numHidLayers, int numOutputs);
		void create(int numInputs, int numHidden, int numHidLayers, int numOutputs);
		std::string toString() const;
		int getNumInputs() const;
		int getNumOutputs() const;
		void randomize();
		BackPropResult backPropagate(const std::vector<FloatVecIO> &vals, float maxError = 0.001f, float learningRate = 0.5f, bool debug = false);
		FloatVec2D calcProbFull(const FloatVec &inputVals) const;
		FloatVec calcProb(const FloatVec &inputVals) const;

	private:
		std::vector<FloatVecIO> resolveConflicts(std::vector<FloatVecIO> vals);
		float backPropagateStep(const FloatVecIO &vals, float learningRate);
		float calcNode(const Layer &prevRow, const FloatVec &prevVals, int id) const;
		FloatVec calcLayerOutputs(const Layer &prevRow, const FloatVec &prevVals) const;

		std::vector<Layer> layers;
	};
}
