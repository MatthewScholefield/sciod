#include <iostream>
#include <ctime>
#include <vector>
#include <cassert>
#include <valarray>
#include <sstream>
#include "sciod/NeuralNet.hpp"

using namespace std;

namespace sciod
{

float squash(float val)
{
	return 1.f / (1 + exp(-val));
}

NeuralNet::NeuralNet(int numInputs, int numHidden, int numHidLayers, int numOutputs)
{
	create(numInputs, numHidden, numHidLayers, numOutputs);
}

void NeuralNet::create(int numInputs, int numHidden, int numHidLayers, int numOutputs)
{
	layers.clear();
	layers.emplace_back(numInputs, numHidden);
	for (int i = 0; i < numHidLayers - 1; ++i)
		layers.emplace_back(numHidden, numHidden);
	layers.emplace_back(numHidden, numOutputs);
}

string NeuralNet::toString() const
{
	const char startChar = 'a';
	const char endChar = 'z';
	const int numChars = endChar - startChar;

	/*
	 * Returns a string unique to that position
	 */
	auto nodeStr = [&](int layerId, int nodeNum) -> string
	{
		int diff = min(layerId, 1) * getNumInputs() + max(0, layerId - 1) * layers[1].numNodes() + nodeNum;
		if (diff > numChars)
			return string() + char(startChar + diff / numChars - 1) + char(startChar + diff % numChars);
		return string() + char(startChar + diff);
	};

	stringstream ss;
	bool moreNodes = true;
	for (int nodeNum = 0; moreNodes; ++nodeNum)
	{
		moreNodes = false;
		for (size_t layerId = 0; layerId <= layers.size(); ++layerId)
		{
			ss << '\t';
			int size = layerId == 0 ? getNumInputs() : layers[layerId - 1].numNodes();
			if (nodeNum < size)
			{
				moreNodes = true;
				ss << nodeStr(layerId, nodeNum);
			}
		}
		ss << endl;
	}
	ss << endl;

	for (size_t layerId = 0; layerId < layers.size(); ++layerId)
	{
		auto &row = layers[layerId];
		for (size_t src = 0; src < row.numPrevNodes(); ++src)
			for (size_t dest = 0; dest < row.numNodes(); ++dest)
				ss << nodeStr(layerId, src) << " - " << nodeStr(layerId + 1, dest) << ": "
				<< row.getLink(src, dest) << endl;
		ss << "== bias " << (layerId + 1) << " ==" << endl;
		for (size_t dest = 0; dest < row.numNodes(); ++dest)
			ss << "b" << dest << ": " << row.getBias(dest) << endl;
	}
	return ss.str();
}

size_t NeuralNet::getNumInputs() const
{
	return layers.size() == 0 ? 0 : layers[0].numPrevNodes();
}

size_t NeuralNet::getNumOutputs() const
{
	return layers.size() == 0 ? 0 : layers.back().numNodes();
}

void NeuralNet::randomize()
{
	for (auto &i : layers)
		i.randomize();
}

// TODO: Move to Row class

float NeuralNet::calcNode(const Layer &row, const FloatVec &prevVals, int dest) const
{
	float activation = 0.f;
	assert(prevVals.size() == row.numPrevNodes());
	for (size_t src = 0; src < prevVals.size(); ++src)
		activation += prevVals[src] * row.getLink(src, dest);
	activation += row.getBias(dest);
	return activation;
}

FloatVec NeuralNet::calcLayerOutputs(const Layer &row, const FloatVec &prevVals) const
{
	FloatVec nextVals(row.numNodes(), 0.f);
	for (size_t dest = 0; dest < row.numNodes(); ++dest)
		nextVals[dest] = squash(calcNode(row, prevVals, dest));
	return nextVals;
}

// Returns initial error

float NeuralNet::backPropagateStep(const FloatVecIO &vals, float learningRate)
{

	assert(vals.out.size() == layers.back().numNodes());
	FloatVec2D nodeProb = calcProbFull(vals.in);
	FloatVec2D actDeriv = nodeProb; // Assign to get correct size. Must reassign later

	float error = 0.f; // Only for return value

	assert(nodeProb.size() == 1 + layers.size());

	// Calculate activation derivatives for last row
	{
		int layerId = nodeProb.size() - 1;
		for (size_t src = 0; src < nodeProb[layerId].size(); ++src)
		{
			float out = nodeProb[layerId][src];
			float correct = vals.out[src];
			float act = (out - correct) * out * (1 - out);
			actDeriv[layerId][src] = act;

			float diff = (out - correct);
			error += diff * diff / 2.f;
		}
	}

	// Calculate for all other rows
	for (int layerId = nodeProb.size() - 2; layerId >= 0; --layerId)
	{
		Layer &row = layers[layerId];
		for (size_t src = 0; src < row.numPrevNodes(); ++src)
		{
			float chainSums = 0.f;
			for (size_t dest = 0; dest < row.numNodes(); ++dest)
				chainSums += row.getLink(src, dest) * actDeriv[layerId + 1][dest];
			float out = nodeProb[layerId][src];
			actDeriv[layerId][src] = out * (1 - out) * chainSums;
		}
	}

	// Use deriv calculations to adjust link weights
	for (int layerId = nodeProb.size() - 2; layerId >= 0; --layerId)
	{
		Layer &row = layers[layerId];
		row.updateBiases(actDeriv[layerId + 1], learningRate * 0.75f);
		for (size_t src = 0; src < row.numPrevNodes(); ++src)
		{
			for (size_t dest = 0; dest < row.numNodes(); ++dest)
			{
				float &link = row.getLinkRef(src, dest);
				float deriv = nodeProb[layerId][src] * actDeriv[layerId + 1][dest];
				link -= learningRate * deriv;
			}
		}
	}

	// Calculate error for return value
	return error;
}

vector<FloatVecIO> NeuralNet::resolveConflicts(vector<FloatVecIO> vals)
{
	for (auto it = vals.begin(); it != vals.end(); ++it)
		for (auto jt = it + 1; jt != vals.end();)
		{
			if (it->in == jt->in)
			{
				for (size_t i = 0; i < jt->out.size(); ++i)
					if (jt->out[i] > it->out[i])
						it->out[i] = jt->out[i];
				jt = vals.erase(jt);
				continue;
			}
			++jt;
		}
	return vals;
}

/* 
 * Returns Epoch
 */
BackPropResult NeuralNet::backPropagate(const vector<FloatVecIO> &vals, float maxError, float learningRate, bool debug)
{
	const float minDiff = 0.000001f;
	const float avErrWeight = 1.f - 5.f * maxError;
	float avErr = 0.f;
	long epoch = 0;

	const auto &adjVals = resolveConflicts(vals);

	while (1)
	{
		++epoch;

		float err = 0.f;
		for (auto &i : adjVals)
			err += backPropagateStep(i, learningRate);

		if (debug && epoch % 1024 == 0)
			cout << "Error: " << err << endl;

		if (err < maxError || abs(avErr - err) < minDiff)
			return {epoch, err};

		avErr = avErrWeight * avErr + (1 - avErrWeight) * err;
	}
}

FloatVec2D NeuralNet::calcProbFull(const FloatVec &inputVals) const
{
	FloatVec2D vec2D;
	vec2D.push_back(inputVals);
	for (auto &i : layers)
		vec2D.push_back(calcLayerOutputs(i, vec2D.back()));
	return vec2D;
}

FloatVec NeuralNet::calcProb(const FloatVec &inputVals) const
{
	FloatVec vals = inputVals;
	for (auto &i : layers)
		vals = calcLayerOutputs(i, vals);
	return vals;
}

}