#include <vector>
#include <cassert>
#include <limits>
#include <cmath>
#include <iostream>
#include "sciod/MetaNet.hpp"

using namespace std;

namespace sciod
{

NetState::NetState(const NeuralNet &net) : net(net), avErrs((size_t) (1.f / avErrResolution), -1.f) { }

FloatVec NetState::createInputs(float error)
{
	FloatVec inputs;

	auto addVar = [&inputs](float val, float min, float max)
	{
		for (float div = min; div < max; div *= NetState::resolutionGrowth)
			inputs.push_back(sciod::squash(val / div));
	};

	addVar(net.getNumInputs(), 1.f, 100.f);
	addVar(net.getNumOutputs(), 1.f, 50.f);

	size_t numNodes = 0;
	for (auto &i : net.layers)
		numNodes += i.numNodes();
	numNodes += net.layers[0].numPrevNodes();
	addVar(numNodes, 4.f, 800.f);

	assert(net.layers.size() > 1);
	addVar(net.layers.size() - 1, 1.f, 10.f);
	addVar(net.layers[0].numNodes(), 1.f, 10.f);

	addVar(epoch, 1.f, 10000.f);
	++epoch;

	addVar(prevErr - prevPrevErr, 1.f, 10.f);
	addVar(abs(prevErr - prevPrevErr), 1.f, 10.f);
	prevPrevErr = prevErr;
	prevErr = error;

	for (size_t i = 0; i < avErrs.size(); ++i)
	{
		addVar(avErrs[i], 1.f, 100.f);
		addVar(avErrs[i] / net.getNumOutputs(), 1.f, 100.f);

		float weight = i / (float) (avErrs.size() - 1);
		avErrs[i] = avErrs[i] * weight + error * (1.f - weight);
	}
	return inputs;
}

long NetState::getEpoch()
{
	return epoch;
}

MetaNet::MetaNet(Type type) : type(type) { }

void MetaNet::train()
{
	vector<FloatVecIO> trainingData;
	/*for (size_t numInputs = 2; numInputs <= 10; numInputs += 3)
		for (size_t numOutputs = 1; numOutputs <= 5; numOutputs += 2)
			for (size_t numHidden = max((size_t) 1, (numInputs + numOutputs + 8 / 2) / 8);
					numHidden <= (3 * (numInputs + numOutputs) + 8 / 2) / 8;
					numHidden += max((size_t) 1, (numInputs + numOutputs + 8 / 2) / 8))
				for (size_t numHiddenLayers = 1; numHiddenLayers <= 5; ++numHiddenLayers)*/

	for (int testNum = 0; testNum < 5; ++testNum)
	{
		const size_t numInputs = 2 + rand() % 10;
		const size_t numOutputs = 1 + rand() % 3;
		const size_t numHidden = 1 + numInputs / 2 + rand()%3;
		const size_t numHiddenLayers = 3;
		vector<FloatVecIO> randData(numTestData);
		for (FloatVecIO &vecIO : randData)
		{
			vecIO.in.resize(numInputs);
			for (size_t i = 0; i < numInputs; ++i)
				vecIO.in[i] = randFloat(0.f, 1.f);

			vecIO.out.resize(numOutputs);
			for (size_t i = 0; i < numOutputs; ++i)
				vecIO.out[i] = randFloat(0.f, 1.f);
		}

		NeuralNet exNet(numInputs, numHidden, numHiddenLayers, numOutputs);
		exNet.randomize();
		NetState state(exNet);
		while (1)
		{
			float bestError = numeric_limits<float>::max();
			float bestAlpha = 1.f;

			const float numSteps = 1000;
			const float minAlpha = 0.0001f;
			for (float alpha = minAlpha; alpha < 50.f; alpha *= 2.f)
			{
				NeuralNet netClone = exNet;
				float error = 0.f;
				for (int numTimes = 0; numTimes < numSteps; ++numTimes)
					for (FloatVecIO &i : randData)
						netClone.backPropagateStep(i, alpha);
				for (FloatVecIO &i : randData)
					error += netClone.backPropagateStep(i, alpha);
				//cout << "Alpha " << alpha << ": " << error << endl;
				if (error < bestError)
				{
					bestError = error;
					bestAlpha = alpha;
				}
			}
			float error = bestError;
			for (int numTimes = 0; numTimes < numSteps; ++numTimes)
			{
				float stepError = 0.f;
				for (FloatVecIO &i : randData)
					stepError += exNet.backPropagateStep(i, bestAlpha);
				state.createInputs(stepError);
			}
			trainingData.emplace_back(state.createInputs(error),
									FloatVecSingle(squash(bestAlpha)));

			cout << state.getEpoch() << "-" << error << ", " << bestAlpha << endl;
			if (bestAlpha <= 4.f * minAlpha || error < 0.0001f)
				break;
		}
	}
	cout << "Got data of size " << trainingData.size() << " with " << trainingData[0].in.size() << " inputs" << endl;
	metaNet.create(trainingData[0].in.size(), 30, 3, trainingData[0].out.size());
	metaNet.randomize();
	metaNet.backPropagate(trainingData, 0.5f, 1.f, true);
}

BackPropResult MetaNet::backProp(NeuralNet &net, const vector<FloatVecIO> &data, float maxError, bool debug)
{
	const float minDiff = 0.000001f;
	const float avErrWeight = 1.f - 5.f * maxError;
	float avErr = 0.f;

	const auto &adjVals = NeuralNet::resolveConflicts(data);

	NetState state(net);
	float alpha = 0.5f;
	float error;
	while (1)
	{
		error = 0.f;
		for (const FloatVecIO &i : adjVals)
			error += net.backPropagateStep(i, alpha);
		alpha = unsquash(metaNet.calcProb(state.createInputs(error))[0]);

		if (debug && state.getEpoch() % 1024 == 1)
			cout << state.getEpoch() << "-" << alpha << ": " << error << endl;

		if (error <= maxError || abs(avErr - error) < minDiff)
			break;

		avErr = avErrWeight * avErr + (1 - avErrWeight) * error;
	}
	return {state.getEpoch(), error};
}

}
