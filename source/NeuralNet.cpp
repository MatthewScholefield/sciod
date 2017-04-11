/*
 * Copyright (C) 2017 Matthew D. Scholefield
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <cassert>
#include <valarray>
#include "NeuralNet.hpp"

using namespace std;

NeuralNet::NeuralNet(int numInputs, int numHidden, int numLayers, int numOutputs) :
numInputs(numInputs), numHidden(numHidden), numLayers(numLayers),
numOutputs(numOutputs), inputs(numInputs, numHidden),
hiddenLayers(numLayers - 1, Row(numHidden, numHidden))
{
	hiddenLayers.emplace_back(numHidden, numOutputs);
}

void NeuralNet::randomize()
{
	inputs.randomize();
	for (auto &i : hiddenLayers)
		i.randomize();
}

float NeuralNet::squash(float val)
{
	return 1.f / (1 + exp(-val));
}

float NeuralNet::calcNode(const Row &prevRow, const vector<float> &prevVals, int id)
{
	float activation = 0.f;
	for (int srcId = 0; srcId < prevVals.size(); ++srcId)
		activation += prevVals[srcId] * prevRow.getLink(srcId, id);
	return activation;
}

vector<float> NeuralNet::calcNextVals(const Row &prevRow, const vector<float> &prevVals, int numNext)
{
	vector<float> nextInputs(numNext, 0.f);
	for (int id = 0; id < numNext; ++id)
		nextInputs[id] = squash(calcNode(prevRow, prevVals, id));
	return nextInputs;
}

vector<float> NeuralNet::calcProb(const vector<float> &inputVals)
{
	assert(inputVals.size() == numInputs);
	vector<float> vals = calcNextVals(inputs, inputVals, numInputs);
	for (int layerId = 0; layerId < numLayers - 1; ++layerId)
		vals = calcNextVals(hiddenLayers[layerId], vals, numHidden);
	vals = calcNextVals(hiddenLayers[numLayers - 1], vals, numOutputs);
	return vals;
}
