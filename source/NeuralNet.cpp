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

NeuralNet::NeuralNet(int numInputs, int numHidden, int numHidLayers, int numOutputs)
{
	layers.emplace_back(numInputs, numHidden);
	for (int i = 0; i < numHidLayers - 1; ++i)
		layers.emplace_back(numHidden, numHidden);
	layers.emplace_back(numHidden, numOutputs);
}

void NeuralNet::randomize()
{
	for (auto &i : layers)
		i.randomize();
}

float NeuralNet::squash(float val)
{
	return 1.f / (1 + exp(-val));
}

float NeuralNet::calcNode(const Row &prevRow, const FloatVec &prevVals, int id)
{
	float activation = 0.f;
	assert(prevVals.size() == prevRow.numNodes());
	for (int srcId = 0; srcId < prevVals.size(); ++srcId)
		activation += prevVals[srcId] * prevRow.getLink(srcId, id);
	return activation;
}

FloatVec NeuralNet::calcNextVals(const Row &prevRow, const FloatVec &prevVals)
{
	FloatVec nextVals(prevRow.numNextNodes(), 0.f);
	for (int id = 0; id < prevRow.numNextNodes(); ++id)
		nextVals[id] = squash(calcNode(prevRow, prevVals, id));
	return nextVals;
}

FloatVec NeuralNet::calcProb(const FloatVec &inputVals)
{
	FloatVec vals = inputVals;
	for (int i = 0; i < layers.size(); ++i)
		vals = calcNextVals(layers[i], vals);
	return vals;
}
