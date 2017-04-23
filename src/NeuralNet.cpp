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

#include <iostream>
#include <ctime>
#include <vector>
#include <cassert>
#include <valarray>
#include <sstream>
#include "NeuralNet.hpp"

using namespace std;

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
	auto nodeStr = [&](int rowId, int nodeNum) -> string
	{
		int diff = min(rowId, 1) * getNumInputs() + max(0, rowId - 1) * layers[1].numNodes() + nodeNum;
		if (diff > numChars)
			return string() + char(startChar + diff / numChars - 1) + char(startChar + diff % numChars);
		return string() + char(startChar + diff);
	};

	stringstream ss;
	bool moreNodes = true;
	for (int nodeNum = 0; moreNodes; ++nodeNum)
	{
		moreNodes = false;
		for (int rowId = 0; rowId < layers.size(); ++rowId)
		{
			ss << '\t';
			int size = rowId == 0 ? getNumInputs() : layers[rowId].numNodes();
			if (nodeNum < size)
			{
				moreNodes = true;
				ss << nodeStr(rowId, nodeNum);
			}
		}
		ss << endl;
	}
	ss << endl;

	for (int rowId = 0; rowId < layers.size(); ++rowId)
	{
		auto &row = layers[rowId];
		for (int src = 0; src < row.numPrevNodes(); ++src)
			for (int dest = 0; dest < row.numNodes(); ++dest)
				ss << nodeStr(rowId, src) << " - " << nodeStr(rowId + 1, dest) << ": "
				<< row.getLink(src, dest) << endl;
	}
	return ss.str();
}

int NeuralNet::getNumInputs() const
{
	assert(layers.size() > 0);
	return layers[0].numPrevNodes();
}

int NeuralNet::getNumOutputs() const
{
	assert(layers.size() > 0);
	return layers.back().numNodes();
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

// TODO: Move to Row class
float NeuralNet::calcNode(const Layer &row, const FloatVec &prevVals, int destId) const
{
	float activation = 0.f;
	assert(prevVals.size() == row.numPrevNodes());
	for (int srcId = 0; srcId < prevVals.size(); ++srcId)
		activation += prevVals[srcId] * row.getLink(srcId, destId);
	activation += row.getBias(destId);
	return activation;
}

FloatVec NeuralNet::calcLayerOutputs(const Layer &row, const FloatVec &prevVals) const
{
	FloatVec nextVals(row.numNodes(), 0.f);
	for (int destId = 0; destId < row.numNodes(); ++destId)
		nextVals[destId] = squash(calcNode(row, prevVals, destId));
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
		int rowId = nodeProb.size() - 1;
		for (int nodeNum = 0; nodeNum < nodeProb[rowId].size(); ++nodeNum)
		{
			float out = nodeProb[rowId][nodeNum];
			float correct = vals.out[nodeNum];
			float act = (out - correct) * out * (1 - out);
			actDeriv[rowId][nodeNum] = act;

			float diff = (out - correct);
			error += diff * diff / 2.f;
		}
	}

	// Calculate for all other rows
	for (int rowId = nodeProb.size() - 2; rowId >= 0; --rowId)
	{
		Layer &row = layers[rowId];
		for (int srcNode = 0; srcNode < row.numPrevNodes(); ++srcNode)
		{
			float chainSums = 0.f;
			for (int destNode = 0; destNode < row.numNodes(); ++destNode)
				chainSums += row.getLink(srcNode, destNode) * actDeriv[rowId + 1][destNode];
			float out = nodeProb[rowId][srcNode];
			actDeriv[rowId][srcNode] = out * (1 - out) * chainSums;
		}
	}

	// Use deriv calculations to adjust link weights
	for (int rowId = nodeProb.size() - 2; rowId >= 0; --rowId)
	{
		Layer &row = layers[rowId];
		row.updateBiases(actDeriv[rowId+1], learningRate * 0.75f);
		for (int src = 0; src < row.numPrevNodes(); ++src)
		{
			for (int dest = 0; dest < row.numNodes(); ++dest)
			{
				float &link = row.getLinkRef(src, dest);
				float deriv = nodeProb[rowId][src] * actDeriv[rowId+1][dest];
				link -= learningRate * deriv;
			}
		}
	}

	// Calculate error for return value
	return error;
}

/* 
 * Returns Epoch
 */
long NeuralNet::backPropagate(const vector<FloatVecIO> &vals, float maxError, float learningRate)
{
	long epoch = 0;
	while (1)
	{
		++epoch;

		float error = 0.f;
		for (auto &i : vals)
			error += backPropagateStep(i, learningRate);

		if (error < maxError)
			return epoch;
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