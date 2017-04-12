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
#include "NeuralNet.hpp"

using namespace std;

NeuralNet::NeuralNet(int numInputs, int numHidden, int numHidLayers, int numOutputs)
{
	layers.emplace_back(numInputs, numHidden);
	for (int i = 0; i < numHidLayers - 1; ++i)
		layers.emplace_back(numHidden, numHidden);
	layers.emplace_back(numHidden, numOutputs);
}

int NeuralNet::getNumInputs() const
{
	assert(layers.size() > 0);
	return layers[0].numNodes();
}

int NeuralNet::getNumOutputs() const
{
	assert(layers.size() > 0);
	return layers.back().numNextNodes();
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

float NeuralNet::calcNode(const Row &prevRow, const FloatVec &prevVals, int id) const
{
	float activation = 0.f;
	assert(prevVals.size() == prevRow.numNodes());
	for (int srcId = 0; srcId < prevVals.size(); ++srcId)
		activation += prevVals[srcId] * prevRow.getLink(srcId, id);
	return activation;
}

FloatVec NeuralNet::calcNextVals(const Row &prevRow, const FloatVec &prevVals) const
{
	FloatVec nextVals(prevRow.numNextNodes(), 0.f);
	for (int id = 0; id < prevRow.numNextNodes(); ++id)
		nextVals[id] = squash(calcNode(prevRow, prevVals, id));
	return nextVals;
}

// Returns initial error

float NeuralNet::backPropogateStep(const FloatVecIO &vals, float learningRate)
{
	assert(vals.out.size() == layers.back().numNextNodes());
	FloatVec2D nodeProb = calcProbFull(vals.in);
	FloatVec2D actDeriv = nodeProb; // Assign to get correct size. Must reassign later

	float error = 0.f; // Only for return value

	assert(nodeProb.size() == layers.size() + 1); // No back layer but back nodes

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
	for (int rowId = layers.size() - 1; rowId >= 0; --rowId)
	{
		Row &row = layers[rowId];
		for (int nodeNum = 0; nodeNum < row.numNodes(); ++nodeNum)
		{
			float chainSums = 0.f;
			for (int destNode = 0; destNode < row.numNextNodes(); ++destNode)
				chainSums += row.getLink(nodeNum, destNode) * actDeriv[rowId + 1][destNode];
			float out = nodeProb[rowId][nodeNum];
			actDeriv[rowId][nodeNum] = out * (1 - out) * chainSums;
		}
	}

	// Use deriv calculations to adjust link weights
	for (int rowId = layers.size() - 1; rowId >= 1; --rowId)
	{
		Row &row = layers[rowId];
		for (int src = 0; src < row.numNodes(); ++src)
		{
			for (int dest = 0; dest < row.numNextNodes(); ++dest)
			{
				float &link = row.getLinkRef(src, dest);
				float deriv = nodeProb[rowId][src] * actDeriv[rowId + 1][dest];
				link -= learningRate * deriv;
			}
		}
	}

	// Calculate error for return value
	return error;
}

void NeuralNet::backPropogate(const vector<FloatVecIO> &vals, float maxError, float initLearningRate)
{
	float learningRate = initLearningRate;
	clock_t begin = clock();
	clock_t calcStart = clock();
	float prevError = -1.f;
	while(1)
	{
		float error = 0.f;
		for (auto &i : vals)
			error += backPropogateStep(i, learningRate);
		
		if (error == prevError)
			learningRate = 2 * initLearningRate * (rand() / (float) RAND_MAX);
		prevError = error;

		clock_t end = clock();
		double elapsed = double(end - begin) / CLOCKS_PER_SEC;
		double totalElapsed = double(end - calcStart) / CLOCKS_PER_SEC;
		if (elapsed > 0.5)
		{
			cout << "Error: " << error << endl;
			begin = end;
		}
		if (totalElapsed > 100.f)
			break;

		if (error < maxError)
			break;
	}
}

FloatVec2D NeuralNet::calcProbFull(const FloatVec &inputVals) const
{
	FloatVec2D vec2D;
	vec2D.push_back(inputVals);
	for (auto &i : layers)
		vec2D.push_back(calcNextVals(i, vec2D.back()));
	return vec2D;
}

FloatVec NeuralNet::calcProb(const FloatVec &inputVals) const
{
	FloatVec vals = inputVals;
	for (int i = 0; i < layers.size(); ++i)
		vals = calcNextVals(layers[i], vals);
	return vals;
}