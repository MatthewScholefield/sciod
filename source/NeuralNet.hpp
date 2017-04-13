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

#pragma once

#include <vector>
#include "Row.hpp"

#include "FloatVec.hpp"

class NeuralNet
{
public:
	NeuralNet(int numInputs, int numHidden, int numLayers, int numOutputs);
	int getNumInputs() const;
	int getNumOutputs() const;
	void randomize();
	long backPropogate(const std::vector<FloatVecIO> &vals, float maxError = 0.001f, float learningRate = 0.5f);
	FloatVec2D calcProbFull(const FloatVec &inputVals) const;
	FloatVec calcProb(const FloatVec &inputVals) const;

private:
	float backPropogateStep(const FloatVecIO &vals, float learningRate);
	static float squash(float val);
	float calcNode(const Row &prevRow, const FloatVec &prevVals, int id) const;
	FloatVec calcNextVals(const Row &prevRow, const FloatVec &prevVals) const;

	std::vector<Row> layers;
};
