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

class NeuralNet
{
public:
	NeuralNet(int numInputs, int numHidden, int numOutputs, int numLayers);
	void randomize();
	std::vector<float> calcProb(const std::vector<float> &inputVals);

private:
	static float squash(float val);
	float calcNode(const Row &prevRow, const std::vector<float> &prevVals, int id);
	std::vector<float> calcNextVals(const Row &prevRow, const std::vector<float> &prevVals, int numNext);
	
	int numInputs, numHidden, numOutputs, numLayers;
	Row inputs;
	std::vector<Row> hiddenLayers;
};
