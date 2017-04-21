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
#include "catch.hpp"
#include "NeuralNet.hpp"
#include "FloatVec.hpp"

using namespace std;

TEST_CASE("Simple 1", "[simple-1]")
{
	const vector<FloatVecIO> testData = {
		{
			{0, 0},
			{0}
		},
		{
			{0, 1},
			{1}
		},
		{
			{1, 0},
			{1}
		},
		{
			{1, 1},
			{0}
		}
	};
	const float maxError = 0.0001f;
	const float learningRate = 4.f;
	const int hiddenSize = 5, hiddenLayers = 1;

	NeuralNet net(testData[0].in.size(), hiddenSize, hiddenLayers,
				testData[0].out.size());
	srand(time(nullptr));
	net.randomize();
	net.backPropagate(testData, maxError, learningRate);

	float error = 0.f;
	for (auto &vecIO : testData)
	{
		auto calcOut = net.calcProb(vecIO.in);
		for (int i = 0; i < vecIO.out.size(); ++i)
			error += (0.5f * pow(calcOut[i] - vecIO.out[i], 2.f));
	}
	REQUIRE(error < maxError);
}
