#include <vector>
#include "catch.hpp"
#include "sciod/NeuralNet.hpp"
#include "sciod/FloatVec.hpp"

using namespace std;
using namespace sciod;

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
		for (size_t i = 0; i < vecIO.out.size(); ++i)
			error += (0.5f * pow(calcOut[i] - vecIO.out[i], 2.f));
	}
	REQUIRE(error < maxError);
}
