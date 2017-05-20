#include <vector>
#include <iostream>
#include "catch.hpp"
#include "sciod/NeuralNet.hpp"
#include "sciod/FloatVec.hpp"
#include "sciod/MetaNet.hpp"

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

TEST_CASE("Performance Test", "[performance-test]")
{
	const int seed = 1;
	srand(seed);
	const size_t inSize = 2 + rand() % 10;
	const size_t outSize = 1 + rand() % 3;
	const size_t hiddenSize = 5;//1 + inSize / 2;
	const size_t numHidden = 1;
	const size_t dataCount = 5;
	
	vector<FloatVecIO> data = {
		{
			{0,0},{0}
		},
		{
			{0,1},{1}
		},
		{
			{1,0},{1}
		},
		{
			{1,1},{0}
		}
	};/*(dataCount);

	for (auto &i : data)
	{
		i.in.resize(inSize);
		for (auto &j : i.in)
			j = randFloat(0.f, 1.f);

		i.out.resize(outSize);
		for (auto &j : i.out)
			j = randFloat(0.f, 1.f);
	}*/
	
	NeuralNet withMetaNet(data[0].in.size(), hiddenSize, numHidden, data[0].out.size());
	withMetaNet.randomize();
	NeuralNet withoutMetaNet(withMetaNet);

	MetaNet meta(MetaNet::general);
	meta.train();
	
	cout << "Testing normal..."  << endl;
	auto withoutMeta = withoutMetaNet.backPropagate(data, 0.1f, 4.f, true);
	cout << "err: " << withoutMeta.error << endl;
	
	REQUIRE(withoutMeta.error <= 0.1f);

	cout << "Testing meta..." << endl;
	auto withMeta = meta.backProp(withMetaNet, data, 0.1f, true);
	
	REQUIRE(withMeta.epoch < withoutMeta.epoch);
	REQUIRE(withMeta.error <= 0.1f);
}
