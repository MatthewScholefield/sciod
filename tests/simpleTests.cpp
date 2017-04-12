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
#include <sstream>
#include <string>
#include <vector>
#include "catch.hpp"
#include "source/NeuralNet.hpp"

using namespace std;

void interact(const NeuralNet &net)
{
	cout << "Enter " << net.getNumInputs() << " inputs." << endl;
	cout << "(q to Quit)" << endl;
	while (1)
	{
		FloatVec inputs;
		while (inputs.size() < net.getNumInputs())
		{
			cout << ": ";
			string query;
			getline(cin, query);
			try
			{
				float val = stof(query);
				if (val >= 0.f && val <= 1.f)
					inputs.push_back(val);
				else
					cout << "Value must be between 0.0 and 1.0." << endl;

			}
			catch (...)
			{
				if (query.compare("q") == 0)
					return;

				cout << "Error parsing " << query << endl;
			}
		}
		auto outputs = net.calcProb(inputs);
		if (outputs.size() == 1)
			cout << outputs[0] << endl;
		else
		{
			cout << "{" << outputs[0];
			for (auto it = outputs.begin() + 1; it != outputs.end(); ++it)
				cout << ", " << *it;
			cout << "}" << endl;
		}
	}
}

TEST_CASE("Simple 1", "[simple-1]")
{
	srand(time(nullptr));
	NeuralNet net(2, 5, 1, 1);
	net.randomize();
	net.backPropogate({
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
	}, 0.0001, 4);

	interact(net);
}