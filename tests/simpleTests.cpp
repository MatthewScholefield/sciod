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
#include "catch.hpp"
#include "source/NeuralNet.hpp"

using namespace std;

TEST_CASE("Simple 1", "[simple-1]")
{
	NeuralNet net(2,2,2,3);
	auto prob = net.calcProb({0.0f, 1.0f});
	stringstream ss;
	ss << "{" << prob[0];
	for (auto it = prob.begin() + 1; it != prob.end(); ++it)
		ss << ", " << *it;
	ss << "}" << endl;
	cout << ss.str() << endl;
}
