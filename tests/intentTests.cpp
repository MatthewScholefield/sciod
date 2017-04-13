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

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include "catch.hpp"
#include "source/NeuralNet.hpp"

using namespace std;

struct IntentLines
{
	string name;
	vector<string> lines;
};

vector<IntentLines> intentLines = {
	{
		"hello",
		{
			"hello",
			"hi there",
			"how are you doing",
			"whats up",
			"hows it going",
			"hey there",
			"hi"
		}
	},
	{
		"good bye",
		{
			"see you later",
			"goodbye",
			"see you tomorrow",
			"bye",
			"ill see you another time",
			"later",
			"goodnight",
			"good night",
			"talk to you soon"
		},
	},
	{
		"weather",
		{
			"is it raining",
			"is it pouring",
			"is it snowing",
			"whats the weather",
			"whats the temperature",
			"what is the temp",
			"what is the temperature",
			"is it cold outside",
			"is it windy outside",
			"hows the weather",
			"is it cloudy",
			"is it sunny right now",
			"will it rain later today",
			"is rain expected tonight"
		}
	}
};

vector<string> split(const string &str, const string &delim)
{
	std::vector<std::string> v;

	std::size_t prev_pos = 0, pos;
	while ((pos = str.find_first_of(delim, prev_pos)) != std::string::npos)
	{
		if (pos > prev_pos)
			v.push_back(str.substr(prev_pos, pos - prev_pos));
		prev_pos = pos + 1;
	}
	if (prev_pos < str.length())
		v.push_back(str.substr(prev_pos, std::string::npos));

	return v;
}

using Dict = map<string, size_t>;

FloatVec createInputs(const Dict &dict, const string &str)
{
	float total = 0.f;
	FloatVec vec(dict.size(), 0.f);
	for (auto &word : split(str, " "))
	{
		auto result = dict.find(word);
		if (result != dict.end())
			vec[result->second] += 1.f;
		total += 1.f;
	}
	if (total != 0.f)
		for (auto &i : vec)
			i /= total;
	return vec;
}

TEST_CASE("Intent Test", "[intent-test]")
{
	srand(time(nullptr));
	vector<NeuralNet> intents;
	vector<Dict> dicts;

	for (auto &i : intentLines)
	{
		dicts.emplace_back();
		Dict &dict = dicts.back();
		size_t id = 0;
		for (auto &j : i.lines)
			for (auto &w : split(j, " "))
			{
				auto result = dict.find(w);
				if (result == dict.end())
					dict.emplace(w, id++);
			}

		intents.emplace_back(dict.size(), 8, 2, 1);
		intents.back().randomize();
		vector<FloatVecIO> vals;

		for (auto &line : i.lines)
			vals.emplace_back(createInputs(dict, line), FloatVec(1, 1.f));

		for (auto &j : intentLines)
		{
			if (j.name.compare(i.name) == 0)
				continue;
			for (auto &line : j.lines)
				vals.emplace_back(createInputs(dict, line), FloatVec(1, 0.f));
		}

		long epoch = intents.back().backPropagate(vals, 0.0001, 23);
		cout << "Epoch: " << epoch << endl;
	}

	// Interact
	cout << "Enter a query." << endl;
	cout << "(q to quit)" << endl;
	while (1)
	{
		cout << ": " << endl;
		string query;
		getline(cin, query);
		if (query.compare("q") == 0)
			break;

		assert(intents.size() == dicts.size());
		for (int i = 0; i < intents.size(); ++i)
			cout << intentLines[i].name << ": " << intents[i].calcProb(createInputs(dicts[i], query))[0] << endl;
	}
}
