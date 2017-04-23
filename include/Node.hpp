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
#include <cstdlib>

#include "FloatVec.hpp"

namespace sciod
{

	/*
	 * Represents a single point
	 * Contains connection weights to the previous nodes
	 */
	class Node
	{
	public:
		Node(int numConnections);
		size_t numLinks() const;
		void randomize();
		float getBias() const;
		void updateBias(float output, float learningRate);
		float &getLinkRef(size_t id);
		float getLink(size_t id) const;

	private:
		float bias;
		FloatVec weights;
	};
}
