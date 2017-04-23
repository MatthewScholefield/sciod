# Sciod

This library provides a very simple API to use fully connected neural networks. It trains using back propagation. Most of the library was created from [this article][1].

# API Example

Very simple demo:

```C++
NeuralNet net(1,1,1,1);
vector<FloatVecIO> vals;
for (float i = 0; i <= 1; i += 0.1f)
    vals.emplace_back(i, 1.f - i);
net.backPropagate(vals, 0.001f, 4.f);

float out = net.calcOut({0.13f})[0]; // Should be around 0.87
cout << net.toString() << endl;
```

Full example (learn XOR operator):
```C++
#include <iostream>
#include <sciod/NeuralNet.hpp>

using namespace std;

int main()
{
    /* ====== Diagram ======
     * 
     *           x
     *   o       x
     *           x      o
     *   o       x
     *           x
     * Input          Ouput
     *        Hidden
     * 
     */
    const float inputLayerSize = 2,
                hiddenLayerSize = 5,
                numHiddenLayers = 1,
                outputLayerSize = 1;
    NeuralNet net(inputLayerSize,
                  hiddenLayerSize,
                  numHiddenLayers,
                  outputLayerSize);
    
    // FloatVec: vector<float>
    // FloatVecIO: struct { FloatVec in, out; }
    
    // Inputs to train XOR operator
    vector<FloatVecIO> vals = {
        {
            {0,0}, // Inputs
            {0} // Outputs
        },
        { {1, 1}, {0} },
        { {0, 1}, {1} },
        { {1, 0}, {1} }
    };
    
    float maxError = 0.001f;
    float learningRate = 4.f;
    net.backPropagate(vals, maxError, learningRate);
    
    // Warning: This is not pretty xD
    cout << net.toString() << endl;
    
    cout << "(1.0,1.0): " << net.calcOut({1.0f, 1.0f})[0] << endl;
    cout << "(0.0,1.0): " << net.calcOut({0.0f, 1.0f})[0] << endl;
    cout << "(0.2,0.8): " << net.calcOut({0.2f, 0.8f})[0] << endl;
    
    return 0;
}
```

# Compile

Install `meson` and `ninja` (Ubuntu: `sudo apt-get install python3 ninja-build build-essential && pip3 install --user meson`)

Compile and install:
```
meson build
cd build
ninja
sudo ninja install
```

### Questions or Comments? ###

Feel free to file an issue or contact me at `matthew3311999@gmail.com`.

[1]:https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
