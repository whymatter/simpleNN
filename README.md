# simpleNN
My first neurnal network

I decided to dive into neural networking, machine learing, data science and so.  
Neural networks are an incredible and tremendously powerful tool for doing things without actually programming.  

---

__They basicly work like your brain.__

Their goal is to map a bunch of (or simply `n` as mathematicians would say) input values to a bunch of (or `m`) output values.
Every individual output value is computed by using the inputs and weigh each individually.  
*E.g. input 1 can have a big impact on output 1 but just a little on output 2.*

Lets assume we have two inputs e.g. `2` and `6` now we want a network which calculates the average of both (which is `4`).  
If we want so do this like the network does it then we have to find a numbers to weigh out inputs.  
Lets say we take `0.3` for input one and `0.8` for input two. This results in:

`output = 2 * 0.3 + 6 * 0.8`  
`=> output = 5.4`

Nope thats far away from `4`. Lets adjust our weights (neurnal networks would do this by using e.g. gradient decent)  
We do this by just moving the two closer together. Lets say weight of input one `0.4` and weight of input two `0.6`

`output = 2 * 0.4 + 6 * 0.6`  
`=> output = 4.4`

Oh yeah, thats quit good. Lets look if we can do this even better.
As we are pretty much there let us modify the weights just a bit, lets say we take `0.55` as new input two weight

`output = 2 * 0.4 + 6 * 0.55`  
`=> output = 4.1`

That look amazing! Just `0.1` apart from out goal `4`. We now have found some weights for our network!!!

This is exactly what happens inside a simple neurnal network, nothing else, no magic (err well maybe a bit ;-) )

__BUT__

But they also have some downsides, imagine we want to use our "trained" network to calculate the average of `9` and `2` (should be `5.5`).  
We do this by calculating:

`output = 9 * 0.4 + 2 * 0.55`  
`=> output = 4.7`

Damm, that differs way more than our last calculation with `2` and `6` and it gets even more strange, lets twist `9` and `2` as the average of `9` and `2` should be the same as the average of `2` and `9`:

`output = 2 * 0.4 + 9 * 0.55`
`=> output = 5.75`

This fits much better... but why? Its because neuronal networks are really sensitive to `overfitting`.
This is what has happened here: We trained our network to calculate the average of `2` and `6`, be we forgot about all the other numbers...
Our training data was to specific to one problem.
If we would diversify it we would be able to archive way better results!
But that is beyond the scope of this introduction. I do not want to calculate by hand if this can be done by a neurnal network! ..

(Well the optimized values are 0.5 and 0.5)

---

So this is my first neurnal network, it has one hidden layer and variable amount of units in the three layers.
You can train it using gradient decent with a optimized learing rate adjustment invented by me (a very simple one, using a moving average over the last `n` error reductions and adjusting the learning rate accodringly)  
It calculates the average of two numbers between `0` and `1` using `3` hidden layers (well maybe it would be better to use no hidden layer...)
There is also one example of my minimization to demonstrate it a bit in two dimensions.
