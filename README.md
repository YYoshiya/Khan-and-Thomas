This project is currently under development and may be incomplete or unstable.


The DeepHAM_solution code directly applies the DeepHAM method, but I found it doesn't perform well because it requires discretization of the adjustment cost, which means that the Law of Large Numbers (LLN) is not fully effective. 
On the other hand, the VFI_NN_solution extracts a generalized moment framework from DeepHAM and solves it in a manner similar to Khan and Thomas (2008) itself. I approximate value, policy, forecasting rule(dist-dist' and dist-price) using neural network.
