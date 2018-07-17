<h1> The virtues of computational learning noise in volatile environments </h1>

This is a project realised at Ecole Normale Supérieure by Charles Findling under the supervisions of Nicolas Chopin and Etienne Koechlin.

<h3> Link to the paper </h3>

Briefly, the paper investigates the virtues of computational learning noise in volatile environments and shows it provides adaptive features.

<h3> Summary of the code </h3>

This code provides all models used in the paper:
* The computational varying volatility model
* The computational constant volatility model
* The algorithmic varying volatility model
* The algorithmic constant volatility model
* The algorithmic noise model
* The reinforcement model
* The noise-free PROBE model
* The noisy PROBE model

The volatility models as well as the PROBE models have their main workers coded in C++.

General arhitecture of the code:
* lib_c - containes C files for the volatility models and PROBE models
* fit_functions - contains fit functions for all volatility models and RL model
* simulation_functions - contains simulation functions for all volatility models and RL model
* utils - contains util python functions

<h3> Code compilation </h3>

To compile the c++ libraries, you will need to install the boost c++ library, version 1.59 - https://www.boost.org/users/history/version_1_59_0.html

Once downloaded, open the compile_c.sh file. Modify it by adding your boost library path. Then launch ./lib_c/compile_c.sh.

<h3> Enquiries </h3>

If you have any questions on this code or related, please do not hesitate to contact me: charles.findling(at)gmail.com

<h3> References </h3>



