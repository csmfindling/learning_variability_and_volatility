
############################################################################################
#  		      	Compilation file				         
#											                       
############################################################################################

# path to boost library : TO BE MODIFIED BY USER
path='/usr/local/boost_1_59_0/'

# compilation 
cd cstvol_backward
rm smc_c.so
python setup.py build_ext --inplace --include-dirs=$path
cd ../cstvol_forward/
rm smc_c.so
python setup.py build_ext --inplace --include-dirs=$path
cd ../varvol_forward/
rm smc_c.so
python setup.py build_ext --inplace --include-dirs=$path
cd ../varvol_backward/
rm smc_c.so
python setup.py build_ext --inplace --include-dirs=$path
cd ../noisy_forward/
rm smc_c.so
python setup.py build_ext --inplace --include-dirs=$path
cd ../probe/noisy
rm smc_c.so
python setup.py build_ext --inplace --include-dirs=$path
cd ../noiseless
rm smc_c.so
python setup.py build_ext --inplace --include-dirs=$path