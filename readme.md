# Desciption

Best practices to code on cuda introduced in ipm HPC workshop 2019. 

files contain 'basic' in their name are basic codes which their performance is not good when running.


## Compile

to comile use `nvcc` compiler.


## Input Error!

to run output files be aware that you should specify a number as argument in order to specify array size for AddFunctions and three numbers m, n, k for matrix multiplications which specify dimension of the 2 matrices (mxn) and (nxm).


## Compile AtomicInstruction files

to compile atomic instructions use following commmand
`$ nvcc file.cu -arch sm_75`

where `sm_75` should be based on your gpu architcture.


## Compile cuBLAS and cuRand

`$ nvcc file.cu -lcurand -lcublas`

