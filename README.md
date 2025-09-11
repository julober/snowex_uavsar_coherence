
## Final project for GEOG6655

This is a structured package for my final project. 

### Structure

`scripts` contains any code that I write to run scripts
    this includes job scripts that I give to slurm for running things on the supercomputer
`outputs` contains `manuscript` for writing and `figures` for figures. 
`data/` contains a script for downloading the data needed for the project, but not the 
    data itself, as that would be much too big. 
    i may create `sample_data` which does sync one or two files that I can use to show 
    a sample analysis. 

### Environment

There is an ENVIRONMENT.yml file included here. I use this to create a conda environment that 
I activate to install needed packages and versions. 

I plan to have clones of this package on CURC and maybe the BSU supercomputer, but will have to 
build and environment in each of those spaces. On the supercomputers, I probably want to use the 
`-p ./env` flag to install the environment into the local directory, where there is probably more 
space than on my home directory. 
