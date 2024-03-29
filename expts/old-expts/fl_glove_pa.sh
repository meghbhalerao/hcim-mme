######START OF EMBEDDED SGE COMMANDS ##########################
#$ -S /bin/bash
#$ -cwd
#$ -N fl_glove_pa
#$ -M 16ee234.megh@nitk.edu.in #### email to nofity with following options/scenarios
#$ -m a #### send mail in case the job is aborted
#$ -m b #### send mail when job begins
#$ -m e #### send mail when job ends
#$ -m s #### send mail when job is suspended
#$ -l h_vmem=32G
#$ -l gpu=1
############################## END OF DEFAULT EMBEDDED SGE COMMANDS #######################
CUDA_VISIBLE_DEVICES=`get_CUDA_VISIBLE_DEVICES` || exit
export CUDA_VISIBLE_DEVICES 

module load pytorch/1.0.1
module load python/anaconda/3
module unload gcc
module load gcc/5.2.0
cd ..
python main.py --method MME --dataset multi --source real --target sketch --net alexnet --loss FL --attribute glove.42B.300d.txt --dim 300 --deep 0 --save_check
