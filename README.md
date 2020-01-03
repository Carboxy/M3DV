# M3DV
Classification of 3D medical CT 
助教需要运行的程序为src/eval.py

How to run it:
首先需要保证para1.pkl和para2.pkl与eval处于同级目录下
usage:
optional arguments:
  -h, --help            show this help message and exit
  -tp TESTSET_PATH, --TestSet_Path TESTSET_PATH
                        测试集的路径，如test/
  -sp SAVE_SCORE_PATH, --Save_Score_Path SAVE_SCORE_PATH
                        保存score的路径，默认为score.npy

example：eval.py -tp test\ -sp result.npy
