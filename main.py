import numpy as np
import os,sys,time
import torch
import options,data,util
import model

print(util.yellow("======================================================="))
print(util.yellow("main.py (photometric mesh optimization)"))
print(util.yellow("======================================================="))

print(util.magenta("setting configurations..."))
opt = options.set()

print(util.magenta("reading list of sequences..."))
seq_list = data.load_sequence_list(opt,subset=1)
seq_list = list(range(5, 75))

with torch.cuda.device(opt.gpu):

    pmo = model.Model(opt)
    pmo.build_network(opt)
    pmo.restore_checkpoint(opt)

    print(util.yellow("======= OPTIMIZATION START ======="))
    for num in seq_list:
        opt.name = num
        try:
            pmo.load_sequence(opt,num)
        except Exception as e:
            raise Exception(e)
            continue
        pmo.setup_visualizer(opt)
        pmo.setup_variables(opt)
        pmo.setup_optimizer(opt)
        pmo.time_start(opt)
        pmo.optimize(opt)
        print('WRITING MESH')
        pmo.save_mesh(opt)
    print(util.yellow("======= OPTIMIZATION DONE ======="))
    #pmo.write_video(opt)
    
    
