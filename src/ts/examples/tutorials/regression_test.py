#!/opt/local/bin/python

import re, string, sys, os, getopt
import time as timing
import matplotlib.pyplot as plt
import numpy as np
import struct
import cPickle as pickle
import PetscBinaryIO
import ConfigParser

def SetUpExperiments(casename='',force=False,experiments=[]):
    
    experiment={}
    experiment['name']='ex22'
    experiment['strRefSolMethod']=' -ts_type arkimex -ts_arkimex_type 3'
    experiment['strTestProblem']=experiment['name']
    experiment['strTestProblemOutFile']=experiment['name']+'.out'
    experiment['strTestProblemFile']=experiment['name']+'_'+casename+'.pcl'
    experiment['strTestProblemRefSolFile']=experiment['name']+'_ref_sol.pcl'
    experiment['optDetails']=False
    experiment['optForce']=force
    experiment['strPETScXtraArguments']=' '
    experiment['PETScOptionsStr']='-ts_max_snes_failures -1  -ksp_max_it 5000000 -ts_atol 1e-5 -ts_rtol 1e-5 -ts_adapt_type none -ksp_rtol 1e-10 -snes_rtol 1e-10 -da_grid_x 100 -k0 100.0 -k1 200.0'
    experiment['PETScOptionsStrDetails']='  -ts_monitor_draw_solution -ts_monitor -ts_adapt_monitor  '
    experiment['n']=2*100
    experiment['tfinal']=1.0
    experiment['tsmaxsteps']=[100,200,500,1000]
    experiments.append(experiment)

    experiment={}
    experiment['name']='ex16'
    experiment['strRefSolMethod']=' -ts_type arkimex -ts_arkimex_type 4 '
    experiment['strTestProblem']=experiment['name']
    experiment['strTestProblemOutFile']=experiment['name']+'.out'
    experiment['strTestProblemFile']=experiment['name']+'_'+casename+'.pcl'
    experiment['strTestProblemRefSolFile']=experiment['name']+'_ref_sol.pcl'
    experiment['optDetails']=False
    experiment['optForce']=force
    experiment['strPETScXtraArguments']=' -ts_type arkimex -ts_arkimex_type 3 '
    experiment['PETScOptionsStr']=' -ts_max_snes_failures -1  -ksp_max_it 5000000 -ts_atol 1e-5 -ts_rtol 1e-5 -ts_adapt_type none -ksp_rtol 1e-10 -snes_rtol 1e-10 -mu 1.0 -imex true '
    experiment['PETScOptionsStrDetails']='  -ts_monitor_draw_solution -ts_monitor -ts_adapt_monitor  '
    experiment['n']=2
    experiment['tfinal']=2.0
    experiment['tsmaxsteps']=[20,40,100,150,200]
    experiments.append(experiment)

    experiment={}
    experiment['name']='ex16'
    experiment['strRefSolMethod']=' -ts_type arkimex -ts_arkimex_type 4 '
    experiment['strTestProblem']=experiment['name']
    experiment['strTestProblemOutFile']=experiment['name']+'_s'+'.out'
    experiment['strTestProblemFile']=experiment['name']+'_'+casename+'_s'+'.pcl'
    experiment['strTestProblemRefSolFile']=experiment['name']+'_s'+'_ref_sol.pcl'
    experiment['optDetails']=False
    experiment['optForce']=force
    experiment['strPETScXtraArguments']=' -ts_type arkimex -ts_arkimex_type 3 '
    experiment['PETScOptionsStr']=' -ts_max_snes_failures -1  -ksp_max_it 5000000 -ts_atol 1e-5 -ts_rtol 1e-5 -ts_adapt_type none -ksp_rtol 1e-10 -snes_rtol 1e-10 -mu 1000.0 -imex true '
    experiment['PETScOptionsStrDetails']='  -ts_monitor_draw_solution -ts_monitor -ts_adapt_monitor  '
    experiment['n']=2
    experiment['tfinal']=2.0
    experiment['tsmaxsteps']=[20,40,100,150,200]
    experiments.append(experiment)

    experiment={}
    experiment['name']='ex36'
    experiment['strTestProblem']=experiment['name']
    experiment['strRefSolMethod']=' -ts_type cn '
    experiment['strTestProblemOutFile']=experiment['name']+'.out'
    experiment['strTestProblemFile']=experiment['name']+'_'+casename+'.pcl'
    experiment['strTestProblemRefSolFile']=experiment['name']+'_ref_sol.pcl'
    experiment['optDetails']=False
    experiment['optForce']=force
    experiment['strPETScXtraArguments']=' '
    experiment['PETScOptionsStr']='-ts_max_snes_failures -1  -ksp_max_it 5000000 -ts_atol 1e-5 -ts_rtol 1e-5 -ts_adapt_type none -ksp_rtol 1e-10 -snes_rtol 1e-10'
    experiment['PETScOptionsStrDetails']=' -ts_monitor_lg_solution -ts_monitor_lg_timestep -lg_indicate_data_points 0 -ts_monitor -ts_adapt_monitor '
    experiment['n']=5
    experiment['tfinal']=0.015
    experiment['tsmaxsteps']=[150,300,600,800,1000,1250,1500]
    experiments.append(experiment)

    experiment={}
    experiment['name']='ex36SE'
    experiment['strTestProblem']=experiment['name']
    experiment['strRefSolMethod']='  -ts_type cn  '
    experiment['strTestProblemOutFile']=experiment['name']+'.out'
    experiment['strTestProblemFile']=experiment['name']+'_'+casename+'.pcl'
    experiment['strTestProblemRefSolFile']=experiment['name']+'_ref_sol.pcl'
    experiment['optDetails']=False
    experiment['optForce']=force
    experiment['strPETScXtraArguments']=' '
    experiment['PETScOptionsStr']='-ts_max_snes_failures -1  -ksp_max_it 5000000 -ts_atol 1e-5 -ts_rtol 1e-5 -ts_adapt_type none -ksp_rtol 1e-10 -snes_rtol 1e-10'
    experiment['PETScOptionsStrDetails']=' -ts_monitor_lg_solution -ts_monitor_lg_timestep -lg_indicate_data_points 0 -ts_monitor -ts_adapt_monitor '
    experiment['n']=5
    experiment['tfinal']=0.015
    experiment['tsmaxsteps']=[150,300,600,800,1000,1250,1500]
    experiments.append(experiment)

    experiment={}
    experiment['name']='ex36A'
    experiment['strTestProblem']=experiment['name']
    experiment['strRefSolMethod']='  -ts_type rosw '
    experiment['strTestProblemOutFile']=experiment['name']+'.out'
    experiment['strTestProblemFile']=experiment['name']+'_'+casename+'.pcl'
    experiment['strTestProblemRefSolFile']=experiment['name']+'_ref_sol.pcl'
    experiment['optDetails']=False
    experiment['optForce']=force
    experiment['strPETScXtraArguments']=' '
    experiment['PETScOptionsStr']='-ts_max_snes_failures -1  -ksp_max_it 5000000 -ts_atol 1e-5 -ts_rtol 1e-5 -ts_adapt_type none -ksp_rtol 1e-10 -snes_rtol 1e-10'
    experiment['PETScOptionsStrDetails']=' -ts_monitor_lg_solution -ts_monitor_lg_timestep -lg_indicate_data_points 0 -ts_monitor -ts_adapt_monitor '
    experiment['n']=6
    experiment['tfinal']=0.015
    experiment['tsmaxsteps']=[150,300,600,800,1000,1250,1500]
    experiments.append(experiment)

    return experiments

def RunExperiment(experiment):
    io = PetscBinaryIO.PetscBinaryIO()


    name=experiment['name']
    strTestProblem=experiment['strTestProblem']
    strRefSolMethod=experiment['strRefSolMethod']
    optDetails=experiment['optDetails']
    strPETScXtraArguments=experiment['strPETScXtraArguments']
    PETScOptionsStr=experiment['PETScOptionsStr']
    n=experiment['n']
    force=experiment['optForce']
    tfinal=experiment['tfinal']
    tsmaxsteps=np.array(experiment['tsmaxsteps'])
    strTestProblemFile=experiment['strTestProblemFile']
    list_supported_problems=['ex36','ex36SE','ex36A','ex22','ex16']

    if (not strTestProblem in list_supported_problems):
        raise NameError('Problem '+ strTestProblem +' is not supported. Aborting.')

    strTestProblemOutFile=experiment['strTestProblemOutFile']
    strTestProblemRefSolFile=experiment['strTestProblemRefSolFile']

    tsmaxsteps=tsmaxsteps.astype(np.int)
    tsdt=np.float(tfinal)/tsmaxsteps
    msims=tsdt.size
    tsmaxsteps_ref=np.int(10*tsmaxsteps[msims-1])
    tsdt_ref=np.float(tfinal)/tsmaxsteps_ref
    timesteps=np.zeros((msims,1))
    solution=np.zeros((msims,n))

    PETScOptionsStr=PETScOptionsStr + ' -ts_final_time '+str(tfinal) 

    if(optDetails):
        PETScOptionsStr=PETScOptionsStr + experiment['PETScOptionsStrDetails']

    print 'Building ' + strTestProblem
    os_out=os.system('make -s ' + strTestProblem)
    if(os_out <> 0):
        raise NameError('Possible compilation errors. Aborting.')

    bWriteReference=os.path.isfile(strTestProblemRefSolFile)

    if bWriteReference==False:
        print 'Running ' + strTestProblem + ' to generate the reference solution with dt = ' + str(tsdt_ref) + '.'
        string_to_run=strTestProblem +  ' -ts_dt '+ str(tsdt_ref) + ' -ts_max_steps ' + str(tsmaxsteps_ref) + ' '  + PETScOptionsStr +' '+ strRefSolMethod  + ' ' +strPETScXtraArguments + ' -ts_view_solution binary:'+ strTestProblemOutFile + ' '
        print string_to_run
        os_out=os.system(string_to_run)
        if(os_out <> 0):
            raise NameError('Error running '+ strTestProblem +'. Aborting.')
        PETSc_objects = io.readBinaryFile(strTestProblemOutFile)

        solution_ref=PETSc_objects[0][:]
        timesteps_ref=tsmaxsteps_ref
        outpcl = open(strTestProblemRefSolFile, 'wb')
        pickle.dump(tsdt_ref,outpcl)
        pickle.dump(timesteps_ref,outpcl)
        pickle.dump(solution_ref,outpcl)
        outpcl.close()

    # Running the simulation with different time steps
    bCaseRun=os.path.isfile(strTestProblemFile)
    if (bCaseRun==False or force==True):
        for simID in range(0,msims):
            print 'Running ' + strTestProblem + ' with dt = '+ str(tsdt[simID])
            str_run=strTestProblem +  ' -ts_dt '+ str(tsdt[simID]) + ' -ts_max_steps ' + str(tsmaxsteps[simID]) + ' '  + PETScOptionsStr + ' ' + ' -ts_view_solution binary:'+ strTestProblemOutFile + ' ' +strPETScXtraArguments+' '
            print str_run
            os_out=os.system(str_run)
            if(os_out <> 0):
                raise NameError('Error running '+ strTestProblem +'. Aborting.')

            PETSc_objects = io.readBinaryFile(strTestProblemOutFile)
            solution[simID,0:n]=PETSc_objects[0][:]
            timesteps[simID]=tsmaxsteps[simID]

        print 'Reading the reference solution.'
        outpcl=open(strTestProblemRefSolFile,'rb')
        tsdt_ref=pickle.load(outpcl)
        timesteps_ref = pickle.load(outpcl)
        solution_ref = pickle.load(outpcl)
        outpcl.close()


        if(strTestProblem=='ex36'):
            err_test=np.abs(solution[0:msims,4]-solution_ref[4])
        elif(strTestProblem=='ex36SE'):
            err_test=np.abs((solution[0:msims,4]-solution[0:msims,2])-(solution_ref[4]-solution_ref[2]))
        elif(strTestProblem=='ex36A'):
            err_test=np.abs(solution[0:msims,4]-solution_ref[4])
        elif(strTestProblem=='ex22' or strTestProblem=='ex16'):
            from numpy import linalg as LA
            err_test=np.zeros((msims))
            for i in range(msims):
                err_test[i]=LA.norm(solution[i,:]-solution_ref[:])

        outpcl = open(strTestProblemFile, 'wb')
        pickle.dump(tsdt,outpcl)
        pickle.dump(err_test,outpcl)
        pickle.dump(msims,outpcl)
        outpcl.close()
    else:
        inpcl = open(strTestProblemFile, 'rb')
        tsdt=pickle.load(inpcl)
        err_test=pickle.load(inpcl)
        msims=pickle.load(inpcl)
        inpcl.close()

    plt.clf
    plt.cla
    plt.close('all')
    plt.plot(tsdt[0:msims],err_test,'ko-', markersize=16)

    plt.xscale('log')
    plt.yscale('log')
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.grid(True)
    print 'Ploting results.'
    plt.xlabel('time step')
    plt.ylabel('error')


    #plt.show()

def Compare(experiment_before,experiment_after):
    strTestProblemFileB=experiment_before['strTestProblemFile']
    strTestProblemFileA=experiment_after['strTestProblemFile']

    inpcl = open(strTestProblemFileB, 'rb')
    tsdtB=pickle.load(inpcl)
    err_testB=pickle.load(inpcl)
    msimsB=pickle.load(inpcl)
    inpcl.close()

    inpcl = open(strTestProblemFileA, 'rb')
    tsdtA=pickle.load(inpcl)
    err_testA=pickle.load(inpcl)
    msimsA=pickle.load(inpcl)
    inpcl.close()

    plt.clf
    plt.cla
    plt.close('all')
    plt.plot(tsdtB[0:msimsB],err_testB,'kp-', markersize=16,label='before')
    plt.plot(tsdtA[0:msimsA],err_testA,'r*-', markersize=16,label='after')
    plt.legend()
    plt.title('Comparison: '+experiment_before['name']+'-'+experiment_after['name'])
    plt.xscale('log')
    plt.yscale('log')
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.grid(True)
    print 'Ploting results.'
    plt.xlabel('time step')
    plt.ylabel('error')

    plt.show()

def main():
    exper_before=SetUpExperiments(casename='before',force=False,experiments=[])
    exper=SetUpExperiments(casename='after',force=True,experiments=[])

    print 'We have '+ str(len(exper)) + ' experiments.'
    for i in range(len(exper)):
        print exper_before[i]
        RunExperiment(exper_before[i])

    for i in range(len(exper)):
        print exper[i]
        RunExperiment(exper[i])
        Compare(exper_before[i],exper[i])
    #RunExperiment(exper[5])
    plt.show()

if __name__ == '__main__':
    main()
