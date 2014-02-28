#!/opt/local/bin/python

import re, string, sys, os, getopt
import time as timing
import matplotlib.pyplot as plt
import numpy as np
import struct
import cPickle as pickle
import PetscBinaryIO

io = PetscBinaryIO.PetscBinaryIO()

list_supported_problems=['ex36','ex36SE','ex36A']

try:
    opts, args = getopt.getopt(sys.argv[1:],"he:d:")
except getopt.GetoptError:
    print 'test.py -e <example: ex36> -d <details: (0) or 1>'
    print 'e.g.  test.py -e ex36'
    sys.exit(2)

optDetails=False
strTestProblem='ex36'
for opt, arg in opts:
    if opt == '-h':
        print 'test.py -e <example: ex36> -d <details: (0) or 1>'
        print 'e.g.  test.py -e ex36'
        sys.exit()
    elif opt in ('-e'):
        strTestProblem = arg.lstrip()
    elif opt in ('-d'):
        print arg
        optDetails = arg in ['true', '1', 't', 'y', 'yes', 'yup']


if (not strTestProblem in list_supported_problems):
    raise NameError('Problem '+ strTestProblem +' is not supported. Aborting.')
strTestProblemOutFile=strTestProblem+'.out'
strTestProblemRefSolFile=strTestProblem+'_ref_sol.pcl'


if (strTestProblem in ['ex36','ex36SE','ex36A']):
    n=5
    if (strTestProblem =='ex36A'):
        n=n+1

    tfinal=0.015
    tsmaxsteps=np.array([150,300,600,800,1000,1250,1500])
    tsmaxsteps=tsmaxsteps.astype(np.int)
    tsdt=np.float(tfinal)/tsmaxsteps
    msims=tsdt.size
    tsmaxsteps_ref=np.int(100*tsmaxsteps[msims-1])
    tsdt_ref=np.float(tfinal)/tsmaxsteps_ref
    timesteps=np.zeros((msims,1))
    solution=np.zeros((msims,n))

    PETScOptionsStr='-ts_type rosw -ts_max_snes_failures -1  -ksp_max_it 5000000 -ts_atol 1e-5 -ts_rtol 1e-5 -ts_adapt_type none -ksp_rtol 1e-10 -snes_rtol 1e-10  -ts_arkimex_type 3 -ts_rosw_type ra3pw '

    if(optDetails):
        PETScOptionsStr=PETScOptionsStr + ' -ts_monitor_lg_solution -ts_monitor_lg_timestep -lg_indicate_data_points 0 -ts_monitor -ts_adapt_monito '


print 'Building ' + strTestProblem
os_out=os.system('make -s ' + strTestProblem + ' PETSC_ARCH=arch-mac-opt')
if(os_out <> 0):
    raise NameError('Possible compilation errors. Aborting.')

bWriteReference=os.path.isfile(strTestProblemRefSolFile)

if bWriteReference==False:
    print 'Running ' + strTestProblem + ' to generate the reference solution with dt = ' + str(tsdt_ref) + '.'
    os_out=os.system(strTestProblem +  ' -ts_dt '+ str(tsdt_ref) + ' -ts_max_steps ' + str(tsmaxsteps_ref) + ' '  + PETScOptionsStr + ' -ts_view_solution binary:'+ strTestProblemOutFile + ' ')
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
for simID in range(0,msims):
    print 'Running ' + strTestProblem + ' with dt = '+ str(tsdt[simID])
    os_out=os.system(strTestProblem +  ' -ts_dt '+ str(tsdt[simID]) + ' -ts_max_steps ' + str(tsmaxsteps[simID]) + ' '  + PETScOptionsStr + ' -ts_view_solution binary:'+ strTestProblemOutFile + ' ')
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

plt.clf
plt.cla
plt.close('all')
if(strTestProblem=='ex36'):
    err_test=np.abs(solution[0:msims,4]-solution_ref[4])
elif(strTestProblem=='ex36SE'):
    err_test=np.abs((solution[0:msims,4]-solution[0:msims,2])-(solution_ref[4]-solution_ref[2]))
elif(strTestProblem=='ex36A'):
    err_test=np.abs(solution[0:msims,4]-solution_ref[4])

plt.plot(tsdt[0:msims],err_test,'ko-', markersize=16)

plt.xscale('log')
plt.yscale('log')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.grid(True)
print 'Ploting results.'

plt.show()
