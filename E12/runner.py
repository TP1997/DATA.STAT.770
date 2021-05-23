import os
os.chdir('/home/tuomas/R/Projects/DATA.STAT.770/E12')
inputfile = 'ckd.dat'
for i in range(0,11):
    outputfile = 'ckd_out0{}.dat'.format(i)
    lam = i*0.1
    os.system('nerv --inputfile {} --outputfile {} --lambda {}'.format(inputfile, outputfile, lam))
    
    print('Lambda {} done.'.format(round(lam, 3)))
    
