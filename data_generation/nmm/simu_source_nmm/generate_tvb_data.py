from scipy.io import savemat
from tvb.simulator.lab import *
import time
import numpy as np
import multiprocessing as mp
import os
import argparse
import matplotlib.pyplot as plt
from scipy.signal import resample

def main(region_id, folder_name = "raw_nmm", reg_reshape=True):
    """ TVB Simulation to generate raw source space dynamics, unit in mV, and ms
    :param region_id: int; source region id, with parameters generating interictal spike activity
    """
    if not os.path.exists(f'../source/{folder_name}'): 
        os.mkdir(f'../source/{folder_name}')
    if not os.path.isdir(f'../source/{folder_name}/a{region_id}/'):
        os.mkdir(f'../source/{folder_name}/a{region_id}/')
    start_time = time.time()
    print('------ Generate data of region_id {} ----------'.format(region_id))
    conn = connectivity.Connectivity.from_file(source_file=os.getcwd()+'/../anatomy/connectivity_998.zip') # connectivity provided by TVB
    conn.configure()

    # define A value
    num_region = conn.number_of_regions
    a_range = [3.6]
    A = np.ones((num_region, len(a_range))) * 3.25                                  # the normal A value is 3.25
    A[region_id, :] = a_range

    # define mean and std
    #mean_and_std = np.array([[0.087, 0.08, 0.083], [1, 1.7, 1.5]])
    mean_and_std = np.array([[0.087], [1]]) # les valeurs qui fonctionnent le meieux
    for iter_a in range(A.shape[1]):
        use_A = A[:, iter_a]
        for iter_m in range(mean_and_std.shape[1]):

            jrm = models.JansenRit(A=use_A, mu=np.array(mean_and_std[0][iter_m]),
                                   v0=np.array([6.]), p_max=np.array([0.15]), p_min=np.array([0.03]))
            phi_n_scaling = (jrm.a * 3.25 * (jrm.p_max - jrm.p_min) * 0.5 * mean_and_std[1][iter_m]) ** 2 / 2.
            sigma = np.zeros(6)
            sigma[4] = phi_n_scaling

            # set the random seed for the random intergrator
            randomStream = np.random.mtrand.RandomState(0)
            noise_class = noise.Additive(random_stream=randomStream, nsig=sigma)
            # integ = integrators.HeunStochastic(dt=2 ** -1, noise=noise_class)

            sim = simulator.Simulator(
                model=jrm,
                connectivity=conn,
                coupling=coupling.SigmoidalJansenRit(a=np.array([1.0])),
                integrator=integrators.HeunStochastic(dt=2 ** -1, noise=noise_class), #noise.Additive(nsig=sigma)),
                monitors=(monitors.Raw(),)
            ).configure()

            # run 200s of simulation, cut it into 20 pieces, 10s each. (Avoid saving large files)
            for iii in range(6): #initialement range(20)
                siml = 1e4
                out = sim.run(simulation_length=siml)
                (t, data), = out
                data = (data[:, 1, :, :] - data[:, 2, :, :]).squeeze().astype(np.float32)

                # # in the fsaverage5 mapping, there is no vertices corresponding to region 7,325,921, 949, so change label 994-998 to those id
                if reg_reshape : 
                    data[:, 7] = data[:, 994]
                    data[:, 325] = data[:, 997]
                    data[:, 921] = data[:, 996]
                    data[:, 949] = data[:, 995]
                    data = data[:, :994]

                # downsample to save memory
                data = resample(data, num=data.shape[0]//4)
                savemat(f'../source/{folder_name}/a{region_id}/mean_iter_{iter_m}_a_iter_{region_id}_{iii}.mat',
                        {'time': t, 'data': data, 'A': use_A})
                
                #if iter_a not in [7, 325, 921, 949] : 
                #    plt.figure()
                #    plt.subplot(1,2,1)
                #    for k in range(data.shape[1]):
                #        plt.plot(data[:,k].transpose())
                #    plt.ylim([-5,15])
                #    plt.title(f"region : {region_id}; mean firing rate : {mean_and_std[0][iter_m]}")
                #    plt.subplot(1,2,2)
                #    if iter_a == 994 : 
                #        a_match = 7 
                #    elif iter_a == 995 : 
                #        a_match = 949 
                #    elif iter_a == 996 : 
                #        a_match = 921
                #    elif iter_a == 997 :
                #        a_match = 325
                #    else : 
                #        a_match = region_id
                #    plt.plot(data[:,a_match].transpose())
                #    plt.title(f"a match {a_match}")
                #    plt.ylim([-5,15])
                #    plt.savefig(f"./figs/tvb_output/region_{region_id}_iterM_{iter_m}_iterT_{iii}.png")
                #    plt.close()
    print('Time for', region_id, time.time() - start_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TVB Simulation')
    parser.add_argument('--folder', type=str, default="raw_nmm", help='folder in which to save results')
    parser.add_argument('--a_start', type=int, default=0, metavar='t/f', help='start region id')
    parser.add_argument('--a_end', type=int, default=1, metavar='t/f', help='end region id')
    #parser.add_argument('--reg_reshape', action='store_false', help="reshape regions")
    args = parser.parse_args()
    os.environ["MKL_NUM_THREADS"] = "1"
    start_time = time.time()
    # RUN THE CODE IN PARALLEL
    processes = [mp.Process(target=main, args=(x,args.folder)) for x in range(args.a_start, args.a_end)]
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    # NO PARALLEL
    #for x in range(args.a_start, args.a_end):
    #   main(x)
    print('Total_time', time.time() - start_time)