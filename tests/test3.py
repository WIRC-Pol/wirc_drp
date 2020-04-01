import sys

try:
    import os
    import numpy as np
    import glob
    import wirc_drp.wirc_object as wo
    from wirc_drp.utils import source_utils
    import matplotlib.pyplot as plt

    import warnings; warnings.simplefilter('ignore')

    print('Starting test3.py script.')

    #First we'll set up all the directories and filenames: 
    wircpol_dir = os.environ['WIRC_DRP'] # Your WIRCPOL_DRP directory (with a "/" at the end!)
    tutorial_dir = wircpol_dir + "Tutorial/sample_data/"

    all_spec_cube = np.load(tutorial_dir+"Elia2-14_HWP_Spectra.npy")
    hwp_ang = np.load(tutorial_dir+"Elia2-14_HWP_Angles.npy")
    nfiles = all_spec_cube.shape[0]

    fig,axes = plt.subplots(2,2,figsize=(10,10))

    for im in np.arange(nfiles):
        axes[0,0].plot(all_spec_cube[im,0,1,:])
        axes[0,1].plot(all_spec_cube[im,1,1,:])
        axes[1,0].plot(all_spec_cube[im,2,1,:])
        axes[1,1].plot(all_spec_cube[im,3,1,:])

    axes[0,0].set_ylim(-2000,30000)
    axes[0,1].set_ylim(-2000,30000)
    axes[1,0].set_ylim(-2000,30000)
    axes[1,1].set_ylim(-2000,30000)
    print('Spec cube plotting functional.')
    plt.close()

    q,u,q_err,u_err,q_ind,u_ind = source_utils.compute_qu_for_obs_sequence(all_spec_cube,hwp_ang,run_alignment=True, method='flux_ratio')

    p = np.sqrt(q**2+u**2)
    theta = 0.5*np.degrees(np.arctan2(u,q))
    theta[theta<0] += 180

    fig,axes = plt.subplots(2,2,figsize=(20,12))

    n_q = q.shape[0]

    q_mean = np.nanmean(q[:15,:],axis=(0))
    u_mean = np.nanmean(u[:15,:],axis=(0))
    p_mean = np.sqrt(q_mean**2+u_mean**2)
    theta_mean = 0.5*np.degrees(np.arctan2(u_mean,q_mean))
    theta_mean[theta_mean<0] += 180


    q_median = np.nanmedian(q[:15,:],axis=(0))
    u_median = np.nanmedian(u[:15,:],axis=(0))
    p_median = np.sqrt(q_median**2+u_median**2)
    theta_median = 0.5*np.degrees(np.arctan2(u_median,q_median))
    theta_median[theta_median<0] += 180

    for i in range(15):
        #Plot Q
        axes[0,0].plot(q[i,:], 'C0', alpha=2/n_q)
        
        #Plot U
        axes[0,1].plot(u[i,:], 'C1', alpha=2/n_q)
        
        #Plot p
        axes[1,0].plot(p[i,:], 'C2', alpha=2/n_q)

        #Plot theta
        axes[1,1].plot(theta[i,:], 'C3', alpha=2/n_q)

    axes[0,0].plot(q_mean,'k',label="Mean")    
    axes[0,1].plot(u_mean,'k',label="Mean")   
    axes[1,0].plot(p_mean,'k',label="Mean")
    axes[1,1].plot(theta_mean,'k', label='Mean')

    axes[0,0].plot(q_median,'k--',label="Median")    
    axes[0,1].plot(u_median,'k--',label="Median")    
    axes[1,0].plot(p_median,'k--',label="Median")    
    axes[1,1].plot(theta_median,'k--',label="Median")

    axes[0,0].legend(fontsize=16)
    axes[0,1].legend(fontsize=16)
    axes[1,0].legend(fontsize=16)
    axes[1,1].legend(fontsize=16)

    axes[0,0].set_xlim(50,150)
    axes[0,1].set_xlim(50,150)
    axes[1,0].set_xlim(50,150)
    axes[1,1].set_xlim(50,150)

    axes[0,0].set_ylim(-0.05,.1)
    axes[0,1].set_ylim(-0.05,.1)
    axes[1,0].set_ylim(-0.05,.1)
    axes[1,1].set_ylim(0,180)

    print('Q and U double difference plots functional')
    plt.close()

    print('End of test script 3.')

except Exception as e:
    print("Failed Test 3 with exception {}".format(e))
    sys.exit(1)
