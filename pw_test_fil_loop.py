import pickle
from os import path

import lightweaver as lw
import matplotlib.pyplot as plt
import numpy as np
from lightweaver.fal import Falc82
import promweaver as pw
from promweaver.iso_prom import IsoPromModel
from promweaver.compute_bc import compute_falc_bc_ctx

from stratified_prom import StratifiedPromModel
import glob
import sys
import gzip

def main():
    # First var is the oe we look at
    var = sys.argv[1]
    imgn = sys.argv[2] #path
    prom = sys.argv[3] #prom mode
    
    print(f"Received variable: {var}")
    print(f"Received path: {imgn}")
    print(f"Received mode: {prom}")
    var = int(var)

    active_atoms = ['H', 'Ca', 'Mg']
    Nthreads = 30
    # NOTE(cmo): String hashes are not stable across python runs. Huh.
    # active_atoms_hash = sum(hash(a) for a in active_atoms)
    active_atoms_hash = ''.join(sorted(active_atoms))

    #data_path = f'FalcTabulatedBcData_{active_atoms_hash}.pickle'
    #if path.isfile(data_path):
    #    with open(data_path, 'rb') as pkl:
    #        bc_table = pickle.load(pkl)
    #else:
    bc_ctx = pw.compute_falc_bc_ctx(active_atoms, Nthreads=Nthreads)

    bc_table = pw.tabulate_bc(bc_ctx)

    #with open(data_path, 'wb') as pkl:
        #pickle.dump(bc_table, pkl)

    #print(f'Made BC data and written to {data_path}')

    bc_provider = pw.TabulatedPromBcProvider(**bc_table)

    #prom = 0
    row = 1584

    zstart = 0
    zend = 1240

    def get_vturb(T, epsilon=0.5, alpha=0.1, i=0, gamma=5/3, mH=1.6735575e-27):
        k = 1.380649e-23
        m = (1 + 4*alpha)/(1 + alpha + i) * mH
        return epsilon * np.sqrt(gamma * k * T / m)

    #imgn = 290
    
    if prom == 'fillamet':
        pn = '0'
        altitude = 1e7
    else:
        pn = '1'
        

    #Valeriia
    list1 = np.sort(glob.glob('/Volumes/LaCie/valeriia/img_'+imgn+'/pRT_a0'+imgn+'_p'+pn+'c*.dat'))
    #name = 'pRT_a0660_p0c768.dat'
    val = np.flip(np.loadtxt(list1[var]), axis=0)
    z = np.ascontiguousarray((val[:,0] - val[-1,0])/100)
    temp = np.ascontiguousarray(val[:,1])
    vlos = np.ascontiguousarray(val[:,4] / 100)
    ne = np.ascontiguousarray(val[:,5] / 1e-6)
    pressure = np.ascontiguousarray(val[:,2]/10 )
    nH = np.ascontiguousarray(val[:,6] / 1e-6)
    vturb = get_vturb(temp)
    
    if prom == 'prominence':
        altitude = val[0,8] / 100 + 1e7
    
    modelv = StratifiedPromModel(
        prom,
        z=z,
        temperature=temp,
        ne=ne,
        vlos=vlos,
        pressure=pressure,
        #nH = nH,
        vturb=vturb,
        altitude=altitude,
        active_atoms=active_atoms,
        Nthreads=30,
        bc_provider=bc_provider,
        do_pressure_updates = True,
        prd=True,
    )
    modelv.ctx.depthData.fill = True
    modelv.iterate_se(prd=True)
    modelv.ctx.formal_sol_gamma_matrices()
    Is_v = modelv.compute_rays()
    wave = modelv.ctx.spect.wavelength

    mu_idx = -1
    fal_interp = bc_provider.compute_I(wave, modelv.atmos.muz[mu_idx])
    

    fig = plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(232)
    c1l = np.where(wave > 393.4)[0][0]
    c1r = np.where(wave > 393.54)[0][0]
    plt.plot(wave[c1l:c1r], fal_interp[c1l:c1r], '--', label=f"FAL C @ mu={modelv.atmos.muz[mu_idx]:.3f}")
    #plt.plot(wave[c1l:c1r], Is[c1l:c1r,-1], label="Jack model")
    plt.plot(wave[c1l:c1r], Is_v[c1l:c1r,-1], label="Valeriia Model")
    #plt.plot(wave2[c1l:c1r], Is2[c1l:c1l,-1], label="Semi-empirical filament")
    #plt.legend()
    plt.title('Ca II K')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    #plt.legend()

    plt.subplot(236)
    c2l = np.where(wave > 102.5)[0][0]
    c2r = np.where(wave > 102.63)[0][0]
    plt.plot(wave[c2l:c2r], fal_interp[c2l:c2r], '--', label=f"FAL C @ mu={modelv.atmos.muz[mu_idx]:.3f}")
    #plt.plot(wave[c2l:c2r], Is[c2l:c2r,-1], label="Jack model")
    plt.plot(wave[c2l:c2r], Is_v[c2l:c2r,-1], label="Valeriia model")
    #plt.plot(wave2[c2l:c2r], Is2[c2l:c2l,-1], label="Semi-empirical filament")
    #plt.legend()
    plt.title(r'Ly$\beta$')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')


    plt.subplot(231)
    c3l = np.where(wave > 656.47-0.15)[0][0]
    c3r = np.where(wave > 656.47+0.15)[0][0]
    plt.plot(wave[c3l:c3r], fal_interp[c3l:c3r], '--', label=f"FAL C @ mu={modelv.atmos.muz[mu_idx]:.3f}")
    #plt.plot(wave[c3l:c3r], Is[c3l:c3r,-1], label="Veronika model")
    plt.plot(wave[c3l:c3r], Is_v[c3l:c3r,-1], label="Valeriia model")
    #plt.plot(wave2[c3l:c3r], Is2[c3l:c3l,-1], label="Semi-empirical filament")
    #plt.legend()
    plt.title(r'H$\alpha$')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')

    plt.subplot(234)
    c4l = np.where(wave > 849.99)[0][0]
    c4r = np.where(wave > 850.1)[0][0]
    plt.plot(wave[c4l:c4r], fal_interp[c4l:c4r], '--', label=f"FAL C @ mu={modelv.atmos.muz[mu_idx]:.3f}")
    #plt.plot(wave[c4l:c4r], Is[c4l:c4r,-1], label="Jack model")
    plt.plot(wave[c4l:c4r], Is_v[c4l:c4r,-1], label="Valeriia model")
    #plt.plot(wave2[c4l:c4r], Is2[c4l:c4l,-1], label="Semi-empirical filament")
    #plt.legend()
    plt.title(r'Ca II 8542 $\AA$')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')

    plt.subplot(233)
    c4l = np.where(wave > 121.50)[0][0]
    c4r = np.where(wave > 121.65)[0][0]
    plt.plot(wave[c4l:c4r], fal_interp[c4l:c4r], '--', label=f"FAL C @ mu={modelv.atmos.muz[mu_idx]:.3f}")
    #plt.plot(wave[c4l:c4r], Is[c4l:c4r,-1], label="Jack model")
    plt.plot(wave[c4l:c4r], Is_v[c4l:c4r,-1], label="Valeriia model")
    #plt.plot(wave2[c4l:c4r], Is2[c4l:c4l,-1], label="Semi-empirical filament")
    #plt.legend()
    plt.title(r'Ly$\alpha$')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')

    plt.subplot(235)
    c4l = np.where(wave > 280.3)[0][0]
    c4r = np.where(wave > 280.4)[0][0]
    plt.plot(wave[c4l:c4r], fal_interp[c4l:c4r], '--', label=f"FAL C @ mu={modelv.atmos.muz[mu_idx]:.3f}")
    #plt.plot(wave[c4l:c4r], Is[c4l:c4r,-1], label="Jack model")
    plt.plot(wave[c4l:c4r], Is_v[c4l:c4r,-1], label="Valeriia model")
    #plt.plot(wave2[c4l:c4r], Is2[c4l:c4l,-1], label="Semi-empirical filament")
    #plt.legend()
    plt.title('Mg II H')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.tight_layout()
    #path
    path = '/Volumes/LaCie/valeriia/'
    plt.savefig(path+'img_'+imgn+'/'+list1[var][32:]+'.png')
    plt.clf()
    
    np.savez_compressed(path+'img_'+imgn+'/'+list1[var][32:]+'.npz', wave=wave, Is=Is_v, z=z, chi=modelv.ctx.depthData.chi, eta=modelv.ctx.depthData.eta, Idir=modelv.ctx.depthData.I, sca=modelv.ctx.background.sca, J=modelv.ctx.spect.J)
    
    #np.save('img_335/'+list1[var][15:]+'_wav.npy', wave)
    #np.save('img_335/'+list1[var][15:]+'_int.npy', Is_v)
    
    if var == 1:
        #Get J and stuff
        bc_ctx.depthData.fill = True
        bc_ctx.formal_sol_gamma_matrices()
        
        mu_idx = -1
        fal_interp = bc_provider.compute_I(wave, modelv.atmos.muz[mu_idx])
        
        falz = bc_ctx.atmos.z

        
        
        np.savez_compressed(path+'img_'+imgn+'/'+'falc'+'.npz', wave=wave, Is=fal_interp, z=falz, chi=bc_ctx.depthData.chi, eta=bc_ctx.depthData.eta, Idir=bc_ctx.depthData.I, sca=bc_ctx.background.sca, J=bc_ctx.spect.J)
        

    

if __name__ == "__main__":
    main()


