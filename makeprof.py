import numpy as np
import glob
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, Si_atom, Al_atom, \
CaII_atom, Fe_atom, He_9_atom, MgII_atom, N_atom, S_atom, NaI_fine_atom
import scipy
import sys
import gzip

def main():

    def gettau(hal,har, chi, eta, sca, J, y):

            #crop data to right wav
            waveCropped_ha = wav[hal:har]
            waveCroppedCoreIdx_ha = np.argmin(abs(waveCropped_ha-waveCore_ha))

            intensityCropped_ha = dat[hal:har]
            
            chiCropped_ha = chi[hal:har]
            #chiCropped = model['chi'][waveIdxs[0]:waveIdxs[1],muIdx, upDown,...]
            #etaCropped = model['eta'][waveIdxs[0]:waveIdxs[1],muIdx, upDown,...]
            etaCropped_ha = eta[hal:har]
            scaCropped_ha = sca[hal:har]
            JCropped_ha = J[hal:har]
            velCropped_ha = ((waveCropped_ha - waveCore_ha) * (2.998e17/waveCore_ha) / 1000) #c in nm/s
            

            sourceCropped_ha = (etaCropped_ha+scaCropped_ha*JCropped_ha)/chiCropped_ha
            #tauCropped_ha = scipy.integrate.cumtrapz(chiCropped_ha, abs(y - y[0]), initial=0)
            tauCropped_ha = scipy.integrate.cumtrapz(chiCropped_ha, abs(y - y[0]), initial=0)
            #tauCropped = tau_(model, waveRange = waveRange, mu = mu, upDown = upDown)
            tauCropped_ha[tauCropped_ha<1e-10] = 1e-10

            velGrid_ha,heightGrid_ha = [obj.T for obj in np.meshgrid(velCropped_ha, y)]
            velGrid_ha = -velGrid_ha

            chitau = chiCropped_ha/tauCropped_ha

            temt_ha = tauCropped_ha * np.exp(-tauCropped_ha)
            temt_ha[temt_ha<1e-10] = 1e-10

            tauone_ha = np.argmin(abs(temt_ha-1/np.exp(1)), axis=1)
            
            return tauone_ha

    # First var is the one we look at
    
    name = sys.argv[1]
    path = sys.argv[2]
    prom = sys.argv[3]

    
    if int(prom):
        name = 'pRT_a0'+ name + '_p1c'
    else:
        name = 'pRT_a0'+ name + '_p0c'
        
    print(f"Looking: {path}{name}*.dat.npz")
    print(f"Prom mode: {prom}")
    
    datlist = np.sort(glob.glob(path+name+'*.dat.npz'))
    #wave, Is, chi, eta, Idir, sca, J
    
    c8_cube = np.zeros([len(datlist),51])
    ha_cube = np.zeros([len(datlist),51])
    ck_cube = np.zeros([len(datlist),23])
    mg_cube = np.zeros([len(datlist),122])
    la_cube = np.zeros([len(datlist),73])
    lb_cube = np.zeros([len(datlist),41])
    
    c8_tau = np.zeros([len(datlist),51])
    ha_tau = np.zeros([len(datlist),51])
    ck_tau = np.zeros([len(datlist),23])
    mg_tau = np.zeros([len(datlist),122])
    la_tau = np.zeros([len(datlist),73])
    lb_tau = np.zeros([len(datlist),41])
    
    
    waveCore_mg = MgII_atom().lines[0].lambda0
    #waveCore = MgII_atom().lines[7].lambda0
    waveCore_ha = H_6_atom().lines[4].lambda0
    waveCore_ck = CaII_atom().lines[0].lambda0
    waveCore_c8 = CaII_atom().lines[-1].lambda0
    waveCore_la = H_6_atom().lines[0].lambda0
    waveCore_lb = H_6_atom().lines[1].lambda0
    
    waveRange = [waveCore_ha - 0.05, waveCore_ha + 0.05]
    waveRange = [waveCore_ha - 0.1, waveCore_ha + 0.1]
    waveRange_ha = [waveCore_ha - 0.15, waveCore_ha + 0.15]
    mu = 1.0
    upDown = 1
    muIdx = -1
    
    falcdat = np.load('/Volumes/LaCie/valeriia/img_290/falc.npz')
    

    
    for i in range(len(datlist)):
    
        print(i)
        
        alldat = np.load(datlist[i])
    
        wav = alldat['wave']
        dat = alldat['Is'][:,muIdx]
        chi = np.concatenate((alldat['chi'][:,muIdx, upDown], falcdat['chi'][:,muIdx, upDown]), axis=1)
        eta = np.concatenate((alldat['eta'][:,muIdx, upDown], falcdat['eta'][:,muIdx, upDown]), axis=1)
        idir = np.concatenate((alldat['Idir'][:,muIdx, upDown], falcdat['Idir'][:,muIdx, upDown]), axis=1)
        sca = np.concatenate((alldat['sca'], falcdat['sca']), axis=1)
        J = np.concatenate((alldat['J'], falcdat['J']), axis=1)
        y = np.concatenate((alldat['z']+1e7, falcdat['z']))
        
        
        
        ##Ca8542
        c8l = np.where(wav > waveCore_c8-0.1)[0][0]
        c8r = np.where(wav > waveCore_c8+0.1)[0][0]
        
        c8w = wav[c8l:c8r]
        #print(c8w.shape)
        c8_cube[i] = dat[c8l:c8r]

        ##Halpha
        hal = np.where(wav > waveCore_ha-0.15)[0][0]
        har = np.where(wav > waveCore_ha+0.15)[0][0]
        
        haw = wav[hal:har]
        ha_cube[i] = dat[hal:har]
        
        ##Ck
        ckl = np.where(wav > waveCore_ck-0.1)[0][0]
        ckr = np.where(wav > waveCore_ck+0.1)[0][0]
        
        ckw = wav[ckl:ckr]
        ck_cube[i] = dat[ckl:ckr]
        
        ##Mg
        mgl = np.where(wav > waveCore_mg-0.1)[0][0]
        mgr = np.where(wav > waveCore_mg+0.1)[0][0]
        
        mgw = wav[mgl:mgr]
        mg_cube[i] = dat[mgl:mgr]
        
        ##ly alpha
        lal = np.where(wav > waveCore_la-0.1)[0][0]
        lar = np.where(wav > waveCore_la+0.1)[0][0]
        
        law = wav[lal:lar]
        la_cube[i] = dat[lal:lar]
        
        ##ly beta
        lbl = np.where(wav > waveCore_lb-0.1)[0][0]
        lbr = np.where(wav > waveCore_lb+0.1)[0][0]
        
        lbw = wav[lbl:lbr]
        lb_cube[i] = dat[lbl:lbr]
        
        
        ha_tau[i] = gettau(hal,har, chi, eta, sca, J, y)
        c8_tau[i] = gettau(c8l,c8r, chi, eta, sca, J, y)
        ck_tau[i] = gettau(ckl,ckr, chi, eta, sca, J, y)
        mg_tau[i] = gettau(mgl,mgr, chi, eta, sca, J, y)
        la_tau[i] = gettau(lal,lar, chi, eta, sca, J, y)
        lb_tau[i] = gettau(lbl,lbr, chi, eta, sca, J, y)
        
        
        

        '''
        layout = [['chi/tau','source'],['temt','cont']]
        fig = plt.figure(figsize=(15,15),constrained_layout=True)
        ax_dict = fig.subplot_mosaic(layout)
        shading = 'nearest'
        lw = 3

        ax_dict['chi/tau'].pcolormesh(velGrid, heightGrid,chitau, norm = LogNorm(vmin = 1e-2*chitau.mean()), cmap = 'Greys_r', shading = shading)
        ax_dict['chi/tau'].plot(model['model']['velocity_y']/1000,model['model']['y'], 'r', lw=lw)
        ax_dict['chi/tau'].plot(velGrid[tauone<len(model['model']['y'])-1,0],model['model']['y'][tauone[tauone<len(model['model']['y'])-1]], lw=lw)
        ax_dict['chi/tau'].axvline(0., linestyle = 'dotted', color = 'y', lw = lw)
        ax_dict['source'].pcolormesh(velGrid, heightGrid,sourceCropped, norm = LogNorm(vmin = 0.1*sourceCropped.mean()), cmap = 'Greys_r', shading = shading)
        ax_dict['source'].plot(model['model']['velocity_y']/1000,model['model']['y'], 'r', lw=lw)
        ax_dict['source'].plot(velGrid[tauone<len(model['model']['y'])-1,0],model['model']['y'][tauone[tauone<len(model['model']['y'])-1]], lw=lw)
        ax_dict['temt'].pcolormesh(velGrid, heightGrid,temt, norm = LogNorm(vmin = 2*temt.mean()), cmap = 'Greys_r', shading = shading)
        ax_dict['temt'].plot(model['model']['velocity_y']/1000,model['model']['y'], 'r', lw=lw)
        ax_dict['temt'].plot(velGrid[tauone<len(model['model']['y'])-1,0],model['model']['y'][tauone[tauone<len(model['model']['y'])-1]], lw=lw) #tau =  layer
        ax_dict['cont'].pcolormesh(velGrid, heightGrid,chitau * sourceCropped * temt, norm = LogNorm(vmin = 1e-6*(chitau * sourceCropped * temt).mean()), cmap = 'Greys_r', shading = shading)#, edgecolor = 'k')
        ax_dict['cont'].plot(model['model']['velocity_y']/1000,model['model']['y'], 'r', lw=lw)
        ax_dict['cont'].plot(velGrid[tauone<len(model['model']['y'])-1,0],model['model']['y'][tauone[tauone<len(model['model']['y'])-1]], lw=lw)
        ax_dict['cont2'] = ax_dict['cont'].twinx()
        ax_dict['cont2'].plot(-velCropped, intensityCropped[:,-1], color='orange')

        labels = [r'$\chi_{\nu}/\tau_{\nu}$', r'$S_{\nu}$', r'$\tau_{\nu} e^{-\tau_{\nu}}$', r'$C_{\nu}$']
        for index, panel in enumerate([item for sublist in layout for item in sublist]):
            ax_dict[panel].set_xlim(ax_dict[panel].get_xlim()[1],ax_dict[panel].get_xlim()[0])
            ax_dict[panel].text(ax_dict[panel].get_xlim()[0]+0.05*(ax_dict[panel].get_xlim()[1]-ax_dict[panel].get_xlim()[0]),ax_dict[panel].get_ylim()[0]+0.9*(ax_dict[panel].get_ylim()[1]-ax_dict[panel].get_ylim()[0]), labels[index], color='w', fontsize=30)

        plt.savefig('time_evolution/four_panel/' + f'source&tau_{current_time}.png', dpi=300, bbox_inches='tight')
    
    '''
    alldat = np.load(datlist[0])
    wav = alldat['wave']

    #hal = np.where(wav > waveCore_ha-0.10)[0][0]
    #har = np.where(wav > waveCore_ha+0.10)[0][0]
        
    #haw = wav[hal:har]
    
    print(ha_tau.shape, ha_tau.max())
    
    plt.imshow(ha_tau)
    plt.show()
    
    wav = wav*10
    
    falc_w = falcdat['wave']
    falc_i = falcdat['Is']

    
    if prom == 1:

        fig = plt.figure(figsize=(12, 6), dpi=100)
        plt.subplot(231)
        
        def reinterp(dat, wav, nwav):
        
            c8wavint = np.linspace(wav[0],wav[-1], 100)
            # Prepare the interpolated result array
            c8int = np.zeros((dat.shape[0], len(c8wavint)))
            # Interpolate each spectrum (column) in c8_cube
            for i in range(dat.shape[0]):
                c8int[i] = np.interp(c8wavint, wav, dat[i])
            return c8int
        
        c8int = reinterp(c8_cube, wav[c8l:c8r], 100)
        haint = reinterp(ha_cube, wav[hal:har], 100)
        ckint = reinterp(ck_cube, wav[ckl:ckr], 100)
        mgint = reinterp(mg_cube, wav[mgl:mgr], 100)
        laint = reinterp(la_cube, wav[lal:lar], 100)
        lbint = reinterp(lb_cube, wav[lbl:lbr], 100)
        
        falc_w = falcdat['wave']
        falc_i = falcdat['Is']

        
        #c8int = np.interp(c8wavint, wav[c8l:c8r], np.rot90(c8_cube))
        
        plt.imshow(((c8int)), origin='lower', cmap='gray', aspect='auto', extent=[wav[c8l], wav[c8r-1],-25,25], vmin=0, vmax=falc_i[c8l])
        plt.plot(wav[c8l:c8r],c8_cube[0]*1.3e13*10)
        plt.ylabel('X [Mm]')
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ca II 8542 $\rm\AA$')
        
        plt.subplot(232)
        plt.imshow(((haint)), origin='lower', cmap='gray', aspect='auto', extent=[wav[hal], wav[har-1], -25,25], vmin=0, vmax=falc_i[hal])
        plt.plot(wav[hal:har],ha_cube[0]*1e13*10)
        plt.ylabel('X [Mm]')
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'H$\rm\alpha$')
        
        plt.subplot(233)
        plt.imshow(((ckint)), origin='lower', cmap='gray', aspect='auto', extent=[wav[ckl], wav[ckr-1],-25,25], vmin=0, vmax=falc_i[ckl])
        plt.plot(wav[ckl:ckr],ck_cube[0]*1e14*10)
        plt.ylabel('X [Mm]')
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ca II K')
        
        plt.subplot(234)
        plt.imshow(((mgint)), origin='lower', cmap='gray', aspect='auto', extent=[wav[mgl], wav[mgr-1],-25,25], vmin=0, vmax=falc_i[mgl])
        plt.plot(wav[mgl:mgr],mg_cube[0]*6e14*5+5)
        plt.ylabel('X [Mm]')
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Mg II h')
        
        plt.subplot(235)
        plt.imshow(((laint)), origin='lower', cmap='gray', aspect='auto', extent=[wav[lal], wav[lar-1],-25,25], vmin=0, vmax=falc_i[lal])
        plt.plot(wav[lal:lar],la_cube[0]*5e13*0.8+10)
        plt.ylabel('X [Mm]')
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ly $\rm\alpha$')
        
        plt.subplot(236)
        plt.imshow(((lbint)), origin='lower', cmap='gray', aspect='auto', extent=[wav[lbl], wav[lbr-1],-25,25], vmin=0, vmax=falc_i[lbl])
        plt.plot(wav[lbl:lbr],lb_cube[0]*7e14*0.8+10)
        plt.ylabel('X [Mm]')
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ly $\rm\beta$')
        
        plt.tight_layout()
        plt.savefig(path+name+'.png')
        plt.show()
        
        ###pic2
        
        fig = plt.figure(figsize=(12, 6), dpi=100)
        plt.subplot(231)
        
        def reinterp(dat, wav, nwav):
        
            c8wavint = np.linspace(wav[0],wav[-1], 100)
            # Prepare the interpolated result array
            c8int = np.zeros((dat.shape[0], len(c8wavint)))
            # Interpolate each spectrum (column) in c8_cube
            for i in range(dat.shape[0]):
                c8int[i] = np.interp(c8wavint, wav, dat[i])
            return c8int
        
        c8int = reinterp(c8_cube, wav[c8l:c8r], 100)
        haint = reinterp(ha_cube, wav[hal:har], 100)
        ckint = reinterp(ck_cube, wav[ckl:ckr], 100)
        mgint = reinterp(mg_cube, wav[mgl:mgr], 100)
        laint = reinterp(la_cube, wav[lal:lar], 100)
        lbint = reinterp(lb_cube, wav[lbl:lbr], 100)
        


        
        #c8int = np.interp(c8wavint, wav[c8l:c8r], np.rot90(c8_cube))
        
        plt.imshow((np.log10(c8int)), origin='lower', cmap='gray', aspect='auto', extent=[wav[c8l], wav[c8r-1],-25,25], vmin=-15, vmax=np.log10(falc_i[c8l]))
        plt.plot(wav[c8l:c8r],c8_cube[0]*1.3e13*10)
        plt.ylabel('X [Mm]')
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ca II 8542 $\rm\AA$')
        
        plt.subplot(232)
        plt.imshow((np.log10(haint)), origin='lower', cmap='gray', aspect='auto', extent=[wav[hal], wav[har-1], -25,25], vmin=-15, vmax=np.log10(falc_i[hal]))
        plt.plot(wav[hal:har],ha_cube[0]*1e13*10)
        plt.ylabel('X [Mm]')
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'H$\rm\alpha$')
        
        plt.subplot(233)
        plt.imshow((np.log10(ckint)), origin='lower', cmap='gray', aspect='auto', extent=[wav[ckl], wav[ckr-1],-25,25], vmin=-15, vmax=np.log10(falc_i[ckl]))
        plt.plot(wav[ckl:ckr],ck_cube[0]*1e14*10)
        plt.ylabel('X [Mm]')
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ca II K')
        
        plt.subplot(234)
        plt.imshow((np.log10(mgint)), origin='lower', cmap='gray', aspect='auto', extent=[wav[mgl], wav[mgr-1],-25,25], vmin=-15, vmax=np.log10(falc_i[mgl]))
        plt.plot(wav[mgl:mgr],mg_cube[0]*6e14*5+5)
        plt.ylabel('X [Mm]')
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Mg II h')
        
        plt.subplot(235)
        plt.imshow((np.log10(laint)), origin='lower', cmap='gray', aspect='auto', extent=[wav[lal], wav[lar-1],-25,25], vmin=-15, vmax=np.log10(falc_i[lal]))
        plt.plot(wav[lal:lar],la_cube[0]*5e13*0.8+10)
        plt.ylabel('X [Mm]')
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ly $\rm\alpha$')
        
        plt.subplot(236)
        plt.imshow((np.log10(lbint)), origin='lower', cmap='gray', aspect='auto', extent=[wav[lbl], wav[lbr-1],-25,25], vmin=-15, vmax=np.log10(falc_i[lbl]))
        plt.plot(wav[lbl:lbr],lb_cube[0]*7e14*0.8+10)
        plt.ylabel('X [Mm]')
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ly $\rm\beta$')
        
        plt.tight_layout()
        plt.savefig(path+name+'_log.png')
        plt.show()
    
    else:
        fig = plt.figure(figsize=(12, 6), dpi=100)
        plt.subplot(231)
        
        def reinterp(dat, wav, nwav):
        
            c8wavint = np.linspace(wav[0],wav[-1], 100)
            # Prepare the interpolated result array
            c8int = np.zeros((dat.shape[0], len(c8wavint)))
            # Interpolate each spectrum (column) in c8_cube
            for i in range(dat.shape[0]):
                c8int[i] = np.interp(c8wavint, wav, dat[i])
            return c8int
        
        c8int = reinterp(c8_cube, wav[c8l:c8r], 100)
        haint = reinterp(ha_cube, wav[hal:har], 100)
        ckint = reinterp(ck_cube, wav[ckl:ckr], 100)
        mgint = reinterp(mg_cube, wav[mgl:mgr], 100)
        laint = reinterp(la_cube, wav[lal:lar], 100)
        lbint = reinterp(lb_cube, wav[lbl:lbr], 100)

        
        #c8int = np.interp(c8wavint, wav[c8l:c8r], np.rot90(c8_cube))
        
        plt.imshow((np.rot90(c8int)), origin='lower', cmap='gray', aspect='auto', extent=[-25,25, wav[c8l], wav[c8r-1]], vmin=0, vmax=(falc_i[c8l]))
        plt.plot(c8_cube[0]*2e8-20, wav[c8l:c8r])
        plt.xlabel('X [Mm]')
        plt.ylabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ca II 8542 $\rm\AA$')
        
        plt.subplot(232)
        plt.imshow((np.rot90(haint)), origin='lower', cmap='gray', aspect='auto', extent=[-25,25, wav[hal], wav[har-1]], vmin=0, vmax=(falc_i[hal]))
        plt.plot(ha_cube[0]*2e8-20, wav[hal:har])
        plt.xlabel('X [Mm]')
        plt.ylabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'H$\rm\alpha$')
        
        plt.subplot(233)
        plt.imshow((np.rot90(ckint)), origin='lower', cmap='gray', aspect='auto', extent=[-25,25, wav[ckl], wav[ckr-1]], vmin=0, vmax=(falc_i[ckl]))
        plt.plot(ck_cube[0]*2e9-20, wav[ckl:ckr])
        plt.xlabel('X [Mm]')
        plt.ylabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ca II K')
        
        plt.subplot(234)
        plt.imshow((np.rot90(mgint)), origin='lower', cmap='gray', aspect='auto', extent=[-25,25, wav[mgl], wav[mgr-1]], vmin=0, vmax=(falc_i[mgl])*2)
        plt.plot(mg_cube[0]*4e9-22, wav[mgl:mgr])
        plt.xlabel('X [Mm]')
        plt.ylabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Mg II h')
        
        plt.subplot(235)
        plt.imshow((np.rot90(laint)), origin='lower', cmap='gray', aspect='auto', extent=[-25,25, wav[lal], wav[lar-1]], vmin=0, vmax=(falc_i[lal])*4)
        plt.plot(la_cube[0]*4e11-22, wav[lal:lar])
        plt.xlabel('X [Mm]')
        plt.ylabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ly $\rm\alpha$')
        
        plt.subplot(236)
        plt.imshow((np.rot90(lbint)), origin='lower', cmap='gray', aspect='auto', extent=[-25,25, wav[lbl], wav[lbr-1]], vmin=0, vmax=(falc_i[lbl])*5)
        plt.plot(lb_cube[0]*3e13-22, wav[lbl:lbr])
        plt.xlabel('X [Mm]')
        plt.ylabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ly $\rm\beta$')
        
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(path+name+'.png')
        plt.show()
        
        #fig log 2
        
        fig = plt.figure(figsize=(12, 6), dpi=100)
        plt.subplot(231)
        
        #c8int = np.interp(c8wavint, wav[c8l:c8r], np.rot90(c8_cube))
        
        plt.imshow(np.log10(np.rot90(c8int)), origin='lower', cmap='gray', aspect='auto', extent=[-25,25, wav[c8l], wav[c8r-1]], vmin=-15, vmax=np.log10(falc_i[c8l]))
        plt.plot(c8_cube[0]*2e8-20, wav[c8l:c8r])
        plt.xlabel('X [Mm]')
        plt.ylabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ca II 8542 $\rm\AA$')
        
        plt.subplot(232)
        plt.imshow(np.log10(np.rot90(haint)), origin='lower', cmap='gray', aspect='auto', extent=[-25,25, wav[hal], wav[har-1]], vmin=-15, vmax=np.log10(falc_i[hal]))
        plt.plot(ha_cube[0]*2e8-20, wav[hal:har])
        plt.xlabel('X [Mm]')
        plt.ylabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'H$\rm\alpha$')
        
        plt.subplot(233)
        plt.imshow(np.log10(np.rot90(ckint)), origin='lower', cmap='gray', aspect='auto', extent=[-25,25, wav[ckl], wav[ckr-1]], vmin=-15, vmax=np.log10(falc_i[ckl]))
        plt.plot(ck_cube[0]*2e9-20, wav[ckl:ckr])
        plt.xlabel('X [Mm]')
        plt.ylabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ca II K')
        
        plt.subplot(234)
        plt.imshow(np.log10(np.rot90(mgint)), origin='lower', cmap='gray', aspect='auto', extent=[-25,25, wav[mgl], wav[mgr-1]], vmin=-15, vmax=np.log10(falc_i[mgl]*2))
        plt.plot(mg_cube[0]*4e9-22, wav[mgl:mgr])
        plt.xlabel('X [Mm]')
        plt.ylabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Mg II h')
        
        plt.subplot(235)
        plt.imshow(np.log10(np.rot90(laint)), origin='lower', cmap='gray', aspect='auto', extent=[-25,25, wav[lal], wav[lar-1]], vmin=-15, vmax=np.log10(falc_i[lal]*4))
        plt.plot(la_cube[0]*4e11-22, wav[lal:lar])
        plt.xlabel('X [Mm]')
        plt.ylabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ly $\rm\alpha$')
        
        plt.subplot(236)
        plt.imshow(np.log10(np.rot90(lbint)), origin='lower', cmap='gray', aspect='auto', extent=[-25,25, wav[lbl], wav[lbr-1]], vmin=-15, vmax=np.log10(falc_i[lbl]*5))
        plt.plot(lb_cube[0]*3e13-22, wav[lbl:lbr])
        plt.xlabel('X [Mm]')
        plt.ylabel(r'Wavelength [$\rm\AA$]')
        plt.title(r'Ly $\rm\beta$')
        
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(path+name+'_log.png')
        plt.show()
        
    
        
    
    
    #plot2
    maps = np.load(path+'maps_'+name[6:9]+'.npy')
    y = np.linspace(10,40,100)
    x = np.linspace(-24,24,100)
    
    if prom:
        pp = 'prominence'
    else:
        pp = 'fillament'
        
        
    for i in range(100):
        plt.figure(figsize=(12,3))
        plt.subplot(131)
        plt.plot(wav[hal:har],ha_cube[i])
        #plt.plot(wav[mgl:mgr],mg_cube[i])
        plt.xlabel(r'Wavelength [$\rm\AA$]')
        plt.ylabel('Intensity')
        plt.title(r'H$\alpha$')

        plt.ylim(1e-11,4e-8)
        plt.xticks(rotation=30)
        plt.subplot(132)
        plt.imshow(maps[0], origin='lower', cmap='seismic', extent=[-24,24,10,40])
        if prom:
            plt.hlines(y[i],-24,24, lw=2, color='black')
        else:
            plt.vlines(x[i],10,40, lw=2, color='black')
        plt.xlabel(r'X [Mm]')
        plt.ylabel('Z [Mm]')
        plt.title('Velocity')
        plt.colorbar(label=r'Velocity [m/s]')
        plt.subplot(133)
        plt.imshow(maps[1], origin='lower', cmap='viridis', extent=[-24,24,10,40])
        if prom:
            plt.hlines(y[i],-24,24, lw=2, color='black')
        else:
            plt.vlines(x[i],10,40, lw=2, color='black')
        plt.xlabel(r'X [Mm]')
        plt.ylabel('Z [Mm]')
        plt.title('Density')
        plt.colorbar(label=r'Density [kg/m$^3$]')
        plt.tight_layout()
        
        
        plt.savefig(f'{path}img/{pp}_{i:03}_mg.png')
        plt.close()
        
    
    
    ha_tau[i] = gettau(hal,har, chi, eta, sca, J, y)
    c8_tau[i] = gettau(c8l,c8r, chi, eta, sca, J, y)
    ck_tau[i] = gettau(ckl,ckr, chi, eta, sca, J, y)
    mg_tau[i] = gettau(mgl,mgr, chi, eta, sca, J, y)
    la_tau[i] = gettau(lal,lar, chi, eta, sca, J, y)
    lb_tau[i] = gettau(lbl,lbr, chi, eta, sca, J, y)
    
    

    plt.figure(figsize=(12,3))
    plt.suptitle('frame '+name[6:9])
    plt.subplot(131)
    plt.imshow(maps[0], origin='lower', cmap='seismic', extent=[-24,24,10,40])
    plt.plot(np.linspace(-24,24,101),y[c8_tau[:,25].astype(int)]/1e6, label='c8')
    plt.plot(np.linspace(-24,24,101),y[ha_tau[:,25].astype(int)]/1e6, label='ha')
    plt.plot(np.linspace(-24,24,101),y[ck_tau[:,11].astype(int)]/1e6, label='ck')
    plt.plot(np.linspace(-24,24,101),y[mg_tau[:,61].astype(int)]/1e6, label='mg')
    plt.plot(np.linspace(-24,24,101),y[la_tau[:,36].astype(int)]/1e6, label='la')
    #plt.plot(np.linspace(-24,24,101),y[lb_tau[:,20].astype(int)]/1e6, label='lb')
    plt.legend()
    plt.xlabel('Mm')
    plt.ylabel('Mm')
    #plt.vlines(xtd/32-24, 0,40, linewidth=0.5)
    plt.ylim(10, 40)
    plt.xlim(-10, 10)
    plt.title('Vlos')
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(maps[1], origin='lower', cmap='viridis', extent=[-24,24,10,40])
    plt.plot(np.linspace(-24,24,101),y[c8_tau[:,25].astype(int)]/1e6, label='c8')
    plt.plot(np.linspace(-24,24,101),y[ha_tau[:,25].astype(int)]/1e6, label='ha')
    plt.plot(np.linspace(-24,24,101),y[ck_tau[:,11].astype(int)]/1e6, label='ck')
    plt.plot(np.linspace(-24,24,101),y[mg_tau[:,61].astype(int)]/1e6, label='mg')
    plt.plot(np.linspace(-24,24,101),y[la_tau[:,36].astype(int)]/1e6, label='la')
    plt.xlabel('Mm')
    plt.ylabel('Mm')
    plt.ylim(10, 40)
    plt.xlim(-10, 10)
    plt.colorbar()
    #plt.vlines(xtd/32-24, 0,40, linewidth=0.5)
    plt.title(r'$\rho$')
    plt.subplot(133)
    plt.imshow(maps[2], origin='lower', cmap='hot', extent=[-24,24,10,40])
    plt.plot(np.linspace(-24,24,101),y[c8_tau[:,25].astype(int)]/1e6, label='c8')
    plt.plot(np.linspace(-24,24,101),y[ha_tau[:,25].astype(int)]/1e6, label='ha')
    plt.plot(np.linspace(-24,24,101),y[ck_tau[:,11].astype(int)]/1e6, label='ck')
    plt.plot(np.linspace(-24,24,101),y[mg_tau[:,61].astype(int)]/1e6, label='mg')
    plt.plot(np.linspace(-24,24,101),y[la_tau[:,36].astype(int)]/1e6, label='la')
    plt.xlabel('Mm')
    plt.ylabel('Mm')
    #plt.vlines(xtd/32-24, 0,40, linewidth=0.5)
    plt.title('Temp')
    plt.ylim(10, 40)
    plt.xlim(-10, 10)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()





