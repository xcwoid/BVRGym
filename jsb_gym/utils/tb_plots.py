import matplotlib.pyplot as plt
# fix IEEE pdf figure font issue  
plt.rcParams['pdf.fonttype'] = 42
import numpy as np
import os 
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Env_plots():
    def __init__(self, conf):
        self.conf = conf

    def load_tb(self, path):
        self.cont = os.listdir(path)  
        self.events = []
        for i in tqdm(range(len(self.cont))):
            event_acc = event_accumulator.EventAccumulator(path+ "/" + self.cont[i])
            event_acc.Reload()
            self.events.append(event_acc)

    def tb2list(self, tb_scalar):
        return [(s.step, s.value) for s in tb_scalar]

    def add2axis(self, axs, time_s, scalar, label, grid= False, xlab= None, ylab=None, legend= False, ylim = None, legend_loc = None):
        scalar = self.tb2list(scalar)
        time_s = self.tb2list(time_s)

        time_s, scalar = self.resize_list( time_s, scalar)
    
        axs.plot([x[1] for x in time_s], [x[1] for x in scalar], label=label, alpha=0.5)
        
        if grid:
            axs.grid()
        if xlab != None:
            axs.set_xlabel(xlab)
        if ylab != None:
            axs.set_ylabel(ylab)
        if legend:
            if legend_loc != None:
                axs.legend(loc=legend_loc)
            else:
                axs.legend()
                
        
        if ylim != None:
            axs.set_ylim(ylim[0], ylim[1])

    def resize_list(self, scalarx, scalary):
        if len(scalarx) == len(scalary):
            pass
        elif len(scalarx) < len(scalary):
            print('Warning')
            print('Length of X and Y axis are different:',  len(scalarx), len(scalary))
            scalary = scalary[:len(scalarx)]
        elif len(scalarx) > len(scalary):
            print('Warning')
            print('Length of X and Y axis are different:',  len(scalarx), len(scalary))
            scalarx = scalarx[:len(scalary)]

        
        return scalarx, scalary

    def add2axis2d(self, axs, axs_i, scalarx, scalary, label, grid= False, xlab= None, ylab=None, legend= False, ylim = None, legend_loc = None, start_from = 0 ):
        scalarx = self.tb2list(scalarx)
        scalary = self.tb2list(scalary)

        scalarx, scalary = self.resize_list(scalarx, scalary)

        axs[axs_i[0],axs_i[1]].plot([x[1] for x in scalarx[start_from:]], [x[1] for x in scalary[start_from:]], label=label, alpha=0.5)
        if grid:
            axs[axs_i[0],axs_i[1]].grid()
        if xlab != None:
            axs[axs_i[0],axs_i[1]].set_xlabel(xlab)
        if ylab != None:
            axs[axs_i[0],axs_i[1]].set_ylabel(ylab)
        if legend:
            if legend_loc != None:
                axs[axs_i[0],axs_i[1]].legend(loc=legend_loc)
            else:
                axs[axs_i[0],axs_i[1]].legend()
                
        
        if ylim != None:
            axs[axs_i[0],axs_i[1]].set_ylim(ylim[0], ylim[1])

    def plot_missile(self, show = False , name = '/AIM'):
        print('Plot AIM')
        logs_load = self.conf.logs['log_path']+name
        logs_save = self.conf.logs['save_to']+name + '/AIM.pdf'
        print('Data taken from : ', logs_load)
        print('Saved to: ', logs_save)
        self.load_tb(logs_load)
        # Get scalar data
        for idx, i in enumerate(self.events):
            #print(i)
            fig, axs = plt.subplots(3,2)            
            
            self.add2axis(axs[0,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('phi'), label='$\phi$    [deg]', legend=False)
            self.add2axis(axs[0,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('phi_ref'), label='$\phi_{ref}$ [deg]', grid=True, legend=True, ylim= [-10, 10])

            self.add2axis(axs[1,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('theta'), label='$\Theta$    [deg]', legend=True)
            self.add2axis(axs[1,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('theta_ref'), label='$\Theta_{ref}$ [deg]', grid=True, legend=True, legend_loc= 'upper right')
            
            self.add2axis(axs[2,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('psi'), label='$\psi$    [deg]')
            self.add2axis(axs[2,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('psi_ref'), label='$\psi_{ref}$ [deg]', grid=True, legend=True, xlab= 'Time Step [sec]')

            self.add2axis(axs[0,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('altitude'), label='$Alt$ [m]', legend=True, grid=True, legend_loc= 'upper right')
            #self.add2axis(axs, axs_i=[0,0], scalar= i.Scalars('Tgt_psi'), label='Tgt_psi', grid=True, legend=True)

            self.add2axis(axs[1,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('Mach'), label='Mach', legend=True, grid= True, ylim= [0, 5])
            #self.add2axis(axs, axs_i=[1,0], scalar= i.Scalars('Tgt_get_lat_gc_deg'), label='Tgt_get_lat_gc_deg', grid=True, legend=True)

            self.add2axis(axs[2,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('Target'), label='Dist. to Target [m]', legend=True, grid= True, ylim= [0, 100e3], xlab= 'Time Step [sec]', legend_loc= 'upper right')
            #self.add2axis(axs, axs_i=[2,0], scalar= i.Scalars('Tgt_get_long_gc_deg'), label='Tgt_get_long_gc_deg', grid=True, legend=True)

            #plt.show()
            fig.savefig(logs_save, bbox_inches="tight")

    def plot_f16(self, show = False, name = '/F16' ):
        print('Plot F16')
        logs_load = self.conf.logs['log_path']+name
        logs_save = self.conf.logs['save_to']+name + '/F16.pdf'
        print('Data taken from : ', logs_load)
        print('Saved to: ', logs_save)

        self.load_tb(logs_load)
        # Get scalar data
        for idx, i in enumerate(self.events):
            #print(i)
            fig, axs = plt.subplots(3,2)            
            

            self.add2axis(axs[0,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('altitude'), label='$Alt$    [m]', legend=True, grid=False, ylim= [0.0, 11e3])
            self.add2axis(axs[0,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('altitude_ref'), label='$Alt_{ref}$ [m]', legend=True, grid=True, ylim= [0.0, 11e3])
            self.add2axis(axs[1,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('Mach'), label='Mach', legend=True, grid= True)

            self.add2axis(axs[0,1], time_s= i.Scalars('sim_time'),scalar= i.Scalars('phi'), label='$\phi$    [deg]', legend=False)
            self.add2axis(axs[0,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('phi_ref'), label='$\phi_{ref}$ [deg]', grid=True, legend=True)

            self.add2axis(axs[1,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('theta'), label='$\Theta$    [deg]', legend=True)
            self.add2axis(axs[1,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('theta_ref'), label='$\Theta_{ref}$ [deg]', grid=True, legend=True)
            
            self.add2axis(axs[2,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('psi'), label='$\psi$    [deg]')
            self.add2axis(axs[2,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('psi_ref'), label='$\psi_{ref}$ [deg]', grid=True, legend=True, ylim= [0, 360], xlab= 'Time Step [sec]', legend_loc= 'upper right')

            
            #self.add2axis(axs, axs_i=[0,0], scalar= i.Scalars('Tgt_psi'), label='Tgt_psi', grid=True, legend=True)

            
            #self.add2axis(axs, axs_i=[1,0], scalar= i.Scalars('Tgt_get_lat_gc_deg'), label='Tgt_get_lat_gc_deg', grid=True, legend=True)

            self.add2axis(axs[2,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('rudder_cmd'), label='Rudder', legend=True, grid= True)
            self.add2axis(axs[2,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('elevator_cmd'), label='Elevator', legend=True, grid= True)
            self.add2axis(axs[2,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('aileron_cmd'), label='Aileron', legend=True, grid= True, xlab= 'Time Step [sec]', ylim= [-2, 2], legend_loc= 'upper right')
            #self.add2axis(axs, axs_i=[2,0], scalar= i.Scalars('Tgt_get_long_gc_deg'), label='Tgt_get_long_gc_deg', grid=True, legend=True)

            #plt.show()
            # fig.savefig(self.conf.logs['save_to'] + '/dog.pdf', bbox_inches="tight")
            fig.savefig( logs_save, bbox_inches="tight")

    def plot_env(self, show = False, name = '/ENV'  ):
        logs_load = self.conf.logs['log_path']+name
        logs_save = self.conf.logs['save_to']+name + '/ENV.pdf'
        print('Data taken from : ', logs_load)
        print('Saved to: ', logs_save)

        self.load_tb(logs_load)
        # Get scalar data
        for idx, i in enumerate(self.events):
            #print(i)
            fig, axs = plt.subplots(2,2)            
            
            # Altitude
            self.add2axis(axs[0,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('M_alt'), label='$Alt_{M}$ [m]', legend=True)
            self.add2axis(axs[0,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('Tgt_alt'), label='$Alt_{F16}$ [m]', grid=True, legend=True)

            # Vel 
            self.add2axis(axs[1,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('M_Mach'), label='$Mach_{M}$', legend=True)
            self.add2axis(axs[1,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('Tgt_Mach'), label='$Mach_{F16}$', grid=True, legend=True)
            
            # distance
            #self.add2axis(axs[2,0], time_s= i.Scalars('sim_time'), scalar= i.Scalars('Target'), label='Dist. to target', grid=True, legend=True)

            # distance 
            #self.add2axis(axs[0,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('M_psi'), label='Missile $\psi$ [deg]', legend=True)
            #self.add2axis(axs[0,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('Tgt_psi'), label='F-16 $\psi$ [deg]', grid=True, legend=True, legend_loc= 'lower right')

            self.add2axis(axs[0,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('M_get_lat_gc_deg'), label='$Latitude_{M}$', legend=True)
            self.add2axis(axs[0,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('Tgt_get_lat_gc_deg'), label='$Latitude_{F16}$', grid=True, legend=True)

            self.add2axis(axs[1,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('M_get_long_gc_deg'), label='Longitude_{M}', legend=True)
            self.add2axis(axs[1,1], time_s= i.Scalars('sim_time'), scalar= i.Scalars('Tgt_get_long_gc_deg'), label='Longitude_{F16}', grid=True, legend=True, legend_loc= 'lower right')

            #plt.show()
            fig.savefig(logs_save, bbox_inches="tight")

    def plot_dog(self, show = False ):
        self.load_tb(path= self.conf.logs['path'])
        # Get scalar data
        for idx, i in enumerate(self.events):
            #print(i)
            fig, axs = plt.subplots(2,2)            
            
            # Altitude
            #self.add2axis(axs, axs_i=[0,0], scalar= i.Scalars('M_alt'), label='Missile alt [m]', legend=True)
            #self.add2axis(axs, axs_i=[0,0], scalar= i.Scalars('Tgt_alt'), label='F-16 Alt [m]', grid=True, legend=True)

            # Vel 
            #self.add2axis(axs, axs_i=[1,0], scalar= i.Scalars('M_Mach'), label='Missile Mach [ ]', legend=True)
            #self.add2axis(axs, axs_i=[1,0], scalar= i.Scalars('Tgt_Mach'), label=' F-16 Mach [ ]', grid=True, legend=True)
            
            # distance 
            #self.add2axis(axs, axs_i=[2,0], scalar= i.Scalars('Target'), label='Dist. to target', grid=True, legend=True)

            # distance 
            #self.add2axis(axs, axs_i=[0,1], scalar= i.Scalars('M_psi'), label='Missile $\psi$ [deg]', legend=True)
            #self.add2axis(axs, axs_i=[0,1], scalar= i.Scalars('Tgt_psi'), label='F-16 $\psi$ [deg]', grid=True, legend=True, legend_loc= 'lower right')

            self.add2axis2d(axs, axs_i=[0,0], scalarx= i.Scalars('f16_long'), scalary= i.Scalars('f16_lat'), label='f16', legend=True)
            self.add2axis2d(axs, axs_i=[0,0], scalarx= i.Scalars('f16r_long'), scalary= i.Scalars('f16r_lat'), label='f16r', legend=True)
            #self.add2axis(axs, axs_i=[1,1], scalar= i.Scalars('f16r_lat'), label='f16r_lat', grid=True, legend=True)
            self.add2axis2d(axs, axs_i=[0,0], scalarx= i.Scalars('aim1r_long'), scalary= i.Scalars('aim1r_lat'), label='aim1r', legend=True)
            self.add2axis2d(axs, axs_i=[0,0], scalarx= i.Scalars('aim2r_long'), scalary= i.Scalars('aim2r_lat'), label='aim2r', legend=True, grid= True)

            
            #plt.show()
            fig.savefig(self.conf.logs['save_to'] + '/dog.pdf', bbox_inches="tight")
        #plt.show()

class Common_Utils(object):
    def __init__(self):
        from jsb_gym.utils.utils import Geo
        self.geo = Geo()
        self.lat_ref = None
        self.long_ref = None

    def load_tb(self, path):
        self.cont = os.listdir(path)  
        self.events = []
        for i in tqdm(range(len(self.cont))):
            event_acc = event_accumulator.EventAccumulator(path+ "/" + self.cont[i])
            event_acc.Reload()
            self.events.append(event_acc)

    def tb2list(self, tb_scalar):
        return [(s.step, s.value) for s in tb_scalar]

    def pp3d(self, x, y , z, time_stamped= True, cut_to_smallest = True):
        # preprocess for 3d plots 
        x = self.tb2list(x)
        y = self.tb2list(y)
        z = self.tb2list(z)
        mv = min([len(x), len(y), len(z)])

        # adjust to the smallest 
        if cut_to_smallest:
            x = x[:mv]
            y = y[:mv]
            z = z[:mv]
        
        # transform to meters 
        x, y  = self.geo.geo2km(lat= x, long= y, lat_ref = self.lat_ref, long_ref = self.long_ref, time_stamped= time_stamped)
        if time_stamped:
            z = [(i[0],i[1]*1e-3) for i in z]
        else:
            z = [i*1e-3 for i in z]
        return x,y,z 

class Dog_3D_plots(object):
    def __init__(self, conf):
        self.conf = conf
        self.cu = Common_Utils()
        self.cu.lat_ref  = self.conf.f16['lat']
        self.cu.long_ref = self.conf.f16['long']     

    def show_3D(self, name = '/DOG'):
        fig = plt.figure()
        axs = plt.axes(projection ='3d')
        logs_load = self.conf.logs['log_path']+name
        logs_save = self.conf.logs['save_to']+name + '/DOG.pdf'
        
        self.cu.load_tb(logs_load)

        for idx, i in enumerate(self.cu.events):
            #print(idx)
            self.add2axis3d(axs, x= i.Scalars('f16_lat'), y= i.Scalars('f16_long'), z=  i.Scalars('f16_alt'), label='f16', legend=True, c = 'cornflowerblue')
            self.add2axis3d(axs, x= i.Scalars('f16r_lat'), y= i.Scalars('f16r_long'), z=  i.Scalars('f16r_alt'), label='f16r', legend=True, c = 'crimson', set_lim = True)
            try:
                self.add2axis3d(axs, x= i.Scalars('aim1r_lat'), y= i.Scalars('aim1r_long'), z=  i.Scalars('aim1r_alt'), label='aim1r', legend=True, c = 'salmon', alpha= 0.2)
            except:
                print('Missile aim1r not Launched')    
            try:
                self.add2axis3d(axs, x= i.Scalars('aim2r_lat'), y= i.Scalars('aim2r_long'), z=  i.Scalars('aim2r_alt'), label='aim2r', legend=True, c = 'salmon', alpha= 0.2)
            except:
                print('Missile aim2r not Launched')    

            try:
                self.add2axis3d(axs, x= i.Scalars('aim1_lat'), y= i.Scalars('aim1_long'), z=  i.Scalars('aim1_alt'), label='aim1', legend=True, c = 'aqua', alpha= 0.2)
            except:
                print('Missile aim1 not Launched')
            try:
                self.add2axis3d(axs, x= i.Scalars('aim2_lat'), y= i.Scalars('aim2_long'), z=  i.Scalars('aim2_alt'), label='aim2', legend=True, c = 'aqua', alpha= 0.2)
            except:
                print('Missile aim2 not Launched')    
        #x, y  = self.pltk.geo2km(lat= logs.logs['f16']['lat'], long= logs.logs['f16']['long'])
        #z = [i*1e-3 for i in logs.logs['f16']['alt']]
        #
        plt.show()
        #fig.savefig(self.conf.logs['save_to'] + '/dog.pdf', bbox_inches="tight")

    def add2axis3d(self, axs, x, y, z, label, grid= False, xlab= None, ylab=None, legend= False, set_lim = None , legend_loc = None, start_from = 0 , c = 'cornflowerblue', alpha = 0.4):
        
        x, y, z = self.cu.pp3d(x,y,z)
        #print(x)
        poly3dCollection = self.get_poly3dCollection(x, y, z, c= c , alpha= alpha )
        axs.add_collection3d(poly3dCollection)
        
        #axs.plot([x[1] for x in scalarx[start_from:]], [x[1] for x in scalary[start_from:]], label=label, alpha=0.5)
        #if grid:
        #    axs[axs_i[0],axs_i[1]].grid()
        #if xlab != None:
        #    axs[axs_i[0],axs_i[1]].set_xlabel(xlab)
        #if ylab != None:
        #    axs[axs_i[0],axs_i[1]].set_ylabel(ylab)
        #if legend:
        #    if legend_loc != None:
        #        axs[axs_i[0],axs_i[1]].legend(loc=legend_loc)
        #    else:
        #        axs[axs_i[0],axs_i[1]].legend()
        if set_lim != None:
            axs.set_xlim(-200, 200)
            axs.set_ylim(-200, 200)
            axs.set_zlim(0, 15)
            #axs.view_init( 50 , 40)
            axs.view_init( 30 , 20)
        
        #if ylim != None:
        #    axs[axs_i[0],axs_i[1]].set_ylim(ylim[0], ylim[1])

    def get_poly3dCollection(self, f16_lat,f16_long,f16_alt, c, alpha, spars = 10, h = 0.0):
        f16_lat = [i [1] for i in f16_lat]
        #f16_lat = f16_lat[::spars]
        #f16_long = f16_long[::spars]
        f16_long = [i [1] for i in f16_long]
        f16_alt = [i [1] for i in f16_alt]
        #f16_alt = f16_alt[::spars]
        # Code to convert data in 3D polygons
        v = []
        for k in range(0, len(f16_lat) - 1):
            x = [f16_lat[k], f16_lat[k+1], f16_lat[k+1], f16_lat[k]]
            y = [f16_long[k], f16_long[k+1], f16_long[k+1], f16_long[k]]
            z = [f16_alt[k], f16_alt[k+1],       h,     h]
            #print(z)
            v.append(list(zip(x, y, z))) 
        poly3dCollection = Poly3DCollection(v, color = c, alpha=alpha, linewidths=0.0)
        return poly3dCollection

class Runs_plots():
    def __init__(self):
        
        self.path = []

    def load_tb(self, event_file, sc):
        event_acc = event_accumulator.EventAccumulator(event_file)
        event_acc.Reload()
        loss_values = event_acc.Scalars(sc) 
        steps = [event.step for event in loss_values]
        value = [event.value for event in loss_values]
        return steps, value

    def tb2list(self, tb_scalar):
        return [(s.step, s.value) for s in tb_scalar]

    def add2axis(self, axs, time_s, scalar, label, grid= False, xlab= None, ylab=None, legend= False, ylim = None, legend_loc = None):
        scalar = self.tb2list(scalar)
        time_s = self.tb2list(time_s)

        time_s, scalar = self.resize_list( time_s, scalar)
    
        axs.plot([x[1] for x in time_s], [x[1] for x in scalar], label=label, alpha=0.5)
        
        if grid:
            axs.grid()
        if xlab != None:
            axs.set_xlabel(xlab)
        if ylab != None:
            axs.set_ylabel(ylab)
        if legend:
            if legend_loc != None:
                axs.legend(loc=legend_loc)
            else:
                axs.legend()
                
        
        if ylim != None:
            axs.set_ylim(ylim[0], ylim[1])

    def resize_list(self, scalarx, scalary):
        if len(scalarx) == len(scalary):
            pass
        elif len(scalarx) < len(scalary):
            print('Warning')
            print('Length of X and Y axis are different:',  len(scalarx), len(scalary))
            scalary = scalary[:len(scalarx)]
        elif len(scalarx) > len(scalary):
            print('Warning')
            print('Length of X and Y axis are different:',  len(scalarx), len(scalary))
            scalarx = scalarx[:len(scalary)]

        
        return scalarx, scalary


    def get_np_from_tb(self, runs_path, test_type = '1', scalar = 'Loss'):
        arr = os.listdir(runs_path)
        arr1 = []
        for i in arr:
            if i[-3] == test_type:                
                print(runs_path + i)
                step, value = self.load_tb(runs_path + i, sc= scalar)
                arr1.append(value)
        
        y_min = np.min([arr1[0], arr1[1], arr1[2], arr1[3], arr1[4]], axis=0)
        y_max = np.max([arr1[0], arr1[1], arr1[2], arr1[3], arr1[4]], axis=0)
        np.save(runs_path + 'y_min'+ test_type+ scalar+'.npy', y_min)
        np.save(runs_path + 'y_max'+ test_type+ scalar+'.npy', y_max)
        np.save(runs_path + 'step'+ test_type+ scalar+'.npy', step)
    
    def create_plt(self):
        self.fig, self.axs = plt.subplots(2,1)
        
    def plot_np(self, runs_path, test_type = '1', scalar = 'Loss', d = 50, row = 0, c = 'blue', label = None):
        y_min = np.load(runs_path + 'y_min'+ test_type+ scalar+'.npy' )
        y_max = np.load(runs_path + 'y_max'+ test_type+ scalar+'.npy' )
        step = np.load(runs_path + 'step'+ test_type+ scalar+'.npy' )

        self.axs[row].fill_between(step[::d], y_min[::d], y_max[::d], color=c, alpha=0.4, label=label)

    def add_grid(self, r = 0):
        self.axs[r].grid()

    def add_xlabel(self, label, r = 0):
        self.axs[r].set_xlabel(label)

    def add_ylabel(self, label, r = 0):
        self.axs[r].set_ylabel(label)

    def add_legend(self, r = 0):
        self.axs[r].legend()
        # Show the plot
    def show_plt(self, save_to= None):
        if not None:
            self.fig.savefig( save_to, bbox_inches="tight")
        else:
            plt.show()


if __name__ == '__main__':
    runs_path = '../../../runs/'
    rp = Runs_plots()
    #rp.get_np_from_tb(runs_path, test_type = '1', scalar = 'Loss')
    #rp.get_np_from_tb(runs_path, test_type = '1', scalar = 'Test_loss')
    
    #rp.get_np_from_tb(runs_path, test_type = '3', scalar = 'Loss')
    #rp.get_np_from_tb(runs_path, test_type = '3', scalar = 'Test_loss')

    #rp.get_np_from_tb(runs_path, test_type = '5', scalar = 'Loss')
    #rp.get_np_from_tb(runs_path, test_type = '5', scalar = 'Test_loss')

    rp.create_plt()
    d = 15
    rp.plot_np(runs_path, test_type = '1', scalar = 'Loss', c= 'gray', d=d, label= '$BL$')
    rp.plot_np(runs_path, test_type = '3', scalar = 'Loss', c= 'blue', d=d, label= '$T1$')
    rp.plot_np(runs_path, test_type = '5', scalar = 'Loss', c= 'red', d=d, label=  '$T2$')

    rp.plot_np(runs_path, test_type = '1', scalar = 'Test_loss', c= 'gray', d=1, row = 1, label= '$BL$')
    rp.plot_np(runs_path, test_type = '3', scalar = 'Test_loss', c= 'blue', d=1, row = 1, label= '$T1$')
    rp.plot_np(runs_path, test_type = '5', scalar = 'Test_loss', c= 'red', d=1, row = 1, label= '$T2$')

    rp.add_grid(0)
    rp.add_grid(1)

    rp.add_ylabel(r = 0, label= '$Loss$')
    rp.add_ylabel(r = 1, label= '$Test$ $Loss$')

    rp.add_legend(r=0)
    rp.add_legend(r=1)
    rp.show_plt('../plots/SSA/Loss.pdf')
            

