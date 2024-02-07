import matplotlib.pyplot as plt 
plt.rcParams['pdf.fonttype'] = 42
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mpimg
import scipy.ndimage
from matplotlib.cm import ScalarMappable
import matplotlib.colors
#from matplotlib._png import read_png

def plot_env(flight_data, env):
    #plot 1:

    plt.subplot(3, 1, 1)
    plt.title('Lat Long')
    for key in env.missile_block_names:
        plt.plot(flight_data.aim_long[key], flight_data.aim_lat[key], label= key)
        plt.scatter(flight_data.aim_long[key][0], flight_data.aim_lat[key][0])

    plt.plot(flight_data.f16_long, flight_data.f16_lat, label = 'f16')    
    plt.scatter(flight_data.f16_long[0], flight_data.f16_lat[0])
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.title('CBF as policy selector')
    plt.plot(flight_data.policy_log)
    plt.yticks( list(range(len(env.missile_block_names))), env.missile_block_names)
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.title('Miss Distance from Missile perspective')
    for key in env.missile_block_names:
        plt.plot(flight_data.aim_dist[key], label = key)
    plt.grid()
    plt.legend()
    plt.show()

def plot_flight_path(flight_data, env):
    ptk = tools.PlotTools()
    #fig = plt.figure() 
    #ax  = plt.axes(projection ='3d')
    lat_ref = flight_data.f16_lat[0]
    long_ref = flight_data.f16_long[0]
    for key in env.missile_block_names:
        x, y  = ptk.geo2km(lat= flight_data.aim_lat[key], long= flight_data.aim_long[key], lat_ref= lat_ref, long_ref= long_ref)
        plt.plot(y, x, label= key)
        plt.scatter(y[0], x[0])
        
    x, y  = ptk.geo2km(lat= flight_data.f16_lat, long= flight_data.f16_long, lat_ref= lat_ref, long_ref= long_ref)
    plt.plot( y, x, label = 'f16')    
    plt.scatter(y[0], x[0])
    plt.grid()
    plt.axis('equal')
    plt.xlabel('km')
    plt.ylabel('km')
    plt.legend()
    plt.savefig('jsb_gym/figures/CBF/flight_path2D.pdf', bbox_inches="tight")

def plot_policy_log(flight_data, env):
    #plt.clear()
    plt.title('CBF as policy selector')
    plt.plot(flight_data.policy_log)
    plt.yticks( list(range(len(env.missile_block_names))), env.missile_block_names)
    plt.grid()
    plt.savefig('jsb_gym/figures/CBF/policy_log.pdf', bbox_inches="tight")

class EVS_plots():
    def __init__(self):
        self.aim1_pos_lat = None
        self.aim1_pos_long = None
        self.aim2_pos_lat = None
        self.aim2_pos_long = None

    def show_target_dynamics(self, logs):
        fig, axs = plt.subplots(3,1)
        axs[0].plot(logs.sim_time, logs.alt_tgt, label='Alt')
        axs[0].set_ylabel('Altitude [m]')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(logs.sim_time, logs.vel_tgt, label='Vel')
        axs[1].set_ylabel('Mach')
        axs[1].grid()
        axs[1].legend()

        axs[2].plot(logs.long_tgt, logs.lat_tgt, label='Position')
        axs[2].set_ylabel('LatLong')
        axs[2].grid()
        axs[2].legend()

        axs[2].scatter(self.aim1_pos_long, self.aim1_pos_lat)
        #axs[2].scatter(self.aim2_pos_long, self.aim2_pos_lat)
        fig.savefig('jsb_gym/figures/mcts_evs_dev.pdf', bbox_inches="tight")



    def show_dynamics(self, logs):
        fig, axs = plt.subplots(5,2)
        axs[0,0].plot(logs.sim_time, logs.alt, label='Msl')
        axs[0,0].set_ylabel('Altitude [m]')
        axs[0,0].plot(logs.sim_time, logs.alt_tgt, color = 'r', linestyle = '--', label='Tgt')
        axs[0,0].plot(logs.sim_time, logs.alt_ref, linestyle = '-.', label='ref')
        axs[0,0].grid()
        axs[0,0].legend()

        axs[1,0].plot(logs.sim_time, logs.vel, label='Msl')
        axs[1,0].plot(logs.sim_time, logs.vel_tgt, color = 'r', linestyle = '--', label='Tgt')
        axs[1,0].grid()
        axs[1,0].set_ylabel('Mach')
        axs[1,0].legend()
        
        axs[2,0].plot(logs.sim_time, logs.lat, label = 'Msl')
        axs[2,0].plot(logs.sim_time, logs.lat_tgt, color = 'r', linestyle = '--', label = 'Tgt')
        axs[2,0].grid()
        axs[2,0].set_ylabel('Latitude')
        axs[2,0].legend()

        xy = []
        for i in range(len(logs.tgt_north)):
            a = np.array([logs.tgt_north[i],logs.tgt_east[i] ])
            xy.append(np.linalg.norm(a))
        axs[3,0].plot(logs.sim_time, xy, label = 'xy')
        axs[3,0].plot(logs.sim_time, logs.tgt_down, label = 'down')
        axs[3,0].grid()
        axs[3,0].set_ylabel('Relative pos')        
        axs[3,0].legend()

        axs[4,0].plot(logs.sim_time, logs.dist_to_tgt)
        axs[4,0].grid()
        axs[4,0].set_ylabel('Distance to target')
        idx = logs.dist_to_tgt.index(min(logs.dist_to_tgt))
        axs[4,0].scatter(logs.sim_time[idx], logs.dist_to_tgt[idx], label = int(logs.dist_to_tgt[idx]))
        axs[4,0].legend()

        axs[0,1].plot(logs.sim_time, logs.phi)
        axs[0,1].plot(logs.sim_time, logs.phi_ref, color = 'r', linestyle = '--')
        axs[0,1].grid()
        axs[0,1].set_ylabel('Roll')


        axs[1,1].plot(logs.sim_time, logs.theta)
        axs[1,1].plot(logs.sim_time, logs.theta_ref, color = 'r', linestyle = '--')
        axs[1,1].grid()
        axs[1,1].set_ylabel('Pitch')

        axs[2,1].plot(logs.sim_time, logs.psi)
        axs[2,1].plot(logs.sim_time, logs.psi_ref, color = 'r', linestyle = '--')
        axs[2,1].grid()
        axs[2,1].set_ylim([0, 360])
        axs[2,1].set_ylabel('Yaw')

        axs[3,1].plot(logs.long, logs.lat)
        axs[3,1].plot(logs.long_tgt, logs.lat_tgt)
        axs[3,1].scatter(logs.long[0], logs.lat[0])
        axs[3,1].scatter(logs.long_tgt[0], logs.lat_tgt[0])
        axs[3,1].set_ylabel('Latitude North')
        axs[3,1].set_xlabel('Longitude South')
        axs[3,1].grid()
        axs[3,1].set_aspect('equal', adjustable="datalim")
        

        axs[4,1].plot(logs.sim_time, logs.PN_accx, label= 'AccX')
        axs[4,1].plot(logs.sim_time, logs.PN_accy, label= 'AccY')
        axs[4,1].plot(logs.sim_time, logs.PN_accz, label= 'AccZ')
        axs[4,1].grid()
        axs[4,1].set_ylabel('PN')
        axs[4,1].legend()

        fig.savefig('jsb_gym/figures/evs_dev.pdf', bbox_inches="tight")
        #plt.show()

class MCTS_plots():
    def __init__(self):
        pass
    def show(self, logs):
        fig, axs = plt.subplots(3,1)
        
        
        axs[0].plot(logs[0].long, logs[0].lat, label='f16')
        axs[0].plot(logs[1].long, logs[1].lat, label='aim1')
        axs[0].plot(logs[2].long, logs[2].lat, label='aim2')

        axs[0].scatter(logs[0].long[0], logs[0].lat[0])
        axs[0].scatter(logs[1].long[0], logs[1].lat[0])
        axs[0].scatter(logs[2].long[0], logs[2].lat[0])

        axs[0].grid()
        axs[0].legend()
        
        
        axs[1].plot(logs[0].sim_time, logs[0].alt, label='f16')
        axs[1].plot(logs[1].sim_time, logs[1].alt, label='aim1')
        axs[1].plot(logs[2].sim_time, logs[2].alt, label='aim2')

        axs[2].plot(logs[0].sim_time, logs[0].vel, label='f16')
        axs[2].plot(logs[1].sim_time, logs[1].vel, label='aim1')
        axs[2].plot(logs[2].sim_time, logs[2].vel, label='aim2')

        plt.show()


class AIM_plots():
    def __init__(self):
        pass

    def show_dynamics(self, logs):
        fig, axs = plt.subplots(3,2)
        
        axs[0,0].plot(logs.sim_time, logs.phi)
        axs[0,0].plot(logs.sim_time, logs.phi_ref, color = 'r', linestyle = '--')
        axs[0,0].grid()
        axs[0,0].set_ylabel('Roll')


        axs[1,0].plot(logs.sim_time, logs.theta)
        axs[1,0].plot(logs.sim_time, logs.theta_ref, color = 'r', linestyle = '--')
        axs[1,0].grid()
        axs[1,0].set_ylabel('Pitch')

        axs[2,0].plot(logs.sim_time, logs.psi)
        axs[2,0].plot(logs.sim_time, logs.psi_ref, color = 'r', linestyle = '--')
        axs[2,0].grid()
        axs[2,0].set_ylim([0, 360])
        axs[2,0].set_ylabel('Yaw')

        axs[2,1].plot(logs.sim_time, logs.dist)
        axs[2,1].grid()
        axs[2,1].set_ylabel('Distance to target')
        idx = logs.dist.index(min(logs.dist))
        axs[2,1].scatter(logs.sim_time[idx], logs.dist[idx], label = int(logs.dist[idx]))
        axs[2,1].legend()


        fig.savefig('jsb_gym/figures/missile_dev.pdf', bbox_inches="tight")
        plt.show()


class F16_plots():
    def __init__(self):
        pass

    def show_dynamics(self, logs):

        fig, axs = plt.subplots(3,2)
        
        axs[0,0].plot(logs.sim_time, logs.phi)
        axs[0,0].plot(logs.sim_time, logs.phi_ref, color = 'r', linestyle = '--')
        axs[0,0].grid()
        axs[0,0].set_ylabel('Roll')


        axs[1,0].plot(logs.sim_time, logs.theta)
        axs[1,0].plot(logs.sim_time, logs.theta_ref, color = 'r', linestyle = '--')
        axs[1,0].grid()
        axs[1,0].set_ylabel('Pitch')

        axs[2,0].plot(logs.sim_time, logs.psi)
        axs[2,0].plot(logs.sim_time, logs.psi_ref, color = 'r', linestyle = '--')
        axs[2,0].grid()
        axs[2,0].set_ylim([0, 360])
        axs[2,0].set_ylabel('Yaw')


        axs[0,1].plot(logs.sim_time, logs.alt)
        axs[0,1].plot(logs.sim_time, logs.alt_ref, color = 'r', linestyle = '--')
        axs[0,1].grid()
        axs[0,1].set_ylabel('Altitude')

       

        axs[2,1].plot(logs.sim_time, logs.elevator_cmd, label= 'elevator')
        axs[2,1].plot(logs.sim_time, logs.aileron_cmd, label= 'aileron')
        axs[2,1].plot(logs.sim_time, logs.rudder_cmd,  label = 'rudder')
        axs[2,1].hlines(y = 1, xmin = 0, xmax = max(logs.sim_time), color = 'r', linestyle = '--')
        axs[2,1].hlines(y = -1, xmin = 0, xmax = max(logs.sim_time), color = 'r', linestyle = '--')
        axs[2,1].grid()
        axs[2,1].legend()
        axs[2,1].set_ylabel('cmd')


        fig.savefig('jsb_gym/figures/missile_dev.pdf', bbox_inches="tight")
        plt.show()


class testNN_plots():
    def __init__(self):
        from jsb_gym.tools.tools import PlotTools
        self.pltk = PlotTools(lat_ref=59, long_ref= 18)
        self.heading_names = {'sm1': 'N', 'sm075': 'NE', 'sm050': 'E', 'sm025': 'SE', 's00': 'S', 's025': 'SW', 's05':'W', 's075': 'NW'}
        self.c_ = ['blue', 'green', 'red', 'magenta', 'orange', 'brown', 'teal', 'gray']

    def show(self, logs):
        fig, axs = plt.subplots(2,1)
        axs[0].plot( logs.logs['f16']['sim_time'], logs.logs['f16']['alt'], label='F16')
        axs[0].plot(logs.logs['aim1']['sim_time'], logs.logs['aim1']['alt'], label='M1')
        axs[0].plot(logs.logs['aim2']['sim_time'], logs.logs['aim2']['alt'], label='M2')
        axs[0].plot(logs.logs['aim3']['sim_time'], logs.logs['aim3']['alt'], label='M3')
        axs[0].set_ylabel('Altitude [m]')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot( logs.logs['f16']['sim_time'], logs.logs['f16']['vel'], label='F16')
        axs[1].plot(logs.logs['aim1']['sim_time'], logs.logs['aim1']['vel'], label='M1')
        axs[1].plot(logs.logs['aim2']['sim_time'], logs.logs['aim2']['vel'], label='M2')
        axs[1].plot(logs.logs['aim3']['sim_time'], logs.logs['aim3']['vel'], label='M3')
        axs[1].set_ylabel('Mach')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_xlabel('Time [sec]')

        fig.savefig('jsb_gym/figures/testnn.pdf', bbox_inches="tight")

    def show4(self, logs):
        fig, axs = plt.subplots(2,1)
        axs[0].plot( logs.logs['f16']['sim_time'], logs.logs['f16']['alt'], label='F16')
        axs[0].plot(logs.logs['aim1']['sim_time'], logs.logs['aim1']['alt'], label='M1')
        axs[0].plot(logs.logs['aim2']['sim_time'], logs.logs['aim2']['alt'], label='M2')
        axs[0].plot(logs.logs['aim3']['sim_time'], logs.logs['aim3']['alt'], label='M3')
        axs[0].plot(logs.logs['aim4']['sim_time'], logs.logs['aim4']['alt'], label='M4')
        axs[0].set_ylabel('Altitude [m]')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot( logs.logs['f16']['sim_time'], logs.logs['f16']['vel'], label='F16')
        axs[1].plot(logs.logs['aim1']['sim_time'], logs.logs['aim1']['vel'], label='M1')
        axs[1].plot(logs.logs['aim2']['sim_time'], logs.logs['aim2']['vel'], label='M2')
        axs[1].plot(logs.logs['aim3']['sim_time'], logs.logs['aim3']['vel'], label='M3')
        axs[1].plot(logs.logs['aim4']['sim_time'], logs.logs['aim4']['vel'], label='M4')
        axs[1].set_ylabel('Mach')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_xlabel('Time [sec]')

        fig.savefig('jsb_gym/figures/testnn.pdf', bbox_inches="tight")


    def show6(self, logs):
        fig, axs = plt.subplots(2,1)
        axs[0].plot( logs.logs['f16']['sim_time'], logs.logs['f16']['alt'], label='F16')
        axs[0].plot(logs.logs['aim1']['sim_time'], logs.logs['aim1']['alt'], label='M1')
        axs[0].plot(logs.logs['aim2']['sim_time'], logs.logs['aim2']['alt'], label='M2')
        axs[0].plot(logs.logs['aim3']['sim_time'], logs.logs['aim3']['alt'], label='M3')
        axs[0].plot(logs.logs['aim4']['sim_time'], logs.logs['aim4']['alt'], label='M4')
        axs[0].plot(logs.logs['aim5']['sim_time'], logs.logs['aim5']['alt'], label='M5')
        axs[0].plot(logs.logs['aim6']['sim_time'], logs.logs['aim6']['alt'], label='M6')
        axs[0].set_ylabel('Altitude [m]')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot( logs.logs['f16']['sim_time'], logs.logs['f16']['vel'], label='F16')
        axs[1].plot(logs.logs['aim1']['sim_time'], logs.logs['aim1']['vel'], label='M1')
        axs[1].plot(logs.logs['aim2']['sim_time'], logs.logs['aim2']['vel'], label='M2')
        axs[1].plot(logs.logs['aim3']['sim_time'], logs.logs['aim3']['vel'], label='M3')
        axs[1].plot(logs.logs['aim4']['sim_time'], logs.logs['aim4']['vel'], label='M4')
        axs[1].plot(logs.logs['aim5']['sim_time'], logs.logs['aim5']['vel'], label='M5')
        axs[1].plot(logs.logs['aim6']['sim_time'], logs.logs['aim6']['vel'], label='M6')
        axs[1].set_ylabel('Mach')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_xlabel('Time [sec]')

        fig.savefig('jsb_gym/figures/testnn.pdf', bbox_inches="tight")


    def show_circle(self, logs, t, side_box = True):

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.cm import ScalarMappable

        import matplotlib.colors
        
        #xx = logs.logs['f16']['sim_time'][t]
        print(logs.logs['f16']['psi'][t])
        keys = [key for key in logs.est['f16'][0]]

        est = {}
        for key in keys:            
            x = self.get_est_data(logs.est['f16'], key)
            est[self.heading_names[key]] = x[t]
        
        Salary = [1]*8
  
        val = [0]*8
        #val = [(i +1)*10 for i in val] 
        val = [(est[key] +1)*20 for key in est] 
        
        # Setting labels for items in Chart
        Employee = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        Salary = [1]*8
        
        explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
        
        # Pie Chart
        cmap = plt.cm.RdYlGn
        norm = matplotlib.colors.Normalize(vmin= 0 , vmax= 20)
        colors = cmap(norm(val))
        plt.pie(Salary, colors=colors, labels=Employee,
                explode=explode, startangle=112, counterclock=False)
        # draw circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        # Adding Circle in Pie chart
        fig.gca().add_artist(centre_circle)        
        # Adding Title of chart
        plt.title('Time ' + str(round(logs.logs['f16']['sim_time'][t])) + ' [sec]')
        plt.grid()
        #fig.gca().add_artist(self.add_f16_img(0, 0, 0.15, 0.0, fig='jsb_gym/figures/Afterburner_DS/f16_up.png'))
        cbaxes = fig.add_axes([0.85, 0.1, 0.05, 0.7])
        if side_box:
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ticks= list(np.linspace( 0, 20, 11)), orientation="vertical", cax=cbaxes)
        fig.savefig('jsb_gym/figures/NESW'+str(t)+'.pdf', bbox_inches="tight")
        #plt.show()
        
    def add_f16_img(self,x, y, zoom, rot , fig):
        img = mpimg.imread(fig)
        img = scipy.ndimage.rotate(img, rot)
        imageBox = OffsetImage(img, zoom = zoom)
        xy = [0,0]
        ab = AnnotationBbox(imageBox, xy, xybox=(x,y), boxcoords='offset points', frameon=False)
        return ab

    def show_path(self, logs):
        import matplotlib.ticker as ticker

        fig, axs = plt.subplots(1,1)
        from matplotlib.patches import Rectangle
        from matplotlib.cm import ScalarMappable
        # path
        x0, y0  = self.pltk.geo2km(lat= logs.logs['f16']['lat'], long= logs.logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16 ', color = self.c_[0])
        
        x1, y1  = self.pltk.geo2km(lat= logs.logs['aim1']['lat'], long= logs.logs['aim1']['long'])
        axs.plot(x1, y1, label='Flight Traj. M1', linestyle = '-', alpha = 0.8 , color = self.c_[4])

        x2, y2  = self.pltk.geo2km(lat= logs.logs['aim2']['lat'], long= logs.logs['aim2']['long'])
        axs.plot(x2, y2, label='Flight Traj. M2', linestyle = '-', alpha = 0.8, color = self.c_[1])

        x3, y3  = self.pltk.geo2km(lat= logs.logs['aim3']['lat'], long= logs.logs['aim3']['long'])
        axs.plot(x3, y3, label='Flight Traj. M3', linestyle = '-', alpha = 0.8, color = self.c_[2])

        axs.scatter(x0[0], y0[0], label='Init. Pos. F16', color = self.c_[0], marker='.')
        axs.scatter(x1[0], y1[0], label='Init. Pos. M1', color = self.c_[4], marker='.')
        axs.scatter(x2[0], y2[0], label='Init. Pos. M2', color = self.c_[1], marker='.')
        axs.scatter(x3[0], y3[0], label='Init. Pos. M3', color = self.c_[2], marker='.')

        # scatter initial pos
        axs.set_ylabel('North')
        axs.set_xlabel('East')
        axs.grid()
        axs.axis('equal')
        #axs.set_xlim([-110, 110])
        #axs.set_ylim([-110, 110])
        x_ticks = np.linspace(-100, 100, 9)
        axs.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
        y_ticks = np.linspace(-100, 100, 9)
        axs.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))

        axs.legend()
        fig.savefig('jsb_gym/figures/testnn_path.pdf', bbox_inches="tight")

    def show_path3vs4(self, logs_):
        import matplotlib.ticker as ticker
        
        fig, axs = plt.subplots(1,1)
        from matplotlib.patches import Rectangle
        from matplotlib.cm import ScalarMappable
        # path
        logs = logs_[0]
        x0, y0  = self.pltk.geo2km(lat= logs['f16']['lat'], long= logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16-0', color = self.c_[0], linestyle = '-')
        axs.scatter(x0[0], y0[0], color = self.c_[0], marker='.')

        x1, y1  = self.pltk.geo2km(lat= logs['aim1']['lat'], long= logs['aim1']['long'])
        axs.plot(x1, y1, label='Flight Traj. M1', linestyle = '-', alpha = 0.8 , color = self.c_[4])

        logs = logs_[1]
        x0, y0  = self.pltk.geo2km(lat= logs['f16']['lat'], long= logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16-1',color = self.c_[0], linestyle = 'dashdot')
        axs.scatter(x0[0], y0[0], color = self.c_[0], marker='.')


        logs = logs_[2]
        x0, y0  = self.pltk.geo2km(lat= logs['f16']['lat'], long= logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16-2',color = self.c_[0], linestyle = 'dashed')
        axs.scatter(x0[0], y0[0], color = self.c_[0], marker='.')

        x2, y2  = self.pltk.geo2km(lat= logs['aim2']['lat'], long= logs['aim2']['long'])
        axs.plot(x2, y2, label='Flight Traj. M2', linestyle = '-', alpha = 0.8, color = self.c_[1])

        logs = logs_[3]
        x0, y0  = self.pltk.geo2km(lat= logs['f16']['lat'], long= logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16-3',color = self.c_[0], linestyle = 'dotted')

        x3, y3  = self.pltk.geo2km(lat= logs['aim3']['lat'], long= logs['aim3']['long'])
        axs.plot(x3, y3, label='Flight Traj. M3', linestyle = '-', alpha = 0.8, color = self.c_[2])


        plt.text(-10, -10, '(a)')
        plt.text( 10,  10, '(b)')

        plt.text(70, -140, '(c)')
        plt.text(90, -120, '(d)')

        axs.scatter(x0[0], y0[0], label='Init. Pos. F16-0..3', color = self.c_[0], marker='.')
        axs.scatter(x1[0], y1[0], label='Init. Pos. M1', color = self.c_[4], marker='.')
        axs.scatter(x2[0], y2[0], label='Init. Pos. M2', color = self.c_[1], marker='.')
        axs.scatter(x3[0], y3[0], label='Init. Pos. M3', color = self.c_[2], marker='.')

        # scatter initial pos
        axs.set_ylabel('North')
        axs.set_xlabel('East')
        axs.grid()
        axs.axis('equal')
        #axs.set_xlim([-110, 110])
        #axs.set_ylim([-110, 110])
        x_ticks = np.linspace(-100, 100, 9)
        axs.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
        y_ticks = np.linspace(-100, 100, 9)
        axs.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))

        axs.legend()
        fig.savefig('jsb_gym/figures/testnn_path3vs4.pdf', bbox_inches="tight")


    def show_path_4(self, logs):
        import matplotlib.ticker as ticker

        fig, axs = plt.subplots(1,1)
        from matplotlib.patches import Rectangle
        from matplotlib.cm import ScalarMappable
        # path
        x0, y0  = self.pltk.geo2km(lat= logs.logs['f16']['lat'], long= logs.logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16 ', color = self.c_[0])
        
        x1, y1  = self.pltk.geo2km(lat= logs.logs['aim1']['lat'], long= logs.logs['aim1']['long'])
        axs.plot(x1, y1, label='Flight Traj. M1', linestyle = '-', alpha = 0.8 , color = self.c_[4])

        x2, y2  = self.pltk.geo2km(lat= logs.logs['aim2']['lat'], long= logs.logs['aim2']['long'])
        axs.plot(x2, y2, label='Flight Traj. M2', linestyle = '-', alpha = 0.8, color = self.c_[1])

        x3, y3  = self.pltk.geo2km(lat= logs.logs['aim3']['lat'], long= logs.logs['aim3']['long'])
        axs.plot(x3, y3, label='Flight Traj. M3', linestyle = '-', alpha = 0.8, color = self.c_[2])

        x4, y4  = self.pltk.geo2km(lat= logs.logs['aim4']['lat'], long= logs.logs['aim4']['long'])
        axs.plot(x4, y4, label='Flight Traj. M4', linestyle = '-', alpha = 0.8, color = self.c_[3])

        axs.scatter(x0[0], y0[0], label='Init. Pos. F16', color = self.c_[0], marker='.')
        axs.scatter(x1[0], y1[0], label='Init. Pos. M1', color = self.c_[4], marker='.')
        axs.scatter(x2[0], y2[0], label='Init. Pos. M2', color = self.c_[1], marker='.')
        axs.scatter(x3[0], y3[0], label='Init. Pos. M3', color = self.c_[2], marker='.')
        axs.scatter(x4[0], y4[0], label='Init. Pos. M4', color = self.c_[3], marker='.')

        # scatter initial pos
        axs.set_ylabel('North')
        axs.set_xlabel('East')
        axs.grid()
        axs.axis('equal')
        #axs.set_xlim([-110, 110])
        #axs.set_ylim([-110, 110])
        #x_ticks = np.linspace(-100, 100, 9)
        #axs.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
        #y_ticks = np.linspace(-100, 100, 9)
        #axs.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))

        axs.legend()
        fig.savefig('jsb_gym/figures/testnn_path.pdf', bbox_inches="tight")


    def show_path_6(self, logs):
        import matplotlib.ticker as ticker

        fig, axs = plt.subplots(1,1)
        from matplotlib.patches import Rectangle
        from matplotlib.cm import ScalarMappable
        # path
        x0, y0  = self.pltk.geo2km(lat= logs.logs['f16']['lat'], long= logs.logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16 ', color = self.c_[0])
        
        x1, y1  = self.pltk.geo2km(lat= logs.logs['aim1']['lat'], long= logs.logs['aim1']['long'])
        axs.plot(x1, y1, label='Flight Traj. M1', linestyle = '-', alpha = 0.8 , color = self.c_[4])

        x2, y2  = self.pltk.geo2km(lat= logs.logs['aim2']['lat'], long= logs.logs['aim2']['long'])
        axs.plot(x2, y2, label='Flight Traj. M2', linestyle = '-', alpha = 0.8, color = self.c_[1])

        x3, y3  = self.pltk.geo2km(lat= logs.logs['aim3']['lat'], long= logs.logs['aim3']['long'])
        axs.plot(x3, y3, label='Flight Traj. M3', linestyle = '-', alpha = 0.8, color = self.c_[2])

        x4, y4  = self.pltk.geo2km(lat= logs.logs['aim4']['lat'], long= logs.logs['aim4']['long'])
        axs.plot(x4, y4, label='Flight Traj. M4', linestyle = '-', alpha = 0.8, color = self.c_[3])


        x5, y5  = self.pltk.geo2km(lat= logs.logs['aim5']['lat'], long= logs.logs['aim5']['long'])
        axs.plot(x5, y5, label='Flight Traj. M5', linestyle = '-', alpha = 0.8, color = self.c_[5])

        x6, y6  = self.pltk.geo2km(lat= logs.logs['aim6']['lat'], long= logs.logs['aim6']['long'])
        axs.plot(x6, y6, label='Flight Traj. M6', linestyle = '-', alpha = 0.8, color = self.c_[6])

        axs.scatter(x0[0], y0[0], label='Init. Pos. F16', color = self.c_[0], marker='.')
        axs.scatter(x1[0], y1[0], label='Init. Pos. M1', color = self.c_[4], marker='.')
        axs.scatter(x2[0], y2[0], label='Init. Pos. M2', color = self.c_[1], marker='.')
        axs.scatter(x3[0], y3[0], label='Init. Pos. M3', color = self.c_[2], marker='.')
        axs.scatter(x4[0], y4[0], label='Init. Pos. M4', color = self.c_[3], marker='.')
        print(len(self.c_))
        axs.scatter(x5[0], y5[0], label='Init. Pos. M5', color = self.c_[5], marker='.')
        axs.scatter(x6[0], y6[0], label='Init. Pos. M6', color = self.c_[6], marker='.')


        # Create a sub-plot in the top-right corner
        ax_sub = fig.add_axes([0.2, 0.4, 0.2, 0.2])

        # Plot the sub-plot data
        ax_sub.plot(x0, y0, color = self.c_[0])
        ax_sub.plot(x1, y1, alpha = 0.2, color = self.c_[4])
        ax_sub.plot(x2, y2, alpha = 0.2, color = self.c_[1])
        ax_sub.plot(x3, y3, alpha = 0.2, color = self.c_[2])
        ax_sub.plot(x4, y4, alpha = 0.2, color = self.c_[3])
        ax_sub.plot(x5, y5, alpha = 0.2, color = self.c_[5])
        ax_sub.plot(x6, y6, alpha = 0.2, color = self.c_[6])
        ax_sub.scatter(x0[0], y0[0], color = self.c_[0], marker='.')

        ax_sub.grid()
        ax_sub.axis('equal')
        ax_sub.set_xlim(-40, 40)
        ax_sub.set_ylim(-20, 60)

        # scatter initial pos
        axs.set_ylabel('North')
        axs.set_xlabel('East')
        axs.grid()
        axs.axis('equal')
        #axs.set_xlim([-110, 110])
        #axs.set_ylim([-110, 110])
        #x_ticks = np.linspace(-100, 100, 9)
        #axs.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
        #y_ticks = np.linspace(-100, 100, 9)
        #axs.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))

        axs.legend()
        fig.savefig('jsb_gym/figures/testnn_path.pdf', bbox_inches="tight")


    def show_3D(self, logs):
        fig = plt.figure() 
        ax = plt.axes(projection ='3d')
        
        x, y  = self.pltk.geo2km(lat= logs.logs['f16']['lat'], long= logs.logs['f16']['long'])
        z = [i*1e-3 for i in logs.logs['f16']['alt']]
        poly3dCollection = self.get_poly3dCollection(x, y, z, c = 'cornflowerblue' )
        ax.add_collection3d(poly3dCollection)
        
        x, y  = self.pltk.geo2km(lat= logs.logs['aim1']['lat'], long= logs.logs['aim1']['long'])
        z = [i*1e-3 for i in logs.logs['aim1']['alt']]
        poly3dCollection = self.get_poly3dCollection(x, y, z, c = 'crimson')
        ax.add_collection3d(poly3dCollection)

        x, y  = self.pltk.geo2km(lat= logs.logs['aim2']['lat'], long= logs.logs['aim2']['long'])
        z = [i*1e-3 for i in logs.logs['aim2']['alt']]
        poly3dCollection = self.get_poly3dCollection(x, y, z, c = 'crimson')
        ax.add_collection3d(poly3dCollection)

        x, y  = self.pltk.geo2km(lat= logs.logs['aim3']['lat'], long= logs.logs['aim3']['long'])
        z = [i*1e-3 for i in logs.logs['aim3']['alt']]
        poly3dCollection = self.get_poly3dCollection(x, y, z, c = 'crimson')
        ax.add_collection3d(poly3dCollection)


        mid_x = 30
        max_range = 30
        mid_y = 10
        mid_z = 8
        #ax.set_xlim(mid_x - max_range, mid_x + max_range)
        
        #ax.set_ylim(mid_y - max_range, mid_y + max_range)
        #ax.set_ylim(-10, 10)

        ax.view_init( 50 , 40)
        #ax.set_zlim(0, 20)

        #ax.text(f16_lat[0] + 10 , f16_long[0] -5, f16_alt[0] + 5, "(" + str(round(f16_lat[0],2)) + ', ' + str(round(f16_long[0],2)) + ', ' + str(round(f16_alt[0],2)) + ")" , color='black', size=18)
        #ax.text(aim_lat[0] - 5, aim_long[0] - 10, aim_alt[0] + 3, "(" + str(round(aim_lat[0],2)) + ', ' + str(round(-aim_long[0],2)) + ', ' + str(round(aim_alt[0],2)) + ")" , color='black', size=16)

        #ax.add_artist(self.add_f16_img(120, 80, 0.03, 0.0, fig='jsb_gym/figures/f16_side_tp.png'))
        #ax.add_artist(self.add_f16_img(-220, -50, 0.04, 0.0, fig= 'jsb_gym/figures/f16_side.png'))

        #tick_size = 16
        #ax.set_xlabel('   x axis [km]', fontsize = tick_size)
        #ax.set_ylabel('  y axis [km]', fontsize = tick_size)
        #ax.set_zlabel('  z axis [km]', fontsize = tick_size)
        
        #tick_size = 16
        #ax.tick_params(axis='x', labelsize=tick_size)
        #ax.tick_params(axis='y', labelsize=tick_size)
        #ax.tick_params(axis='z', labelsize=tick_size)
        #plt.subplots_adjust(top=0.6,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)        
        #ax.legend(loc='upper center', bbox_to_anchor= (0.8, 0.70),shadow=True, ncol=1,fontsize=18)
        #plt.show()
        #fig.savefig('jsb_gym/figures/line_' + self.args['plot'] + '.pdf', bbox_inches="tight")
        
        plt.show()

    def get_poly3dCollection(self, f16_lat,f16_long,f16_alt, c):
        h = 0.0
        spars = 10
        f16_lat = f16_lat[::spars]
        f16_long = f16_long[::spars]
        f16_alt = f16_alt[::spars]
        # Code to convert data in 3D polygons
        v = []
        for k in range(0, len(f16_lat) - 1):
            x = [f16_lat[k], f16_lat[k+1], f16_lat[k+1], f16_lat[k]]
            y = [f16_long[k], f16_long[k+1], f16_long[k+1], f16_long[k]]
            z = [f16_alt[k], f16_alt[k+1],       h,     h]
            #list is necessary in python 3/remove for python 2
            v.append(list(zip(x, y, z))) 
        poly3dCollection = Poly3DCollection(v, color = c, alpha=0.4, linewidths=0.0)
        return poly3dCollection

    def get_est_data(self, logs, direction):
        x = []
        for i in logs:
            x.append(i[direction])
        return x 

    def show_est(self, logs):
        
        fig, axs = plt.subplots(1,1)
        keys = [key for key in logs.est['f16'][0]]

        for key in keys:            
            x = self.get_est_data(logs.est['f16'], key)
            x = [(i +1)*10 for i in x]
            axs.plot( x, label=self.heading_names[key])
        axs.grid()
        axs.legend()
        fig.savefig('jsb_gym/figures/testnn_est.pdf', bbox_inches="tight")

    def show_est_fill(self, logs):
        c_ = ['blue', 'green', 'red', 'magenta', 'orange', 'brown', 'teal', 'gray']
        fig, axs = plt.subplots(1,1)
        keys = [key for key in logs.est['f16'][0]]
        
        print(keys)
        
        for idx, key in enumerate(keys):            
            x = self.get_est_data(logs.est['f16'], key)
            xx = np.linspace(0, max(logs.logs['f16']['sim_time']), len(x))
            x_min = self.get_est_data(logs.est_min['f16'], key)
            x_max = self.get_est_data(logs.est_max['f16'], key)
            x = [(i +1)*10 for i in x]
            x_min = [(i +1)*10 for i in x_min]
            x_max = [(i +1)*10 for i in x_max]
            axs.plot( xx, x, label=self.heading_names[key], color = c_[idx])
            axs.fill_between(list(xx), x_min, x_max, color = c_[idx], alpha=0.35)

        axs.grid()
        axs.legend()
        axs.set_xlabel('Time [sec]')
        axs.set_ylabel('Estimated MD [km]')
        axs.set_xlim(0,max(logs.logs['f16']['sim_time']))
        axs.set_ylim(0, 3)
        fig.savefig('jsb_gym/figures/testnn_est_fill.pdf', bbox_inches="tight")

    def show_est_list(self, logs):
        
        fig, axs = plt.subplots(1,1)
        keys = [key for key in logs.est['f16'][0]]

        for key in keys:            
            x = self.get_est_data(logs.est['f16'], key)
            x = [(i +1)*10 for i in x]
            axs.plot( x, label=self.heading_names[key])
        axs.grid()
        axs.legend()
        fig.savefig('jsb_gym/figures/testnn_est.pdf', bbox_inches="tight")




class testNN_plots():
    def __init__(self):
        from jsb_gym.utils.utils import Geo
        self.pltk = Geo()
        self.heading_names = {'sm1': 'N', 'sm075': 'NE', 'sm050': 'E', 'sm025': 'SE', 's00': 'S', 's025': 'SW', 's05':'W', 's075': 'NW'}
        self.c_ = ['blue', 'green', 'red', 'magenta', 'orange', 'brown', 'teal', 'gray']

    def create_canvas(self):
        #self.fig, self.axs = plt.subplots(1,1)
        self.fig = plt.gcf()
        # Adding Circle in Pie chart
                
        

    def add_to_canvas(self, act_dict, name, side_box = False):
        
        
        #keys = [key for key in logs.est['f16'][0]]

        est = {}
        for key in act_dict:
            #print(key)            
            #x = self.get_est_data(logs.est['f16'], key)
            est[self.heading_names[key]] = act_dict[key]
        
        Salary = [1]*8
  
        val = [0]*8
        #val = [(i +1)*10 for i in val] 
        val = [(est[key] +1)*20 for key in est] 
        
        # Setting labels for items in Chart
        Employee = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        Salary = [1]*8
        
        explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
        
        # Pie Chart
        cmap = plt.cm.RdYlGn
        norm = matplotlib.colors.Normalize(vmin= 0 , vmax= 20)
        colors = cmap(norm(val))
        
        plt.pie(Salary, colors=colors, labels=Employee,
                explode=explode, startangle=112, counterclock=False)
        # draw circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        self.fig.gca().add_artist(centre_circle)
        # Adding Title of chart


        #plt.title('Time ' + str(round(logs.logs['f16']['sim_time'][t])) + ' [sec]')
        #plt.grid()
        #fig.gca().add_artist(self.add_f16_img(0, 0, 0.15, 0.0, fig='jsb_gym/figures/Afterburner_DS/f16_up.png'))
        #cbaxes = fig.add_axes([0.85, 0.1, 0.05, 0.7])
        
        #if side_box:
        #    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ticks= list(np.linspace( 0, 20, 11)), orientation="vertical", cax=cbaxes)
    
        self.fig.savefig('jsb_gym/logs/SSA/'+ name +'.pdf', bbox_inches="tight")
        





    def show(self, logs):
        fig, axs = plt.subplots(2,1)
        axs[0].plot( logs.logs['f16']['sim_time'], logs.logs['f16']['alt'], label='F16')
        axs[0].plot(logs.logs['aim1']['sim_time'], logs.logs['aim1']['alt'], label='M1')
        axs[0].plot(logs.logs['aim2']['sim_time'], logs.logs['aim2']['alt'], label='M2')
        axs[0].plot(logs.logs['aim3']['sim_time'], logs.logs['aim3']['alt'], label='M3')
        axs[0].set_ylabel('Altitude [m]')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot( logs.logs['f16']['sim_time'], logs.logs['f16']['vel'], label='F16')
        axs[1].plot(logs.logs['aim1']['sim_time'], logs.logs['aim1']['vel'], label='M1')
        axs[1].plot(logs.logs['aim2']['sim_time'], logs.logs['aim2']['vel'], label='M2')
        axs[1].plot(logs.logs['aim3']['sim_time'], logs.logs['aim3']['vel'], label='M3')
        axs[1].set_ylabel('Mach')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_xlabel('Time [sec]')

        fig.savefig('jsb_gym/figures/testnn.pdf', bbox_inches="tight")

    def show4(self, logs):
        fig, axs = plt.subplots(2,1)
        axs[0].plot( logs.logs['f16']['sim_time'], logs.logs['f16']['alt'], label='F16')
        axs[0].plot(logs.logs['aim1']['sim_time'], logs.logs['aim1']['alt'], label='M1')
        axs[0].plot(logs.logs['aim2']['sim_time'], logs.logs['aim2']['alt'], label='M2')
        axs[0].plot(logs.logs['aim3']['sim_time'], logs.logs['aim3']['alt'], label='M3')
        axs[0].plot(logs.logs['aim4']['sim_time'], logs.logs['aim4']['alt'], label='M4')
        axs[0].set_ylabel('Altitude [m]')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot( logs.logs['f16']['sim_time'], logs.logs['f16']['vel'], label='F16')
        axs[1].plot(logs.logs['aim1']['sim_time'], logs.logs['aim1']['vel'], label='M1')
        axs[1].plot(logs.logs['aim2']['sim_time'], logs.logs['aim2']['vel'], label='M2')
        axs[1].plot(logs.logs['aim3']['sim_time'], logs.logs['aim3']['vel'], label='M3')
        axs[1].plot(logs.logs['aim4']['sim_time'], logs.logs['aim4']['vel'], label='M4')
        axs[1].set_ylabel('Mach')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_xlabel('Time [sec]')

        fig.savefig('jsb_gym/figures/testnn.pdf', bbox_inches="tight")


    def show6(self, logs):
        fig, axs = plt.subplots(2,1)
        axs[0].plot( logs.logs['f16']['sim_time'], logs.logs['f16']['alt'], label='F16')
        axs[0].plot(logs.logs['aim1']['sim_time'], logs.logs['aim1']['alt'], label='M1')
        axs[0].plot(logs.logs['aim2']['sim_time'], logs.logs['aim2']['alt'], label='M2')
        axs[0].plot(logs.logs['aim3']['sim_time'], logs.logs['aim3']['alt'], label='M3')
        axs[0].plot(logs.logs['aim4']['sim_time'], logs.logs['aim4']['alt'], label='M4')
        axs[0].plot(logs.logs['aim5']['sim_time'], logs.logs['aim5']['alt'], label='M5')
        axs[0].plot(logs.logs['aim6']['sim_time'], logs.logs['aim6']['alt'], label='M6')
        axs[0].set_ylabel('Altitude [m]')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot( logs.logs['f16']['sim_time'], logs.logs['f16']['vel'], label='F16')
        axs[1].plot(logs.logs['aim1']['sim_time'], logs.logs['aim1']['vel'], label='M1')
        axs[1].plot(logs.logs['aim2']['sim_time'], logs.logs['aim2']['vel'], label='M2')
        axs[1].plot(logs.logs['aim3']['sim_time'], logs.logs['aim3']['vel'], label='M3')
        axs[1].plot(logs.logs['aim4']['sim_time'], logs.logs['aim4']['vel'], label='M4')
        axs[1].plot(logs.logs['aim5']['sim_time'], logs.logs['aim5']['vel'], label='M5')
        axs[1].plot(logs.logs['aim6']['sim_time'], logs.logs['aim6']['vel'], label='M6')
        axs[1].set_ylabel('Mach')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_xlabel('Time [sec]')

        fig.savefig('jsb_gym/figures/testnn.pdf', bbox_inches="tight")


    def show_circle(self, logs, t, side_box = True):

        #import matplotlib.pyplot as plt
        #import numpy as np
        from matplotlib.cm import ScalarMappable
        import matplotlib.colors
        
        #xx = logs.logs['f16']['sim_time'][t]
        print(logs.logs['f16']['psi'][t])
        keys = [key for key in logs.est['f16'][0]]

        est = {}
        for key in keys:            
            x = self.get_est_data(logs.est['f16'], key)
            est[self.heading_names[key]] = x[t]
        
        Salary = [1]*8
  
        val = [0]*8
        #val = [(i +1)*10 for i in val] 
        val = [(est[key] +1)*20 for key in est] 
        
        # Setting labels for items in Chart
        Employee = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        Salary = [1]*8
        
        explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
        
        # Pie Chart
        cmap = plt.cm.RdYlGn
        norm = matplotlib.colors.Normalize(vmin= 0 , vmax= 20)
        colors = cmap(norm(val))
        plt.pie(Salary, colors=colors, labels=Employee,
                explode=explode, startangle=112, counterclock=False)
        # draw circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        # Adding Circle in Pie chart
        fig.gca().add_artist(centre_circle)        
        # Adding Title of chart
        plt.title('Time ' + str(round(logs.logs['f16']['sim_time'][t])) + ' [sec]')
        plt.grid()
        #fig.gca().add_artist(self.add_f16_img(0, 0, 0.15, 0.0, fig='jsb_gym/figures/Afterburner_DS/f16_up.png'))
        cbaxes = fig.add_axes([0.85, 0.1, 0.05, 0.7])
        if side_box:
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ticks= list(np.linspace( 0, 20, 11)), orientation="vertical", cax=cbaxes)
        fig.savefig('jsb_gym/figures/NESW'+str(t)+'.pdf', bbox_inches="tight")
        #plt.show()
        
    def add_f16_img(self,x, y, zoom, rot , fig):
        img = mpimg.imread(fig)
        img = scipy.ndimage.rotate(img, rot)
        imageBox = OffsetImage(img, zoom = zoom)
        xy = [0,0]
        ab = AnnotationBbox(imageBox, xy, xybox=(x,y), boxcoords='offset points', frameon=False)
        return ab

    def show_path(self, logs):
        import matplotlib.ticker as ticker

        fig, axs = plt.subplots(1,1)
        from matplotlib.patches import Rectangle
        from matplotlib.cm import ScalarMappable
        # path
        x0, y0  = self.pltk.geo2km(lat= logs.logs['f16']['lat'], long= logs.logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16 ', color = self.c_[0])
        
        x1, y1  = self.pltk.geo2km(lat= logs.logs['aim1']['lat'], long= logs.logs['aim1']['long'])
        axs.plot(x1, y1, label='Flight Traj. M1', linestyle = '-', alpha = 0.8 , color = self.c_[4])

        x2, y2  = self.pltk.geo2km(lat= logs.logs['aim2']['lat'], long= logs.logs['aim2']['long'])
        axs.plot(x2, y2, label='Flight Traj. M2', linestyle = '-', alpha = 0.8, color = self.c_[1])

        x3, y3  = self.pltk.geo2km(lat= logs.logs['aim3']['lat'], long= logs.logs['aim3']['long'])
        axs.plot(x3, y3, label='Flight Traj. M3', linestyle = '-', alpha = 0.8, color = self.c_[2])

        axs.scatter(x0[0], y0[0], label='Init. Pos. F16', color = self.c_[0], marker='.')
        axs.scatter(x1[0], y1[0], label='Init. Pos. M1', color = self.c_[4], marker='.')
        axs.scatter(x2[0], y2[0], label='Init. Pos. M2', color = self.c_[1], marker='.')
        axs.scatter(x3[0], y3[0], label='Init. Pos. M3', color = self.c_[2], marker='.')

        # scatter initial pos
        axs.set_ylabel('North')
        axs.set_xlabel('East')
        axs.grid()
        axs.axis('equal')
        #axs.set_xlim([-110, 110])
        #axs.set_ylim([-110, 110])
        x_ticks = np.linspace(-100, 100, 9)
        axs.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
        y_ticks = np.linspace(-100, 100, 9)
        axs.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))

        axs.legend()
        fig.savefig('jsb_gym/figures/testnn_path.pdf', bbox_inches="tight")

    def show_path3vs4(self, logs_):
        import matplotlib.ticker as ticker
        
        fig, axs = plt.subplots(1,1)
        from matplotlib.patches import Rectangle
        from matplotlib.cm import ScalarMappable
        # path
        logs = logs_[0]
        x0, y0  = self.pltk.geo2km(lat= logs['f16']['lat'], long= logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16-0', color = self.c_[0], linestyle = '-')
        axs.scatter(x0[0], y0[0], color = self.c_[0], marker='.')

        x1, y1  = self.pltk.geo2km(lat= logs['aim1']['lat'], long= logs['aim1']['long'])
        axs.plot(x1, y1, label='Flight Traj. M1', linestyle = '-', alpha = 0.8 , color = self.c_[4])

        logs = logs_[1]
        x0, y0  = self.pltk.geo2km(lat= logs['f16']['lat'], long= logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16-1',color = self.c_[0], linestyle = 'dashdot')
        axs.scatter(x0[0], y0[0], color = self.c_[0], marker='.')


        logs = logs_[2]
        x0, y0  = self.pltk.geo2km(lat= logs['f16']['lat'], long= logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16-2',color = self.c_[0], linestyle = 'dashed')
        axs.scatter(x0[0], y0[0], color = self.c_[0], marker='.')

        x2, y2  = self.pltk.geo2km(lat= logs['aim2']['lat'], long= logs['aim2']['long'])
        axs.plot(x2, y2, label='Flight Traj. M2', linestyle = '-', alpha = 0.8, color = self.c_[1])

        logs = logs_[3]
        x0, y0  = self.pltk.geo2km(lat= logs['f16']['lat'], long= logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16-3',color = self.c_[0], linestyle = 'dotted')

        x3, y3  = self.pltk.geo2km(lat= logs['aim3']['lat'], long= logs['aim3']['long'])
        axs.plot(x3, y3, label='Flight Traj. M3', linestyle = '-', alpha = 0.8, color = self.c_[2])


        plt.text(-10, -10, '(a)')
        plt.text( 10,  10, '(b)')

        plt.text(70, -140, '(c)')
        plt.text(90, -120, '(d)')

        axs.scatter(x0[0], y0[0], label='Init. Pos. F16-0..3', color = self.c_[0], marker='.')
        axs.scatter(x1[0], y1[0], label='Init. Pos. M1', color = self.c_[4], marker='.')
        axs.scatter(x2[0], y2[0], label='Init. Pos. M2', color = self.c_[1], marker='.')
        axs.scatter(x3[0], y3[0], label='Init. Pos. M3', color = self.c_[2], marker='.')

        # scatter initial pos
        axs.set_ylabel('North')
        axs.set_xlabel('East')
        axs.grid()
        axs.axis('equal')
        #axs.set_xlim([-110, 110])
        #axs.set_ylim([-110, 110])
        x_ticks = np.linspace(-100, 100, 9)
        axs.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
        y_ticks = np.linspace(-100, 100, 9)
        axs.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))

        axs.legend()
        fig.savefig('jsb_gym/figures/testnn_path3vs4.pdf', bbox_inches="tight")


    def show_path_4(self, logs):
        import matplotlib.ticker as ticker

        fig, axs = plt.subplots(1,1)
        from matplotlib.patches import Rectangle
        from matplotlib.cm import ScalarMappable
        # path
        x0, y0  = self.pltk.geo2km(lat= logs.logs['f16']['lat'], long= logs.logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16 ', color = self.c_[0])
        
        x1, y1  = self.pltk.geo2km(lat= logs.logs['aim1']['lat'], long= logs.logs['aim1']['long'])
        axs.plot(x1, y1, label='Flight Traj. M1', linestyle = '-', alpha = 0.8 , color = self.c_[4])

        x2, y2  = self.pltk.geo2km(lat= logs.logs['aim2']['lat'], long= logs.logs['aim2']['long'])
        axs.plot(x2, y2, label='Flight Traj. M2', linestyle = '-', alpha = 0.8, color = self.c_[1])

        x3, y3  = self.pltk.geo2km(lat= logs.logs['aim3']['lat'], long= logs.logs['aim3']['long'])
        axs.plot(x3, y3, label='Flight Traj. M3', linestyle = '-', alpha = 0.8, color = self.c_[2])

        x4, y4  = self.pltk.geo2km(lat= logs.logs['aim4']['lat'], long= logs.logs['aim4']['long'])
        axs.plot(x4, y4, label='Flight Traj. M4', linestyle = '-', alpha = 0.8, color = self.c_[3])

        axs.scatter(x0[0], y0[0], label='Init. Pos. F16', color = self.c_[0], marker='.')
        axs.scatter(x1[0], y1[0], label='Init. Pos. M1', color = self.c_[4], marker='.')
        axs.scatter(x2[0], y2[0], label='Init. Pos. M2', color = self.c_[1], marker='.')
        axs.scatter(x3[0], y3[0], label='Init. Pos. M3', color = self.c_[2], marker='.')
        axs.scatter(x4[0], y4[0], label='Init. Pos. M4', color = self.c_[3], marker='.')

        # scatter initial pos
        axs.set_ylabel('North')
        axs.set_xlabel('East')
        axs.grid()
        axs.axis('equal')
        #axs.set_xlim([-110, 110])
        #axs.set_ylim([-110, 110])
        #x_ticks = np.linspace(-100, 100, 9)
        #axs.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
        #y_ticks = np.linspace(-100, 100, 9)
        #axs.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))

        axs.legend()
        fig.savefig('jsb_gym/figures/testnn_path.pdf', bbox_inches="tight")


    def show_path_6(self, logs):
        import matplotlib.ticker as ticker

        fig, axs = plt.subplots(1,1)
        from matplotlib.patches import Rectangle
        from matplotlib.cm import ScalarMappable
        # path
        x0, y0  = self.pltk.geo2km(lat= logs.logs['f16']['lat'], long= logs.logs['f16']['long'])
        axs.plot(x0, y0, label='Flight Traj. F16 ', color = self.c_[0])
        
        x1, y1  = self.pltk.geo2km(lat= logs.logs['aim1']['lat'], long= logs.logs['aim1']['long'])
        axs.plot(x1, y1, label='Flight Traj. M1', linestyle = '-', alpha = 0.8 , color = self.c_[4])

        x2, y2  = self.pltk.geo2km(lat= logs.logs['aim2']['lat'], long= logs.logs['aim2']['long'])
        axs.plot(x2, y2, label='Flight Traj. M2', linestyle = '-', alpha = 0.8, color = self.c_[1])

        x3, y3  = self.pltk.geo2km(lat= logs.logs['aim3']['lat'], long= logs.logs['aim3']['long'])
        axs.plot(x3, y3, label='Flight Traj. M3', linestyle = '-', alpha = 0.8, color = self.c_[2])

        x4, y4  = self.pltk.geo2km(lat= logs.logs['aim4']['lat'], long= logs.logs['aim4']['long'])
        axs.plot(x4, y4, label='Flight Traj. M4', linestyle = '-', alpha = 0.8, color = self.c_[3])


        x5, y5  = self.pltk.geo2km(lat= logs.logs['aim5']['lat'], long= logs.logs['aim5']['long'])
        axs.plot(x5, y5, label='Flight Traj. M5', linestyle = '-', alpha = 0.8, color = self.c_[5])

        x6, y6  = self.pltk.geo2km(lat= logs.logs['aim6']['lat'], long= logs.logs['aim6']['long'])
        axs.plot(x6, y6, label='Flight Traj. M6', linestyle = '-', alpha = 0.8, color = self.c_[6])

        axs.scatter(x0[0], y0[0], label='Init. Pos. F16', color = self.c_[0], marker='.')
        axs.scatter(x1[0], y1[0], label='Init. Pos. M1', color = self.c_[4], marker='.')
        axs.scatter(x2[0], y2[0], label='Init. Pos. M2', color = self.c_[1], marker='.')
        axs.scatter(x3[0], y3[0], label='Init. Pos. M3', color = self.c_[2], marker='.')
        axs.scatter(x4[0], y4[0], label='Init. Pos. M4', color = self.c_[3], marker='.')
        print(len(self.c_))
        axs.scatter(x5[0], y5[0], label='Init. Pos. M5', color = self.c_[5], marker='.')
        axs.scatter(x6[0], y6[0], label='Init. Pos. M6', color = self.c_[6], marker='.')


        # Create a sub-plot in the top-right corner
        ax_sub = fig.add_axes([0.2, 0.4, 0.2, 0.2])

        # Plot the sub-plot data
        ax_sub.plot(x0, y0, color = self.c_[0])
        ax_sub.plot(x1, y1, alpha = 0.2, color = self.c_[4])
        ax_sub.plot(x2, y2, alpha = 0.2, color = self.c_[1])
        ax_sub.plot(x3, y3, alpha = 0.2, color = self.c_[2])
        ax_sub.plot(x4, y4, alpha = 0.2, color = self.c_[3])
        ax_sub.plot(x5, y5, alpha = 0.2, color = self.c_[5])
        ax_sub.plot(x6, y6, alpha = 0.2, color = self.c_[6])
        ax_sub.scatter(x0[0], y0[0], color = self.c_[0], marker='.')

        ax_sub.grid()
        ax_sub.axis('equal')
        ax_sub.set_xlim(-40, 40)
        ax_sub.set_ylim(-20, 60)

        # scatter initial pos
        axs.set_ylabel('North')
        axs.set_xlabel('East')
        axs.grid()
        axs.axis('equal')
        #axs.set_xlim([-110, 110])
        #axs.set_ylim([-110, 110])
        #x_ticks = np.linspace(-100, 100, 9)
        #axs.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
        #y_ticks = np.linspace(-100, 100, 9)
        #axs.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))

        axs.legend()
        fig.savefig('jsb_gym/figures/testnn_path.pdf', bbox_inches="tight")


    def show_3D(self, logs):
        fig = plt.figure() 
        ax = plt.axes(projection ='3d')
        
        x, y  = self.pltk.geo2km(lat= logs.logs['f16']['lat'], long= logs.logs['f16']['long'])
        z = [i*1e-3 for i in logs.logs['f16']['alt']]
        poly3dCollection = self.get_poly3dCollection(x, y, z, c = 'cornflowerblue' )
        ax.add_collection3d(poly3dCollection)
        
        x, y  = self.pltk.geo2km(lat= logs.logs['aim1']['lat'], long= logs.logs['aim1']['long'])
        z = [i*1e-3 for i in logs.logs['aim1']['alt']]
        poly3dCollection = self.get_poly3dCollection(x, y, z, c = 'crimson')
        ax.add_collection3d(poly3dCollection)

        x, y  = self.pltk.geo2km(lat= logs.logs['aim2']['lat'], long= logs.logs['aim2']['long'])
        z = [i*1e-3 for i in logs.logs['aim2']['alt']]
        poly3dCollection = self.get_poly3dCollection(x, y, z, c = 'crimson')
        ax.add_collection3d(poly3dCollection)

        x, y  = self.pltk.geo2km(lat= logs.logs['aim3']['lat'], long= logs.logs['aim3']['long'])
        z = [i*1e-3 for i in logs.logs['aim3']['alt']]
        poly3dCollection = self.get_poly3dCollection(x, y, z, c = 'crimson')
        ax.add_collection3d(poly3dCollection)


        mid_x = 30
        max_range = 30
        mid_y = 10
        mid_z = 8
        #ax.set_xlim(mid_x - max_range, mid_x + max_range)
        
        #ax.set_ylim(mid_y - max_range, mid_y + max_range)
        #ax.set_ylim(-10, 10)

        ax.view_init( 50 , 40)
        #ax.set_zlim(0, 20)

        #ax.text(f16_lat[0] + 10 , f16_long[0] -5, f16_alt[0] + 5, "(" + str(round(f16_lat[0],2)) + ', ' + str(round(f16_long[0],2)) + ', ' + str(round(f16_alt[0],2)) + ")" , color='black', size=18)
        #ax.text(aim_lat[0] - 5, aim_long[0] - 10, aim_alt[0] + 3, "(" + str(round(aim_lat[0],2)) + ', ' + str(round(-aim_long[0],2)) + ', ' + str(round(aim_alt[0],2)) + ")" , color='black', size=16)

        #ax.add_artist(self.add_f16_img(120, 80, 0.03, 0.0, fig='jsb_gym/figures/f16_side_tp.png'))
        #ax.add_artist(self.add_f16_img(-220, -50, 0.04, 0.0, fig= 'jsb_gym/figures/f16_side.png'))

        #tick_size = 16
        #ax.set_xlabel('   x axis [km]', fontsize = tick_size)
        #ax.set_ylabel('  y axis [km]', fontsize = tick_size)
        #ax.set_zlabel('  z axis [km]', fontsize = tick_size)
        
        #tick_size = 16
        #ax.tick_params(axis='x', labelsize=tick_size)
        #ax.tick_params(axis='y', labelsize=tick_size)
        #ax.tick_params(axis='z', labelsize=tick_size)
        #plt.subplots_adjust(top=0.6,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)        
        #ax.legend(loc='upper center', bbox_to_anchor= (0.8, 0.70),shadow=True, ncol=1,fontsize=18)
        #plt.show()
        #fig.savefig('jsb_gym/figures/line_' + self.args['plot'] + '.pdf', bbox_inches="tight")
        
        plt.show()

    def get_poly3dCollection(self, f16_lat,f16_long,f16_alt, c):
        h = 0.0
        spars = 10
        f16_lat = f16_lat[::spars]
        f16_long = f16_long[::spars]
        f16_alt = f16_alt[::spars]
        # Code to convert data in 3D polygons
        v = []
        for k in range(0, len(f16_lat) - 1):
            x = [f16_lat[k], f16_lat[k+1], f16_lat[k+1], f16_lat[k]]
            y = [f16_long[k], f16_long[k+1], f16_long[k+1], f16_long[k]]
            z = [f16_alt[k], f16_alt[k+1],       h,     h]
            #list is necessary in python 3/remove for python 2
            v.append(list(zip(x, y, z))) 
        poly3dCollection = Poly3DCollection(v, color = c, alpha=0.4, linewidths=0.0)
        return poly3dCollection

    def get_est_data(self, logs, direction):
        x = []
        for i in logs:
            x.append(i[direction])
        return x 

    def show_est(self, logs):
        
        fig, axs = plt.subplots(1,1)
        keys = [key for key in logs.est['f16'][0]]

        for key in keys:            
            x = self.get_est_data(logs.est['f16'], key)
            x = [(i +1)*10 for i in x]
            axs.plot( x, label=self.heading_names[key])
        axs.grid()
        axs.legend()
        fig.savefig('jsb_gym/figures/testnn_est.pdf', bbox_inches="tight")

    def show_est_fill(self, logs):
        c_ = ['blue', 'green', 'red', 'magenta', 'orange', 'brown', 'teal', 'gray']
        fig, axs = plt.subplots(1,1)
        keys = [key for key in logs.est['f16'][0]]
        
        print(keys)
        
        for idx, key in enumerate(keys):            
            x = self.get_est_data(logs.est['f16'], key)
            xx = np.linspace(0, max(logs.logs['f16']['sim_time']), len(x))
            x_min = self.get_est_data(logs.est_min['f16'], key)
            x_max = self.get_est_data(logs.est_max['f16'], key)
            x = [(i +1)*10 for i in x]
            x_min = [(i +1)*10 for i in x_min]
            x_max = [(i +1)*10 for i in x_max]
            axs.plot( xx, x, label=self.heading_names[key], color = c_[idx])
            axs.fill_between(list(xx), x_min, x_max, color = c_[idx], alpha=0.35)

        axs.grid()
        axs.legend()
        axs.set_xlabel('Time [sec]')
        axs.set_ylabel('Estimated MD [km]')
        axs.set_xlim(0,max(logs.logs['f16']['sim_time']))
        axs.set_ylim(0, 3)
        fig.savefig('jsb_gym/figures/testnn_est_fill.pdf', bbox_inches="tight")

    def show_est_list(self, logs):
        
        fig, axs = plt.subplots(1,1)
        keys = [key for key in logs.est['f16'][0]]

        for key in keys:            
            x = self.get_est_data(logs.est['f16'], key)
            x = [(i +1)*10 for i in x]
            axs.plot( x, label=self.heading_names[key])
        axs.grid()
        axs.legend()
        fig.savefig('jsb_gym/figures/testnn_est.pdf', bbox_inches="tight")


