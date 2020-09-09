import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import os.path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Credit to Adam J. Finley

#########################################
Owens_data = './srep41548-s3.txt'
f = open(Owens_data,'r')
Owens_array = genfromtxt(Owens_data, unpack=False, skip_header=13)
f.close()
#########################################
Owens_Time = Owens_array[:,0]
Owens_Br_array = Owens_array[:,1:]*(1e-5)
Owens_Lat= np.arange(-87.5,92.5,5)
TIME, LAT = np.meshgrid(Owens_Time,Owens_Lat)
Owens_OpenFlux=[]
for i in range(0,len(Owens_array[:,0])):
	Owens_OpenFlux.append(2.*np.pi*np.sum(np.abs(Owens_Br_array[i,:]*(1.5e13)**2*np.cos(Owens_Lat*np.pi/180)*np.sin(.5*5*np.pi/180))))

Sunspot_Owens=np.interp(Owens_Time,Sunspot_Time,Sunspot_number)
#########################################
fig, (ax1,ax2,ax3,ax4, ax5) = plt.subplots(5,1,figsize=(15,10),sharex=True)
AX1a=ax1.pcolormesh(TIME,LAT,Owens_Br_array.T,cmap='seismic')
cbaxes = inset_axes(ax1, width="30%", height="10%", loc=1) 
plt.colorbar(AX1a,cax=cbaxes,orientation='horizontal', label=r'B$_{r}$ [G]')
ax1.set_ylabel('Latitude')


AX2a,=ax2.plot(Owens_Time,np.asarray(Owens_OpenFlux),c='k')
AX2b,=ax2.plot(ACE_Time,ACE_OpenFlux,c='grey', alpha=0.5,zorder=0)
ax2.set_ylabel('Open Flux [Mx]')
#########################################
legend_ax2 = ax2.legend([AX2a,AX2b],['Owens Data','ACE Data'],loc='upper left', shadow=False,borderpad=0.2,labelspacing=0.1)
frame_ax2 = legend_ax2.get_frame()
frame_ax2.set_facecolor('0.90')
# Set the fontsize
for label in legend_ax2.get_texts():
    label.set_fontsize(10.)
for label in legend_ax2.get_lines():
    label.set_linewidth(1.)  # the legend line width
#########################################


#########################################
Mdot_ACE_file = '/Users/afinley/SolarStellarCycles/ACEdata/ACE_Mdot.txt'
f = open(Mdot_ACE_file,'r')
ACE_Array = genfromtxt(Mdot_ACE_file, skip_header=2,unpack=False)
f.close()

ACE_Time=ACE_Array[:,0]
ACE_Mdot=ACE_Array[:,1]
#########################################
Owens_data = './srep41548-s2.txt'
f = open(Owens_data,'r')
Owens_array = genfromtxt(Owens_data, unpack=False, skip_header=13)
f.close()
#########################################
Owens_Time = Owens_array[:,0]
Owens_Wind_array = Owens_array[:,1:]
Owens_Lat= np.arange(-87.5,92.5,5)
TIME, LAT = np.meshgrid(Owens_Time,Owens_Lat)
AverageSpeed_Owens=[]
for i in range(0,len(Owens_array[:,0])):
	AverageSpeed_Owens.append(np.mean(Owens_Wind_array[i,:]*np.cos(Owens_Lat*np.pi/180)))

#########################################
# AX3a=ax3.pcolormesh(TIME,LAT,Owens_Wind_array.T,cmap='gnuplot',vmin=200,vmax=800)
# cbaxes = inset_axes(ax3, width="30%", height="10%", loc=1) 
# plt.colorbar(AX3a,cax=cbaxes,orientation='horizontal', label=r'V$_{RECON}$ [km/s]')
# ax3.set_ylabel('Latitude')

Owens_SlowFast=np.empty_like(Owens_Wind_array)
for i in range(0,len(Owens_array[:,0])):
	for j in range(0,len(Owens_Wind_array[i,:])):
		if Owens_Wind_array[i,j]>550:
			Owens_SlowFast[i,j]=1
		else:
			Owens_SlowFast[i,j]=0

def massFluxR2(type,vr,openF):
	if type==0:
		massFlux = (80e6*vr+7.6e10)*(openF/9e22)
	elif type==1:
		massFlux = (-200e6*vr+2.3e11)*(openF/9e22)
	else:
		massFlux = 0
	return massFlux

Owens_massFlux=np.empty_like(Owens_Wind_array)
for i in range(0,len(Owens_array[:,0])):
	for j in range(0,len(Owens_Wind_array[i,:])):
		Owens_massFlux[i,j]=massFluxR2(Owens_SlowFast[i,j],Owens_Wind_array[i,j],Owens_OpenFlux[i])

AX3a=ax3.pcolormesh(TIME,LAT,Owens_massFlux.T,cmap='jet')
cbaxes = inset_axes(ax3, width="30%", height="10%", loc=1) 
plt.colorbar(AX3a,cax=cbaxes,orientation='horizontal', label=r'Mass Flux [g/s]')
ax3.set_ylabel('Latitude')


Owens_mdot=[]
for i in range(0,len(Owens_array[:,0])):
	Owens_mdot.append(np.sum(4.0*np.pi*np.cos(Owens_Lat*np.pi/180)*np.sin(np.pi/180*5/2.)*Owens_massFlux[i,:]))

AX4a,=ax4.plot(Owens_Time,Owens_mdot,c='k')
AX4b,=ax4.plot(ACE_Time,ACE_Mdot,c='grey', alpha=0.5,zorder=0)
ax4.set_ylabel('Mass Loss Rate [g/s]')
#########################################
legend_ax4 = ax4.legend([AX4a,AX4b],['Owens Data','ACE Data'],loc='upper left', shadow=False,borderpad=0.2,labelspacing=0.1)
frame_ax4 = legend_ax4.get_frame()
frame_ax4.set_facecolor('0.90')
# Set the fontsize
for label in legend_ax4.get_texts():
    label.set_fontsize(10.)
for label in legend_ax4.get_lines():
    label.set_linewidth(1.)  # the legend line width
#########################################

Owens_torque=2.3e30*(np.asarray(Owens_mdot)/1.1e12)**0.26*(np.asarray(Owens_OpenFlux)/8e22)**1.48
AX5a,=ax5.plot(Owens_Time,Owens_torque/1e30,c='r')
ax5.set_ylabel(r'Solar Wind Torque [$\times10^{30}$erg]')
AX5c=ax5.axhline(y=np.mean(Owens_torque)/1e30,c='k')

#########################################
Torque_ACE_file = '/Users/afinley/SolarStellarCycles/ACEdata/ACE_Torque.txt'
f = open(Torque_ACE_file,'r')
ACE_Array = genfromtxt(Torque_ACE_file, skip_header=2,unpack=False)
f.close()

ACE_Time=ACE_Array[:,0]
ACE_Torque=ACE_Array[:,1]
#########################################
AX5d=ax5.axhline(y=np.mean(ACE_Torque)/1e30,c='grey',ls='--')
AX5b,=ax5.plot(ACE_Time,np.asarray(ACE_Torque)/1e30,c='grey', alpha=0.5,zorder=0)

AX5e=ax5.errorbar([1983], [3.16], yerr=[[0.65],[0.65]], fmt='o',c='red',lw=3,zorder=0)
AX5f=ax5.scatter([1983],[3.16],color='red',edgecolor='k',s=50)
AX5g=ax5.scatter([1999],[2.1],color='orange',edgecolor='k',s=50,zorder=5)

#########################################
legend_ax5 = ax5.legend([AX5a,AX5b,AX5c,AX5d,AX5f,AX5g],['Owens Data','ACE Data','Centenial Average','Paper I Average',"Pizzo et al. (1983)","Li (1999)"],loc='upper left', shadow=False,borderpad=0.2,labelspacing=0.1)
frame_ax5 = legend_ax5.get_frame()
frame_ax5.set_facecolor('0.90')
# Set the fontsize
for label in legend_ax5.get_texts():
    label.set_fontsize(10.)
for label in legend_ax5.get_lines():
    label.set_linewidth(1.)  # the legend line width
#########################################

ax1.set_xlim(np.min(Owens_Time),np.max(Owens_Time))
plt.rcParams.update({'font.size': 6})
plt.tight_layout()
plt.show()
