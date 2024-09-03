"""
Author: Gabriel Behrendt
Date: Fri Oct 20 15:03:16 2023
"""

import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

def obj_func(x,C,Q,P,R,yRef,numAgents,numStates,sepDesired):
    J = 0
    J += 0.5*(x.T @ Q @ x)
    J += 0.5*((C@x - yRef).T @ P @ (C@x - yRef))

    dist = np.zeros((numAgents-1,1))
    for i in range(numAgents-1):
        dist[i] = x[i*numStates + 4] - x[(i+1)*numStates + 4]
        #print(dist[i])

    J += 0.5*((dist - sepDesired).T @ R @ (dist - sepDesired))

    return J.item()


# np.random.seed('twister', 1337)
np.random.seed(2)

# code switches (1 == on, 0 == off)
delayCom = 1
delayUp = 1
delayMeas = 1
B = 50

# Problem Parameters
gamma = 9e-8 # step size
numFun = 20 # number of switching functions
iters = 500 # number of allowed iterations per function
ts = 5 # sampling time

numAgents = 8 # number of agents
numStates = 5 # number of states
numOutputs = 2 # number of outputs
m = numOutputs*numAgents # total number of network outputs
n = numStates*numAgents # total number of network states

# Generate C matrix
Ci = np.array([[-0.0133, -7.3259, -3.17, -1.1965, 1e-4],
               [0, 0, 0, 0, 1]])
C = np.kron(np.eye(numAgents),Ci)

CC = C[1::2].copy()
for i in range(CC.shape[0]-1):
    CC[i,:] = CC[i,:] - CC[i+1,:]
Cbar = CC[:-1,:]

# Generate Delay probabilities
prob = 0.5

if delayCom == 1:
    comProb = np.ones(numAgents)*prob
else:
    comProb = np.ones(numAgents)
if delayUp == 1:
    upProb = np.ones(numAgents)*prob
else:
    upProb = np.ones(numAgents)
if delayMeas == 1:
    measProb = np.ones(numAgents)*prob
else:
    measProb = np.ones(numAgents)

# Generate Cost Matrices
Q = 1e2*np.eye(n) # cost on states
# P = 1e4*np.eye(m) # cost on Vdot and altitude tracking
R = 1e6*np.eye(numAgents-1) # cost on altitude separation

p = np.empty((m,))
p[::2] = 1e3 # desired acceleration cost
p[1::2] = 5e4 # desired altitude cost
P = np.diag(p)

# Generate desired acceleration, altitude, and agent separation
tt = ts*np.arange(0,numFun)

h0 = 15000 # initial altitude
v0 = 500 # initial velocity


hDesired = h0 + 1500*np.sin(tt*np.pi/24) # desired altitude follows a sampled sine wave
sepDesired = np.ones((numAgents-1,1))*1500

# Define State Limits
Vmax = v0 + 56.2664 # +/- 0.05 mach = 56.2664 fps
Vmin = v0 - 56.2664
aoaMax = 1.5
aoaMin = -13
pitchMax = 25
pitchMin = -pitchMax
pitchRateMax = 60
pitchRateMin = -pitchRateMax
altMax = 40000
altMin = 1000
xMax = np.array([Vmax, aoaMax, pitchMax, pitchRateMax, altMax])
xMin = np.array([Vmin, aoaMin, pitchMin, pitchRateMin, altMin])



## Initialize Agent Values x = [V \alpha \dot{\phi} \phi h]^T, y = [\dot{V} h]^T
Vinit = v0 # Initial velocity for all agents of mach 0.9 or 1012.8 ft/s
hMax = 22000 # I just picked 1600 + 8 * 2000
hInit = [hMax]

for i in range(numAgents-1):
    hInit.append(hInit[i] - 2000)

xInit = np.zeros((n,1))
yInit = np.zeros((m,1))
for i in range(numAgents):
    xInit[i*numStates] = Vinit
    xInit[i*numStates + 4] = hInit[i]

    yInit[i*numOutputs + 1] = hInit[i]

xi = np.matlib.repmat(xInit,1,numAgents) + np.random.rand(n, numAgents) # stores agents' local state vector
yi = np.matlib.repmat(yInit,1,numAgents) + np.random.rand(m, numAgents) # stores agents' local output vector

lastCom = np.zeros(numAgents)
lastUp = np.zeros(numAgents)
lastMeas = np.zeros(numAgents)

# Define true states and outputs of the network
xTrue = np.zeros((n,1))
for k in range(numAgents):
    ind1 = k*numStates
    ind2 = k*numStates + numStates
    xTrue[ind1:ind2] = xi[ind1:ind2,k].reshape((numStates,1))

yTrue = C@xTrue


######################
# Simulation Loop
######################
counter = 0
dist = np.zeros((numAgents-1,numAgents))

xStar = np.zeros((n,numFun))
yStar = np.zeros((m,numFun))

xHist = np.zeros((n,numFun*iters))
yHist = np.zeros((m,numFun*iters))
exHist = np.zeros((numFun*iters))
eyHist = np.zeros((numFun*iters))
eAltitudes = np.zeros((numFun*iters))
eAccelerations = np.zeros((numFun*iters))

xStarHist = np.zeros((n, numFun*iters))
yStarHist = np.zeros((m, numFun*iters))
accelDesHist = np.zeros((numAgents, numFun*iters))
yRef = np.zeros((m))
#xHist[:,0] = xTrue.reshape((n,))

# Function Switching Loop
for i in range(numFun):
    #yRef[::2] = accelDesired[i]
    a = 0.1
    #yRef[::2] = a*(hDesired[i] - np.diag(yi[1::2,:]))/ts
    yRef[::2] = a/ts * (hDesired[i] - np.mean(yi[1::2,:],axis=0) )
    yRef[1::2] = hDesired[i] # Update next desired altitude
    #yRef[::2] = accelDesired + i*1

    result = minimize(obj_func, x0=np.zeros((n,1)), args=(C,Q,P,R,yRef,numAgents,numStates,sepDesired))
    xStar[:,i] = result.x
    yStar[:,i] = C @ xStar[:,i]

    # Asynchronous Agent Update Loop
    for j in range(iters):

        accelDesHist[:,i*iters+j] = yRef[::2]
        ##### COMMUNICATION #####
        comReality = np.random.rand(numAgents, 1)

        # Check if its time to communicate
        communicate = [x for x in range(len(comReality)) if (comReality[x] < comProb[x] or lastCom[x] < counter-B+1)]
        # If list not empty then communicate
        if communicate:
            for k in communicate:
                ind1 = k*numStates
                ind2 = k*numStates + numStates
                ind3 = k*numOutputs
                ind4 = k*numOutputs + numOutputs
                xi[ind1:ind2,:] = np.matlib.repmat(xi[ind1:ind2,k].reshape((numStates,1)),1,numAgents)
                yi[ind3:ind4,:] = np.matlib.repmat(yi[ind3:ind4,k].reshape((numOutputs,1)),1,numAgents)
                lastCom[k] = counter


        ##### UPDATE #####
        # Calculate distance between each agent
        altitudes = yi[1::2]
        for ii in range(numAgents):
            for jj in range(numAgents-1):
                dist[jj,ii] = abs(altitudes[jj,ii] - altitudes[jj+1,ii])
                dist[jj,ii] = altitudes[jj,ii] - altitudes[jj+1,ii]

        #print(dist)
        dJdx1 = Q@xi
        dJdx2 = C.T @ P @ (yi - np.matlib.repmat(yRef.reshape((m,1)),1,numAgents))
        dJdx3 = Cbar.T @ R @ (dist-sepDesired)
        dJdx = dJdx1 + dJdx2 + dJdx3

        xNew = xi - gamma * dJdx
        
        # Project onto constraint set
        for ii in range(numAgents):
            # print(ii)
            # print(xNew[ii*numStates:(ii+1)*numStates,ii])
            maxCheck = xNew[ii*numStates:(ii+1)*numStates,ii]>=xMax
            minCheck = xNew[ii*numStates:(ii+1)*numStates,ii]<=xMin
            # print(maxCheck)
            # print(any(maxCheck))
            if any(maxCheck):
                idx = [i for i, x in enumerate(maxCheck) if x]
                dummy1 = xNew[ii*numStates:(ii+1)*numStates,ii].copy()
                for jj in idx:
                    dummy1[jj] = xMax[jj]
                xNew[ii*numStates:(ii+1)*numStates,ii] = dummy1.copy()
                print("Element over xMax")
            if any(minCheck):
                idx = [i for i, x in enumerate(minCheck) if x]
                dummy2 = xNew[ii*numStates:(ii+1)*numStates,ii].copy()
                for jj in idx:
                    dummy2[jj] = xMin[jj]
                xNew[ii*numStates:(ii+1)*numStates,ii] = dummy2.copy()
                print("Element under xMin")

                

        upReality = np.random.rand(numAgents, 1)
        update = [x for x in range(len(upReality)) if (upReality[x] < upProb[x] or lastUp[x] < counter-B+1)]

        if update:
            for k in update:
                ind1 = k*numStates
                ind2 = k*numStates + numStates
                xi[ind1:ind2,k] = xNew[ind1:ind2,k]
                lastUp[k] = counter


        ##### MEASURE #####
        measReality = np.random.rand(numAgents, 1)

        # Check if its time to communicate
        measure = [x for x in range(len(measReality)) if (measReality[x] < measProb[x] or lastMeas[x] < counter-B+1)]
        # If list not empty then communicate
        if measure:
            for k in measure:
                ind3 = k*numOutputs
                ind4 = k*numOutputs + numOutputs
                yi[ind3:ind4,k] = yTrue[ind3:ind4].reshape((numOutputs,))
                lastMeas[k] = counter



        # Update True State
        for k in range(numAgents):
            ind1 = k*numStates
            ind2 = k*numStates + numStates
            xTrue[ind1:ind2] = xi[ind1:ind2,k].reshape((numStates,1))

        yTrue = C @ xTrue

        xHist[:,counter] = xTrue.reshape((n,))
        yHist[:,counter] = yTrue.reshape((m,))

        exHist[counter] = np.linalg.norm(xHist[:,counter] - xStar[:,i], ord=1)
        eyHist[counter] = np.linalg.norm(yHist[:,counter] - yStar[:,i], ord=1)
        
        # eAccelerations[counter] = np.linalg.norm(yHist[0,counter] - yStar[0,i])
        # eAltitudes[counter] = np.linalg.norm(yHist[1,counter] - yStar[1,i])
        
        eAccelerations[counter] = np.linalg.norm(yHist[::2,counter] - yRef[::2])
        eAltitudes[counter] = np.linalg.norm(yHist[1::2,counter] - yStar[1::2,i]) 

        xStarHist[:, counter] = xStar[:, i].reshape((n,))
        yStarHist[:, counter] = yStar[:, i].reshape((m,))

        counter += 1

vStar = xStar[0::numStates,:]
aoaStar = xStar[1::numStates,:]
pitchRateStar = xStar[2::numStates,:]
pitchStar = xStar[3::numStates,:]
hStar = xStar[4::numStates,:]

#%%
###################
# PLOTS
###################

t = np.arange(numFun*iters)
hStarPlot = np.zeros((numAgents,numFun*iters))

for j in range(numAgents):
    hStarPlot[j,:] = np.repeat(hStar[j,:],iters)
    
    
    
new_tick_locations  = np.arange(start=0, stop=(numFun+1)*iters, step=iters)

fig = plt.figure()
ax5 = fig.add_subplot(111)
ax6 = ax5.twiny()
for i in range(numAgents):
    ax5.plot(t, yHist[i*numOutputs,:], color='b')
    ax5.plot(t, accelDesHist[i, :], color='r', linestyle='dashed')
ax5.set_xlabel("Iterations $(k)$", fontsize =14)
ax5.set_ylabel("Acceleration $(ft/s^2)$", fontsize=14)
# ax3.set_title("Accelerations")
ax5.legend(["Actual","Optimal"], fontsize =11, loc = "lower left")
ax5.grid()
ax6.set_xlim(ax5.get_xlim())
ax6.set_xticks(new_tick_locations)
ax6.set_xticklabels((new_tick_locations/iters).astype(int))
ax6.set_xlabel(r"Time Index $(t_\ell)$", fontsize =14)
#plt.savefig('PythonPlotsNoTitle3/plot1.eps', format = 'eps', bbox_inches='tight')
plt.show()

fig = plt.figure()
ax7 = fig.add_subplot(111)
ax8 = ax7.twiny()
for i in range(numAgents):
    ax7.plot(t, yHist[i*numOutputs+1,:], color='b', label = 'Actual Altitude')
    ax7.plot(t, hStarPlot[i,:], color='r', linestyle='dashed', label = 'Optimal Altitude')
ax7.set_xlabel("Iterations $(k)$", fontsize =14)
ax7.set_ylabel("Altitude $(ft)$", fontsize=14)
ax7.legend(["Actual","Optimal"], fontsize =12)
ax7.grid()
ax8.set_xlim(ax7.get_xlim())
ax8.set_xticks(new_tick_locations)
ax8.set_xticklabels((new_tick_locations/iters).astype(int))
ax8.set_xlabel(r"Time Index $(t_\ell)$", fontsize =14)
# plt.savefig('PythonPlotsNoTitle3/plot2.eps', format = 'eps', bbox_inches='tight')
plt.show()


fig1,ax1 = plt.subplots(2,1)
# ax1[0] = fig.add_subplot(111)
# ax3 = fig.add_subplot(112)
ax2 = ax1[0].twiny()

ax1[0].plot(t, eAltitudes, color = 'r', label = "Altitude Error")
ax1[1].plot(t, eAccelerations,color = 'b', label = "Acceleration Error")

ax1[0].set_ylabel('Altitude Error $(ft)$', fontsize=12)
ax1[0].grid()

ax1[1].set_xlabel("Iterations $(k)$", fontsize=14)
ax1[1].set_ylabel('Acceleration Error $(ft/s^2)$', fontsize=12)
ax1[1].grid()

ax2.set_xlim(ax1[0].get_xlim())
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels((new_tick_locations/iters).astype(int))
ax2.set_xlabel(r"Time Index $(t_\ell)$", fontsize =14)

# plt.savefig('PythonPlotsNoTitle3/plot3.eps', format = 'eps', bbox_inches='tight')
plt.show()


fig4, ax4 = plt.subplots(2,2)
plt.style.use('default')
for i in range(numAgents):
    ax4[0,0].plot(t, xHist[i*numStates, :], color='b', label='true') # Velocity
    ax4[1,0].plot(t, xHist[i*numStates+1, :], color='b', label='true') # Angle of attack
    ax4[0,1].plot(t, xHist[i*numStates+2, :], color='b', label='true') # pitch rate
    ax4[1,1].plot(t, xHist[i*numStates+3, :], color='b', label='true') # pitch
ax4[0,0].set_ylabel("Velocities")
ax4[1,0].set_ylabel("Angle of Attack")
ax4[0,1].set_title("Pitch Rate")
ax4[1,1].set_xlabel("Pitch")
ax4[0,0].grid()
ax4[1,0].grid()
ax4[0,1].grid()
ax4[1,1].grid()
plt.show()

