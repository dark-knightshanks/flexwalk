import numpy as np
import mujoco
import mujoco.viewer
import math
import time

model = mujoco.MjModel.from_xml_path(
    "mujoco_model/flexwalk.xml"
)

data = mujoco.MjData(model)
dt = model.opt.timestep
l1, l2 = 0.35, 0.25
T = 0.7
SL = 0.04
H = 0.01

def footsteps(dt, T, SL, H, a, b):
    a2 = 3.0*SL/(T**2)
    a3 = -2.0*SL/(T**3)
    b3 = -8.0*(H)/(3.0*(T**3))
    b4 = 4.0*(H)/(3.0*(T**4))
    x_traj = []
    z_traj = []
    y = 0
    t = 0
    while t<=T:
        xa = a2*(t**2) + a3*(t**3) + a
        za = b3*(t**3) + b4*(t**4) + b
        x_traj.append(xa)
        z_traj.append(za)
        t+=dt

    return x_traj, z_traj, y

def ik_solver(l1, l2, T, SL, H):
    xa, za, y = footsteps(dt, T, SL, H, 0, 0.0125)
    xh, yh, zh = 0, 0, 0.5
    theta_h_t, theta_k_t = [], []
    for x,z in zip(xa,za):
        D = math.sqrt(((xh-x)**2) + (zh-z)**2)
        if(abs(l1-l2) <= D <= (l1+l2)):
            theta_k = math.acos((l1**2 + l2**2 - D**2)/(2*l1*l2)) - math.pi
            alpha = math.atan2((zh-z),(xh-x))
            beta = math.acos((l1**2 + D**2 - l2**2)/(2*l1*D))
            theta_h = alpha-beta
            print("Foot:", x, y)
            print("alpha:", math.degrees(alpha))
            print("beta :", math.degrees(beta))
            print("hip  :", math.degrees(theta_h))
            print("knee :", math.degrees(theta_k))
            theta_h_t.append(theta_h)
            theta_k_t.append(theta_k)
            
        else:
            print("Not possible")
        
    return theta_h_t,theta_k_t


with mujoco.viewer.launch_passive(model, data) as viewer:
    hip_traj, knee_traj = ik_solver(l1, l2, T, SL, H)
    while viewer.is_running():
        for i,j in zip(hip_traj,knee_traj):
            data.qpos[1], data.qpos[2] = i, j
            mujoco.mj_forward(model,data)
            print(data.body("foot_L").xpos)
            # Update the viewer
            viewer.sync()
            time.sleep(0.02)
            
            


        