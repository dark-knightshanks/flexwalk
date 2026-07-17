import numpy as np
import matplotlib as mtpltlb
import mujoco
import mujoco.viewer
import math
import time
import trajectoryplanning_ik as ik

# Load mujoco model
model = mujoco.MjModel.from_xml_path("mujoco_model/flexwalk.xml")
data = mujoco.MjData(model)
sim_dt = model.opt.timestep


# ---------------------------------------------------------------------------
# Robot dimensions (from the MJCF file)
# ---------------------------------------------------------------------------
L_UPPER = 0.325   # hip_pitch joint → knee_pitch joint (m)
L_LOWER = 0.225   # knee_pitch joint → ankle_pitch joint (m)
L_ANKLE = 0.04    # ankle_pitch joint → foot sole (m)
HIP_Y_OFFSET = 0.07  # lateral hip offset from pelvis centre (m)
PELVIS_Z = 0.6    # fixed pelvis height (m) — no vertical DOF added

# Joint limits (radians, from the XML)
ANKLE_PITCH_LIM = math.radians(45)   # ±45°
HIP_PITCH_LO = math.radians(-60)     # -60°
HIP_PITCH_HI = math.radians(90)      # +90°
KNEE_PITCH_LO = math.radians(-120)   # -120°

# ---------------------------------------------------------------------------
# Gait parameters — tuned for this specific robot
# ---------------------------------------------------------------------------
T_SSP = 0.5       # Single Support Phase duration (s)
T_DSP = 0.15      # Double Support Phase duration (s)
T_STEP = T_SSP + T_DSP
g = 9.81           # gravity

SL = 0.06          # Step length (m) — kept small for the small robot
HM = 0.03          # Maximum foot clearance height during swing (m)
SM = 0.5           # Fractional time of max clearance (0–1)

# Ankle rest z: chosen so hip-to-ankle distance avoids singularity at L1+L2=0.55
# With ANKLE_REST_Z = 0.10, distance = 0.6-0.10 = 0.50, knee bends ~26°
ANKLE_REST_Z = 0.10
z_c = PELVIS_Z - ANKLE_REST_Z

NUM_STEPS = 16
PLAYBACK_DT = 0.01
N_preview = 150

# Lateral sway
SWAY_AMPLITUDE = 0.010

def plan_footsteps(N_steps, SL, hip_y_offset):
    footsteps = {}
    left_x, left_y = -SL/2, hip_y_offset
    right_x, right_y = SL/2, -hip_y_offset

    swing_is_left = True

    for step in range(0, N_steps-1):
        if swing_is_left:
            stance_x = right_x
            swing_x = stance_x + SL
            left_x = swing_x
            footsteps.append(left_x, left_y, "Left")
        else :
            stance_x = left_x
            swing_x = stance_x + SL
            right_x = swing_x
            footsteps.append(right_x, right_y, "Right")
        
        swing_is_left = not swing_is_left

    return footsteps

def zmp_reference(footsteps, T_ssp, T_dsp, dt):
    p_ref_x, p_ref_y = {}, {}

    init_zmp_x = 0
    init_zmp_y = 0
    n_dsp = int(T_dsp/dt)
    for k in range(0, int(T_dsp/dt)-1):
        p_ref_x.append(init_zmp_x)
        p_ref_y.append(init_zmp_y)
    
    for fx, fy, foot in footsteps:
        # During SSP: ZMP sits on the STANCE foot
        #   (the foot that is NOT swinging)
        if foot == "Left":
            stance_x = fx - SL
            stance_y = -HIP_Y_OFFSET
        else:
            stance_x = fx - SL
            stance_y = HIP_Y_OFFSET
        
        n_ssp = int(T_ssp/dt)
        for k in range(0, n_ssp-1):
            p_ref_x.append(stance_x)
            p_ref_y.append(stance_y)

        # During DSP: ZMP transitions linearly from old stance  
        # foot to the next stance foot (the foot that just landed)
        if foot == "Left":
            next_stance_x = fx 
            next_stance_y = -HIP_Y_OFFSET
        else:
            next_stance_x = fx 
            next_stance_y = HIP_Y_OFFSET
        n_dsp = int(T_dsp/dt)
        
        for k in range(0, n_dsp-1):
            alpha = k/max(n_dsp-1,1)
            p_ref_x.append(stance_x + alpha*(next_stance_x - stance_x))
            p_ref_y.append(stance_y + alpha*(next_stance_y - stance_y))

    # terminal padding to hold last zmp vallue 
    for k in range(0, N_preview):
        p_ref_x.append(p_ref_x[-1])
        p_ref_x.append(p_ref_x[-1])
    
    return p_ref_x, p_ref_y

def preview_control(dt, g, z_c):
    A = np.array([
        [1, dt, dt**2/2],
        [0, 1, dt],
        [0, 0, 1]
    ])
    B = np.array(((dt**3)/6), ((dt**2)/2), dt)
    C = np.array(1, 0, -z_c/g)
    


        



    





    





        
