import numpy as np
import matplotlib as mtpltlb
import mujoco
import mujoco.viewer
import math
import time
import trajectoryplanning_ik as ik
import scipy

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
    footsteps = []
    left_x, left_y = -SL/2, hip_y_offset
    right_x, right_y = SL/2, -hip_y_offset

    swing_is_left = True

    for step in range(0, N_steps):
        if swing_is_left:
            stance_x = right_x
            swing_x = stance_x + SL
            left_x = swing_x
            footsteps.append((left_x, left_y, "Left"))
        else :
            stance_x = left_x
            swing_x = stance_x + SL
            right_x = swing_x
            footsteps.append((right_x, right_y, "Right"))
        
        swing_is_left = not swing_is_left

    return footsteps

def zmp_reference(footsteps, T_ssp, T_dsp, dt):
    p_ref_x, p_ref_y = [], []

    init_zmp_x = 0
    init_zmp_y = 0
    left_foot_x = -SL / 2
    right_foot_x = SL / 2
    n_dsp = int(T_dsp/dt)
    for k in range(0, int(T_dsp/dt)):
        p_ref_x.append(init_zmp_x)
        p_ref_y.append(init_zmp_y)
    
    for fx, fy, foot in footsteps:
        # During SSP: ZMP sits on the STANCE foot
        #   (the foot that is NOT swinging)
        if foot == "Left":
            stance_x = right_foot_x
            stance_y = -HIP_Y_OFFSET
        else:
            stance_x = left_foot_x
            stance_y = HIP_Y_OFFSET
        
        n_ssp = int(T_ssp/dt)
        for k in range(0, n_ssp):
            p_ref_x.append(stance_x)
            p_ref_y.append(stance_y)

        # During DSP: ZMP transitions linearly from old stance  
        # foot to the next stance foot (the foot that just landed)
        if foot == "Left":
            next_stance_x = fx 
            next_stance_y = HIP_Y_OFFSET
        else:
            next_stance_x = fx 
            next_stance_y = -HIP_Y_OFFSET
        n_dsp = int(T_dsp/dt)
        
        for k in range(0, n_dsp):
            alpha = k/max(n_dsp-1,1)
            p_ref_x.append(stance_x + alpha*(next_stance_x - stance_x))
            p_ref_y.append(stance_y + alpha*(next_stance_y - stance_y))

        # Update landed foot position
        if foot == "Left":
            left_foot_x = fx
        else:
            right_foot_x = fx

    # terminal padding to hold last zmp vallue 
    for k in range(0, N_preview):
        p_ref_x.append(p_ref_x[-1])
        p_ref_y.append(p_ref_y[-1])
    
    return p_ref_x, p_ref_y

def preview_control(dt, g, z_c, N):
    Q_e = 1.0 
    A = np.array([
        [1, dt, dt**2/2],
        [0, 1, dt],
        [0, 0, 1]
    ])
    B = np.array([[dt**3/6],
              [dt**2/2],
              [dt]])                 # shape (3,1)

    C = np.array([[1, 0, -z_c/g]])       # shape (1,3)
    
        # --- augmented system with integral of tracking error ---
    # state: [ e_zmp, x_state ]^T  where e_zmp integrates (p_ref - p)
    A_hat = np.block([
        [np.eye(1),           C @ A],
        [np.zeros((3, 1)),    A]
    ])
    B_hat = np.vstack([C @ B, B])          # (4,1)

    Q_hat = np.zeros((4, 4))
    Q_hat[0, 0] = Q_e                       # only penalize ZMP error

    R = 1e-6
    # --- solve discrete algebraic Riccati equation ---
    P = scipy.linalg.solve_discrete_are(A_hat, B_hat, Q_hat, np.array([[R]]))

    # --- feedback gains ---
    BPB = (B_hat.T @ P @ B_hat)[0, 0] + R
    K_hat = (1.0 / BPB) * (B_hat.T @ P @ A_hat)   # (1,4)

    Gi = K_hat[0, 0]        # integral (error) gain
    Gx = K_hat[0, 1:]       # state feedback gain (1,3)

    # --- preview gains G_d(l), l = 1..N ---
    Ac_hat = A_hat - B_hat @ K_hat          # closed-loop system matrix
    Gd = np.zeros(N)
    X = -Ac_hat.T @ P @ np.array([[1], [0], [0], [0]])  # seed vector

    for l in range(N):
        Gd[l] = (1.0 / BPB) * (B_hat.T @ X)[0, 0]
        X = Ac_hat.T @ X

    return A, B, C, Gx, Gi, Gd


# ===========================================================================
# Step 5: Preview Controller Servo Loop (CoM trajectory generation)
# ===========================================================================
def generate_com_trajectory(p_ref_x, p_ref_y, A, B, C, Gx, Gi, Gd):
    """
    Run the preview controller forward in time to produce CoM x/y trajectories.

    This replaces compute_smooth_trajectory (sagittal) and compute_lateral_sway
    (lateral) from the Bajrami approach with an optimal ZMP-tracking preview
    servo (Kajita 2003 §IV).
    """
    p_ref_x = np.array(p_ref_x, dtype=float)
    p_ref_y = np.array(p_ref_y, dtype=float)
    N = len(Gd)
    N_total = len(p_ref_x) - N

    # Sagittal (x) axis state
    x_state = np.zeros((3, 1))       # [x, x_dot, x_ddot]^T
    e_sum_x = 0.0                     # integral of ZMP error
    com_x = np.zeros(N_total)

    # Lateral (y) axis state
    y_state = np.zeros((3, 1))
    e_sum_y = 0.0
    com_y = np.zeros(N_total)

    for k in range(N_total):
        # --- Sagittal axis ---
        p_x = (C @ x_state)[0, 0]               # current ZMP x
        e_x = p_x - p_ref_x[k]                  # tracking error
        e_sum_x += e_x                           # integrate error

        # Preview sum: weight future ZMP reference values
        preview_sum_x = np.dot(Gd, p_ref_x[k + 1 : k + 1 + N])

        # Optimal jerk (control input)
        u_x = -Gi * e_sum_x - (Gx @ x_state)[0] - preview_sum_x

        # State update
        x_state = A @ x_state + B * u_x
        com_x[k] = x_state[0, 0]

        # --- Lateral axis (identical structure, different reference) ---
        p_y = (C @ y_state)[0, 0]
        e_y = p_y - p_ref_y[k]
        e_sum_y += e_y

        preview_sum_y = np.dot(Gd, p_ref_y[k + 1 : k + 1 + N])

        u_y = -Gi * e_sum_y - (Gx @ y_state)[0] - preview_sum_y

        y_state = A @ y_state + B * u_y
        com_y[k] = y_state[0, 0]

    return com_x, com_y


# ===========================================================================
# Step 7: Full walking pattern — tie everything together
# ===========================================================================
def generate_walking_pattern():
    """
    Full pipeline: footstep plan → ZMP reference → preview controller →
    CoM trajectory → foot swing trajectories → geometric IK → qpos frames.
    """
    # 1. Plan footsteps
    footsteps = plan_footsteps(NUM_STEPS, SL, HIP_Y_OFFSET)

    # 2. Build ZMP reference
    p_ref_x, p_ref_y = zmp_reference(footsteps, T_SSP, T_DSP, PLAYBACK_DT)

    # 3. Compute preview gains (offline, done once)
    A, B, C, Gx, Gi, Gd = preview_control(PLAYBACK_DT, g, z_c, N_preview)

    # 4. Generate CoM trajectory via preview servo
    com_x, com_y = generate_com_trajectory(
        p_ref_x, p_ref_y, A, B, C, Gx, Gi, Gd
    )

    # 5. Assemble joint-angle frames
    frames = []
    k = 0                           # global timestep index into com_x/com_y

    left_foot_x = -SL / 2.0
    right_foot_x = SL / 2.0
    swing_is_left = True
    hip_z = PELVIS_Z

    # --- Initial DSP (both feet on ground, robot stands still) ---
    n_dsp_init = int(T_DSP / PLAYBACK_DT)
    for i in range(n_dsp_init):
        if k >= len(com_x):
            break
        cur_hip_x = com_x[k]
        y_offset = com_y[k]
        leg_h = hip_z - ANKLE_REST_Z

        l_hip, l_knee, l_ankle = ik.ik_2link(
            cur_hip_x, hip_z, left_foot_x, ANKLE_REST_Z, L_UPPER, L_LOWER
        )
        r_hip, r_knee, r_ankle = ik.ik_2link(
            cur_hip_x, hip_z, right_foot_x, ANKLE_REST_Z, L_UPPER, L_LOWER
        )
        l_hr, l_ar = ik.compute_rolls(y_offset, leg_h)
        r_hr, r_ar = ik.compute_rolls(y_offset, leg_h)

        values = {
            "pelvis_x": cur_hip_x,
            "hip_roll_L": l_hr, "hip_pitch_L": l_hip, "knee_pitch_L": l_knee,
            "ankle_pitch_L": l_ankle, "ankle_roll_L": l_ar,
            "hip_roll_R": r_hr, "hip_pitch_R": r_hip, "knee_pitch_R": r_knee,
            "ankle_pitch_R": r_ankle, "ankle_roll_R": r_ar,
        }
        frames.append(ik.make_qpos(values))
        k += 1

    # --- Walking steps ---
    for step_idx in range(NUM_STEPS):
        if swing_is_left:
            stance_x = right_foot_x
            swing_start_x = left_foot_x
            swing_end_x = stance_x + SL
        else:
            stance_x = left_foot_x
            swing_start_x = right_foot_x
            swing_end_x = stance_x + SL

        # === SSP: swing foot lifts and advances ===
        n_ssp = int(T_SSP / PLAYBACK_DT)
        swing_x_traj, swing_z_traj = ik.compute_foot_swing_trajectory(
            PLAYBACK_DT, T_SSP, swing_end_x - swing_start_x,
            HM, swing_start_x, ANKLE_REST_Z
        )

        for i in range(n_ssp):
            if k >= len(com_x):
                break
            cur_hip_x = com_x[k]
            y_offset = com_y[k]
            leg_h = hip_z - ANKLE_REST_Z

            # Swing foot position from polynomial trajectory
            si = min(i, len(swing_x_traj) - 1)
            sw_ankle_x = swing_x_traj[si]
            sw_ankle_z = swing_z_traj[si]

            # Swing leg IK
            sw_hip_p, sw_knee_p, sw_ankle_p = ik.ik_2link(
                cur_hip_x, hip_z, sw_ankle_x, sw_ankle_z, L_UPPER, L_LOWER
            )
            sw_hr, sw_ar = ik.compute_rolls(y_offset, leg_h)

            # Stance leg IK (foot stays planted)
            st_hip_p, st_knee_p, st_ankle_p = ik.ik_2link(
                cur_hip_x, hip_z, stance_x, ANKLE_REST_Z, L_UPPER, L_LOWER
            )
            st_hr, st_ar = ik.compute_rolls(y_offset, leg_h)

            values = {"pelvis_x": cur_hip_x}
            if swing_is_left:
                values.update({
                    "hip_roll_L": sw_hr, "hip_pitch_L": sw_hip_p,
                    "knee_pitch_L": sw_knee_p, "ankle_pitch_L": sw_ankle_p,
                    "ankle_roll_L": sw_ar,
                    "hip_roll_R": st_hr, "hip_pitch_R": st_hip_p,
                    "knee_pitch_R": st_knee_p, "ankle_pitch_R": st_ankle_p,
                    "ankle_roll_R": st_ar,
                })
            else:
                values.update({
                    "hip_roll_L": st_hr, "hip_pitch_L": st_hip_p,
                    "knee_pitch_L": st_knee_p, "ankle_pitch_L": st_ankle_p,
                    "ankle_roll_L": st_ar,
                    "hip_roll_R": sw_hr, "hip_pitch_R": sw_hip_p,
                    "knee_pitch_R": sw_knee_p, "ankle_pitch_R": sw_ankle_p,
                    "ankle_roll_R": sw_ar,
                })
            frames.append(ik.make_qpos(values))
            k += 1

        # Update landed foot position
        if swing_is_left:
            left_foot_x = swing_end_x
        else:
            right_foot_x = swing_end_x

        # === DSP: both feet on ground, weight transfers ===
        n_dsp = int(T_DSP / PLAYBACK_DT)
        for i in range(n_dsp):
            if k >= len(com_x):
                break
            cur_hip_x = com_x[k]
            y_offset = com_y[k]
            leg_h = hip_z - ANKLE_REST_Z

            l_hip, l_knee, l_ankle = ik.ik_2link(
                cur_hip_x, hip_z, left_foot_x, ANKLE_REST_Z, L_UPPER, L_LOWER
            )
            r_hip, r_knee, r_ankle = ik.ik_2link(
                cur_hip_x, hip_z, right_foot_x, ANKLE_REST_Z, L_UPPER, L_LOWER
            )
            l_hr, l_ar = ik.compute_rolls(y_offset, leg_h)
            r_hr, r_ar = ik.compute_rolls(y_offset, leg_h)

            values = {
                "pelvis_x": cur_hip_x,
                "hip_roll_L": l_hr, "hip_pitch_L": l_hip,
                "knee_pitch_L": l_knee, "ankle_pitch_L": l_ankle,
                "ankle_roll_L": l_ar,
                "hip_roll_R": r_hr, "hip_pitch_R": r_hip,
                "knee_pitch_R": r_knee, "ankle_pitch_R": r_ankle,
                "ankle_roll_R": r_ar,
            }
            frames.append(ik.make_qpos(values))
            k += 1

        swing_is_left = not swing_is_left

    return frames, com_x, com_y, p_ref_x, p_ref_y


# ===========================================================================
# Main — debug plots + MuJoCo playback
# ===========================================================================
def main():
    print("=" * 60)
    print("FlexWalk — Hybrid IK + ZMP Preview Control")
    print("  Based on Kajita et al. (ICRA 2003)")
    print("=" * 60)
    print(f"  Upper leg  : {L_UPPER} m")
    print(f"  Lower leg  : {L_LOWER} m")
    print(f"  CoM height : {z_c} m")
    print(f"  Step length: {SL} m,  Foot clearance: {HM} m")
    print(f"  SSP: {T_SSP}s,  DSP: {T_DSP}s,  Steps: {NUM_STEPS}")
    print(f"  Preview horizon: {N_preview} samples ({N_preview * PLAYBACK_DT:.2f}s)")
    print("=" * 60)

    print("\nGenerating walking pattern...")
    frames, com_x, com_y, p_ref_x, p_ref_y = generate_walking_pattern()
    print(f"  Generated {len(frames)} frames ({len(frames) * PLAYBACK_DT:.2f}s)")
    print(f"  Net forward travel: {frames[-1][ik.QADR['pelvis_x']]:.3f} m")

    # Joint-limit verification
    violations = 0
    for fi, f in enumerate(frames):
        for name in ik.JOINT_NAMES:
            ji = model.joint(name).id
            lo, hi = model.jnt_range[ji]
            val = f[ik.QADR[name]]
            if val < lo - 0.01 or val > hi + 0.01:
                if violations < 3:
                    print(f"  WARN: frame {fi}, {name} = "
                          f"{math.degrees(val):+.1f}°/m, "
                          f"range=[{lo:.3f}, {hi:.3f}]")
                violations += 1
    if violations == 0:
        print("  ✓ All joint values within limits")
    else:
        print(f"  ⚠ {violations} limit violations (clamped at runtime)")

    # --- Debug plots ---
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Hybrid IK + ZMP Preview Control — Debug Plots", fontsize=14)

    t_com = np.arange(len(com_x)) * PLAYBACK_DT
    t_ref = np.arange(len(p_ref_x)) * PLAYBACK_DT

    # Sagittal: CoM x vs ZMP ref x
    axes[0, 0].plot(t_ref[:len(com_x)], p_ref_x[:len(com_x)],
                    'r--', linewidth=1, label='ZMP ref')
    axes[0, 0].plot(t_com, com_x, 'b-', linewidth=1.2, label='CoM x')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('x (m)')
    axes[0, 0].set_title('Sagittal: CoM vs ZMP Reference')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Lateral: CoM y vs ZMP ref y
    axes[0, 1].plot(t_ref[:len(com_y)], p_ref_y[:len(com_y)],
                    'r--', linewidth=1, label='ZMP ref')
    axes[0, 1].plot(t_com, com_y, 'b-', linewidth=1.2, label='CoM y')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('y (m)')
    axes[0, 1].set_title('Lateral: CoM vs ZMP Reference')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Top-down CoM path
    axes[1, 0].plot(com_x, com_y, 'b-', linewidth=1.2, label='CoM path')
    axes[1, 0].set_xlabel('x (m)')
    axes[1, 0].set_ylabel('y (m)')
    axes[1, 0].set_title('CoM Trajectory (Top View)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')

    # Pelvis x from frames over time
    pelvis_xs = [f[ik.QADR['pelvis_x']] for f in frames]
    t_frames = np.arange(len(frames)) * PLAYBACK_DT
    axes[1, 1].plot(t_frames, pelvis_xs, 'g-', linewidth=1.2,
                    label='pelvis_x (frames)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('x (m)')
    axes[1, 1].set_title('Pelvis X Position Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zmp_preview_debug.png', dpi=150)
    plt.show()
    print("  Debug plots saved to zmp_preview_debug.png")

    # --- MuJoCo playback ---
    print("\nLaunching MuJoCo viewer — gait plays in loop.")
    print("  Close the viewer window to exit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.2
        viewer.cam.lookat[:] = [0.0, 0.0, 0.35]

        while viewer.is_running():
            for qpos in frames:
                if not viewer.is_running():
                    break
                data.qpos[:] = ik.clamp_joints(qpos.copy(), model)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(PLAYBACK_DT)

            if viewer.is_running():
                time.sleep(0.3)


if __name__ == "__main__":
    main()