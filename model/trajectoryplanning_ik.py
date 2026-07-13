"""
Trajectory Planning and Inverse Kinematics for a 10-DOF Biped Robot.

Implementation based on:
  Bajrami et al., "Trajectory Planning and Inverse Kinematics Solver
  for Real Biped Robot with 10 DOF-s", IFAC-PapersOnLine 49-29 (2016).

The biped has a fixed pelvis and 5 DOF per leg:
  hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll.

Gait is decomposed into:
  - SSP (Single Support Phase): swing foot follows polynomial trajectory
  - DSP (Double Support Phase): weight transfer between feet

Joint mapping (qpos indices):
  Left leg:  0=hip_roll_L, 1=hip_pitch_L, 2=knee_pitch_L, 3=ankle_pitch_L, 4=ankle_roll_L
  Right leg: 5=hip_roll_R, 6=hip_pitch_R, 7=knee_pitch_R, 8=ankle_pitch_R, 9=ankle_roll_R

MuJoCo Y-axis hinge convention:
  Positive hip_pitch → thigh swings backward (-X)
  Negative hip_pitch → thigh swings forward (+X)
  FK: knee_x = hip_x - L1*sin(θh), knee_z = hip_z - L1*cos(θh)
"""

import numpy as np
import mujoco
import mujoco.viewer
import math
import time

# ---------------------------------------------------------------------------
# Load MuJoCo model
# ---------------------------------------------------------------------------
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
PELVIS_Z = 0.6    # fixed pelvis height (m)

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

SL = 0.06          # Step length (m) — kept small for the small robot
HM = 0.03          # Maximum foot clearance height during swing (m)
SM = 0.5           # Fractional time of max clearance (0–1)

# Ankle rest z: chosen so hip-to-ankle distance avoids singularity at L1+L2=0.55
# With ANKLE_REST_Z = 0.10, distance = 0.6-0.10 = 0.50, knee bends ~26°
ANKLE_REST_Z = 0.10

NUM_STEPS = 16
PLAYBACK_DT = 0.01

# Lateral sway
SWAY_AMPLITUDE = 0.010


# ===========================================================================
# Section 2.2: Swing foot trajectory — polynomial (paper eqs. 12-13)
# ===========================================================================
def compute_foot_swing_trajectory(dt, T_s, step_length, foot_clearance,
                                  x_start, z_rest):
    """
    Swing foot ankle trajectory during SSP.

    x(t): 3rd-order polynomial — smooth start/stop.
    z(t): 5th-order polynomial — lifts, peaks at SM*Ts, returns.

    Returns lists of (ankle_x, ankle_z) positions.
    """
    Ts = T_s
    Sl = step_length
    Hm = foot_clearance
    Sm_t = SM * Ts

    # x-trajectory coefficients
    a0 = x_start
    a1 = 0.0
    a2 = 3.0 * Sl / (Ts**2)
    a3 = -2.0 * Sl / (Ts**3)

    # z-trajectory coefficients (5th order, solve 4×4 system)
    K = np.array([
        [Ts**2,   Ts**3,     Ts**4,     Ts**5    ],
        [2*Ts,    3*Ts**2,   4*Ts**3,   5*Ts**4  ],
        [Sm_t**2, Sm_t**3,   Sm_t**4,   Sm_t**5  ],
        [2*Sm_t,  3*Sm_t**2, 4*Sm_t**3, 5*Sm_t**4],
    ])
    rhs = np.array([0.0, 0.0, Hm, 0.0])
    b2, b3, b4, b5 = np.linalg.solve(K, rhs)

    x_traj, z_traj = [], []
    t = 0.0
    while t <= Ts + 1e-9:
        x_traj.append(a0 + a1*t + a2*t**2 + a3*t**3)
        z_traj.append(z_rest + b2*t**2 + b3*t**3 + b4*t**4 + b5*t**5)
        t += dt
    return x_traj, z_traj


# ===========================================================================
# Section 2.3: Hip trajectory — 3rd-order polynomial
# ===========================================================================
def compute_smooth_trajectory(dt, duration, x_start, x_end, v_start, v_end):
    """
    Smooth 3rd-order polynomial trajectory (hip motion per paper eq. 16/17).
    """
    T = duration
    c0 = x_start
    c1 = v_start
    A = np.array([[T**2, T**3], [2*T, 3*T**2]])
    b = np.array([x_end - c0 - c1*T, v_end - c1])
    c2, c3 = np.linalg.solve(A, b)

    traj = []
    t = 0.0
    while t <= T + 1e-9:
        traj.append(c0 + c1*t + c2*t**2 + c3*t**3)
        t += dt
    return traj


# ===========================================================================
# Lateral sway
# ===========================================================================
def compute_lateral_sway(dt, duration, y_start, y_end):
    """Half-cosine lateral weight shift."""
    traj = []
    t = 0.0
    while t <= duration + 1e-9:
        s = 0.5 * (1.0 - math.cos(math.pi * t / duration))
        traj.append(y_start + s * (y_end - y_start))
        t += dt
    return traj


# ===========================================================================
# 2-Link Geometric IK (matched to MuJoCo Y-axis hinge convention)
# ===========================================================================
def ik_2link(hip_x, hip_z, ankle_x, ankle_z, l1, l2):
    """
    Solve 2-link planar IK for hip→knee→ankle.

    MuJoCo FK:
        knee_x  = hip_x  - l1 · sin(θh)
        knee_z  = hip_z  - l1 · cos(θh)
        ankle_x = knee_x - l2 · sin(θh + θk)
        ankle_z = knee_z - l2 · cos(θh + θk)

    Returns (θ_hip, θ_knee, θ_ankle) where ankle keeps foot flat on ground.
    """
    dx = hip_x - ankle_x
    dz = hip_z - ankle_z
    D = math.sqrt(dx**2 + dz**2)

    # Clamp to feasible range
    D = max(abs(l1 - l2) + 1e-4, min(D, l1 + l2 - 1e-4))

    # Knee angle (law of cosines)
    cos_k = (l1**2 + l2**2 - D**2) / (2.0 * l1 * l2)
    cos_k = np.clip(cos_k, -1.0, 1.0)
    theta_knee = math.acos(cos_k) - math.pi   # ≤ 0

    # Hip angle
    gamma = math.atan2(dx, dz)
    cos_b = (l1**2 + D**2 - l2**2) / (2.0 * l1 * D)
    cos_b = np.clip(cos_b, -1.0, 1.0)
    beta = math.acos(cos_b)
    theta_hip = gamma + beta   # "knee-forward" solution

    # Ankle pitch to keep foot flat
    theta_ankle = -(theta_hip + theta_knee)

    # Clamp ankle pitch within limits; redistribute excess to hip
    if abs(theta_ankle) > ANKLE_PITCH_LIM:
        excess = theta_ankle - np.clip(theta_ankle, -ANKLE_PITCH_LIM, ANKLE_PITCH_LIM)
        theta_ankle = np.clip(theta_ankle, -ANKLE_PITCH_LIM, ANKLE_PITCH_LIM)
        theta_hip += excess  # absorb excess into hip
        # Q. why excess into hip here ? The knee primarily determines leg extension (how bent or straight the leg is). Changing the knee would change the distance between the hip and ankle, which would move the foot away from its desired position.
        #The hip, however, rotates the entire leg while keeping the same knee bend. A small hip adjustment mainly changes the leg's overall orientation, making it a better place to compensate for ankle saturation.

    return theta_hip, theta_knee, theta_ankle


def compute_rolls(y_offset, leg_height):
    """Hip roll and ankle roll for lateral weight shifting."""
    if abs(leg_height) < 0.01:
        return 0.0, 0.0
    hip_roll = math.atan2(y_offset, leg_height)
    return hip_roll, -hip_roll


# ===========================================================================
# Full gait trajectory generator
# ===========================================================================
def generate_gait_trajectory(num_steps, dt):
    """
    Generate a repeatable walking gait pattern.

    Since the pelvis is fixed, foot positions are computed relative to the
    fixed hip. The swing foot advances forward while the stance foot holds.
    After each complete step, both foot positions are re-centred around the
    hip to prevent drift — this creates a repeatable "marching" gait.

    Returns list of np.ndarray(10,) joint angle frames.
    """
    frames = []

    hip_x = 0.0
    hip_z = PELVIS_Z

    swing_is_left = True

    for step_idx in range(num_steps):
        # Re-centre foot positions at the start of each step to prevent drift.
        # This creates an alternating pattern: one foot forward, one back.
        half_step = SL / 2.0

        if swing_is_left:
            # Left foot is behind (about to swing forward), right is stance (ahead)
            swing_start_x = half_step
            stance_x = -half_step
        else:
            # Right foot is behind (about to swing forward), left is stance (ahead)
            swing_start_x = half_step
            stance_x = -half_step

        swing_end_x = half_step  # swing foot lands at the forward position

        # ---------------------------------------------------------------
        # Phase 1: SSP — swing foot lifts and advances
        # ---------------------------------------------------------------
        swing_x, swing_z = compute_foot_swing_trajectory(
            dt, T_SSP, SL, HM, swing_start_x, ANKLE_REST_Z
        )

        # Lateral sway: shift toward stance foot
        sway_target = -SWAY_AMPLITUDE if swing_is_left else SWAY_AMPLITUDE
        sway_start = 0.0
        y_sway_ssp = compute_lateral_sway(dt, T_SSP, sway_start, sway_target)

        n_ssp = min(len(swing_x), len(y_sway_ssp))

        for i in range(n_ssp):
            y_sway = y_sway_ssp[i]
            leg_h = hip_z - ANKLE_REST_Z

            # Swing leg IK
            sw_hip, sw_knee, sw_ankle = ik_2link(
                hip_x, hip_z, swing_x[i], swing_z[i], L_UPPER, L_LOWER
            )
            sw_hr, sw_ar = compute_rolls(y_sway, leg_h)

            # Stance leg IK — foot stays planted
            st_hip, st_knee, st_ankle = ik_2link(
                hip_x, hip_z, stance_x, ANKLE_REST_Z, L_UPPER, L_LOWER
            )
            st_hr, st_ar = compute_rolls(y_sway, leg_h)

            qpos = np.zeros(10)
            if swing_is_left:
                qpos[0], qpos[1], qpos[2] = sw_hr, sw_hip, sw_knee
                qpos[3], qpos[4] = sw_ankle, sw_ar
                qpos[5], qpos[6], qpos[7] = st_hr, st_hip, st_knee
                qpos[8], qpos[9] = st_ankle, st_ar
            else:
                qpos[0], qpos[1], qpos[2] = st_hr, st_hip, st_knee
                qpos[3], qpos[4] = st_ankle, st_ar
                qpos[5], qpos[6], qpos[7] = sw_hr, sw_hip, sw_knee
                qpos[8], qpos[9] = sw_ankle, sw_ar
            frames.append(qpos)

        # ---------------------------------------------------------------
        # Phase 2: DSP — both feet on ground, weight shifts to centre
        # ---------------------------------------------------------------
        y_sway_dsp = compute_lateral_sway(dt, T_DSP, sway_target, 0.0)

        n_dsp = len(y_sway_dsp)
        for i in range(n_dsp):
            y_sway = y_sway_dsp[i]
            leg_h = hip_z - ANKLE_REST_Z

            # Both feet on ground: one at +half_step, one at -half_step
            # After swing, the swing foot has landed at swing_end_x
            front_x = half_step
            back_x = -half_step

            l_ankle_x = front_x if swing_is_left else back_x
            r_ankle_x = back_x if swing_is_left else front_x

            l_hip, l_knee, l_ankle = ik_2link(
                hip_x, hip_z, l_ankle_x, ANKLE_REST_Z, L_UPPER, L_LOWER
            )
            r_hip, r_knee, r_ankle = ik_2link(
                hip_x, hip_z, r_ankle_x, ANKLE_REST_Z, L_UPPER, L_LOWER
            )

            l_hr, l_ar = compute_rolls(y_sway, leg_h)
            r_hr, r_ar = compute_rolls(y_sway, leg_h)

            qpos = np.zeros(10)
            qpos[0], qpos[1], qpos[2] = l_hr, l_hip, l_knee
            qpos[3], qpos[4] = l_ankle, l_ar
            qpos[5], qpos[6], qpos[7] = r_hr, r_hip, r_knee
            qpos[8], qpos[9] = r_ankle, r_ar
            frames.append(qpos)

        # Swap swing leg
        swing_is_left = not swing_is_left

    return frames


# ===========================================================================
# Joint limit clamping
# ===========================================================================
def clamp_joints(qpos, mdl):
    """Clamp joint angles to model ranges."""
    for i in range(mdl.njnt):
        lo, hi = mdl.jnt_range[i]
        idx = mdl.jnt_qposadr[i]
        qpos[idx] = np.clip(qpos[idx], lo, hi)
    return qpos


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("FlexWalk — Trajectory Planning & IK for 10-DOF Biped")
    print("  Based on Bajrami et al. (IFAC 2016)")
    print("=" * 60)
    print(f"  Upper leg  : {L_UPPER} m")
    print(f"  Lower leg  : {L_LOWER} m")
    print(f"  Ankle link : {L_ANKLE} m")
    print(f"  Pelvis z   : {PELVIS_Z} m (fixed)")
    print(f"  Ankle rest : {ANKLE_REST_Z} m (knee bend "
          f"≈ {abs(math.degrees(math.acos((L_UPPER**2+L_LOWER**2-(PELVIS_Z-ANKLE_REST_Z)**2)/(2*L_UPPER*L_LOWER)) - math.pi)):.0f}°)")
    print(f"  Step length: {SL} m,  Foot clearance: {HM} m")
    print(f"  SSP: {T_SSP}s,  DSP: {T_DSP}s,  Steps: {NUM_STEPS}")
    print("=" * 60)

    print("\nGenerating gait trajectory...")
    frames = generate_gait_trajectory(NUM_STEPS, PLAYBACK_DT)
    print(f"  Generated {len(frames)} frames ({len(frames)*PLAYBACK_DT:.2f}s)")

    # Verify joint limits
    joint_names = [
        "hip_roll_L", "hip_pitch_L", "knee_pitch_L",
        "ankle_pitch_L", "ankle_roll_L",
        "hip_roll_R", "hip_pitch_R", "knee_pitch_R",
        "ankle_pitch_R", "ankle_roll_R",
    ]
    violations = 0
    for fi, f in enumerate(frames):
        for ji in range(10):
            lo, hi = model.jnt_range[ji]
            if f[ji] < lo - 0.01 or f[ji] > hi + 0.01:
                if violations < 3:
                    print(f"  WARN: frame {fi}, {joint_names[ji]} = "
                          f"{math.degrees(f[ji]):+.1f}°, "
                          f"range=[{math.degrees(lo):.0f}°, {math.degrees(hi):.0f}°]")
                violations += 1
    if violations == 0:
        print("  ✓ All joint angles within limits")
    else:
        print(f"  ⚠ {violations} limit violations (clamped at runtime)")

    # Print sample angles
    print("\nSample frames (degrees):")
    for idx in [0, len(frames)//4, len(frames)//2]:
        print(f"  Frame {idx}:")
        for name, val in zip(joint_names, np.degrees(frames[idx])):
            print(f"    {name:20s} = {val:+7.2f}°")

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
                data.qpos[:] = clamp_joints(qpos.copy(), model)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(PLAYBACK_DT)

            if viewer.is_running():
                time.sleep(0.3)


if __name__ == "__main__":
    main()