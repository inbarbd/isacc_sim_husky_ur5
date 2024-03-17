from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import (quat_conjugate, quat_mul, quat_apply)
import numpy as np
import torch
import math
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
gym_GREEN = gymapi.Vec3(0., 1., 0.)
gym_BLUE = gymapi.Vec3(0., 0., 1.)

def get_sim_params(args):
    """Start up a common set of simulation parameters.

    Based on `franka_cube_ik_osc.py` provided by Isaac Gym authors.
    """
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1/120
    sim_params.substeps = 2
    # sim_params.stress_visualization = False
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 2
        sim_params.physx.num_position_iterations = 10
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset =0.02 #no less than1-2cm
        sim_params.physx.friction_offset_threshold = 0.005
        sim_params.physx.friction_correlation_distance = 0.005
        # sim_params.flex.num_outer_iterations = 4
        # sim_params.flex.num_inner_iterations = 10
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.always_use_articulations = False
    else:
        raise Exception("This example can only be used with PhysX")
    return sim_params

def get_robot_asset(gym, sim, asset_root, asset_file):
    asset_options = gymapi.AssetOptions()
    asset_options.flip_visual_attachments = True
    asset_options.fix_base_link = False
    asset_options.collapse_fixed_joints = False
    asset_options.thickness = 0.0  # default = 0.02
    asset_options.density = 1000.0  # default = 1000.0
    asset_options.armature = 0.01  # default = 0.0
    asset_options.use_physx_armature = True
    # if self.cfg_base.sim.add_damping:
    # asset_options.linear_damping = 0.0  # default = 0.0; increased to improve stability
    # asset_options.max_linear_velocity = 10.0  # default = 1000.0; reduced to prevent CUDA errors
    # asset_options.angular_damping = 5.0  # default = 0.5; increased to improve stability
    # asset_options.max_angular_velocity = 2 * math.pi  # default = 64.0; reduced to prevent CUDA errors
    # else:
    asset_options.linear_damping = 0.0  # default = 0.0
    asset_options.max_linear_velocity = 1000.0  # default = 1000.0
    asset_options.angular_damping = 0.5  # default = 0.5
    asset_options.max_angular_velocity = 64.0  # default = 64.0
    asset_options.disable_gravity = False
    asset_options.enable_gyroscopic_forces = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL  # DOF_MODE_NONE
    asset_options.use_mesh_materials = True
    # if self.cfg_base.mode.export_scene:
    # asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE
    asset_options.override_com = False
    asset_options.override_inertia = False
    asset_options.replace_cylinder_with_capsule = True
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return robot_asset

def get_sphere_asset(gym, sim, radius):
    """Create a sphere asset and disable gravity."""
    asset_options = gymapi.AssetOptions()
    asset_options.disable_gravity = True
    sphere_asset = gym.create_sphere(sim, radius, asset_options)
    return sphere_asset

def get_capsule_asset(gym, sim, radius,length):
    """Create a capsule asset and disable gravity."""
    asset_options = gymapi.AssetOptions()
    asset_options.disable_gravity = True
    # asset_options.fix_base_link = True
    capsule_asset = gym.create_capsule(sim, radius,length, asset_options)
    return capsule_asset

def get_box_asset(gym, sim, width, height, depth):
    """Create a capsule asset and disable gravity."""
    asset_options = gymapi.AssetOptions()
    asset_options.disable_gravity = True
    asset_options.fix_base_link = True
    capsule_asset = gym.create_box(sim, width, height, depth, asset_options)
    return capsule_asset

def sample_sphere_surface(center_sphere, radius, n_points=1):
    """Sample about a sphere surface, centered at `center_sphere`.

    Samples IID standard Gaussians. Then normalize and multiply each by the radius.
    Returns: points, shaped (N,3).
    """
    assert radius > 0
    xyz_N3 = np.random.normal(loc=0.0, scale=1.0, size=(n_points, 3))
    xyz_N3 = xyz_N3 * (radius / np.linalg.norm(xyz_N3, axis=1, keepdims=True))
    return center_sphere + xyz_N3

def get_target_orientation(direction='top_down'):
    """Some sample target poses."""
    if direction == 'top_down':
        return gymapi.Quat(1.0, 0.0, 0.0, 0.0)
    elif direction == 'bottom_up':
        return gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    else:
        raise ValueError(direction)

def control_ik(j_eef, dpose, num_envs, num_dofs, damping=0.05):
    """Solve damped least squares, from `franka_cube_ik_osc.py` in Isaac Gym.

    Returns: Change in DOF positions, [num_envs,num_dofs], to add to current positions.
    """
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6).to(j_eef_T.device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, num_dofs)
    return u

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def move_sphere_ee_vectorized(hand_pos, hand_rot, eetip_offsets):
    """Move spheres representing the EE tip by applying an offset correction.

    Returns: eetip_pos, shape (num_envs,3)
    """
    eetip_pos = hand_pos + quat_apply(hand_rot, eetip_offsets)
    return eetip_pos

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="IK example, debug CPU vs GPU differences.",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 10},
        {"name": "--seed", "type": int, "default": 10},
        ])

np.random.seed(args.seed)
torch.manual_seed(args.seed)
num_envs = args.num_envs
args.use_gpu = args.use_gpu_pipeline
sim_params = get_sim_params(args)
device = args.sim_device if args.use_gpu_pipeline else 'cpu'
print(f"args: {args}\ndevice: {device}")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0.0
gym.add_ground(sim, plane_params)

# Create viewer.
camera_props = gymapi.CameraProperties()
camera_props.horizontal_fov = 75.0
camera_props.width = 1920
camera_props.height = 1080
viewer = gym.create_viewer(sim, camera_props)
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load assets
asset_root = "/home/inbarm/isacc_sim_share"
asset_file = "husky_tau_isaac_2nd.urdf"

robot_asset = get_robot_asset(gym, sim, asset_root, asset_file)
eetip_asset = get_sphere_asset(gym, sim, radius=0.03)
box_asset = get_box_asset(gym, sim, width=0.1, height=0.1, depth=0.1)
ur5_dof_props = gym.get_asset_dof_properties(robot_asset)
ur5_lower_limits = ur5_dof_props["lower"]
ur5_upper_limits = ur5_dof_props["upper"]

ur5_ranges = ur5_upper_limits - ur5_lower_limits
ur5_mids = 0.5 * (ur5_upper_limits + ur5_lower_limits)

# UR5-specific starting pose.
ur5_pose = gymapi.Transform()
ur5_position = [0.0, 0.0, 0.1]
ur5_pose.p = gymapi.Vec3(ur5_position[0], ur5_position[1], ur5_position[2])
ur5_pose.r = gymapi.Quat(0.0, 0, 0.0, 1)

# Use position drive for all dofs, following Franka example.
ur5_dof_props["driveMode"].fill(gymapi.DOF_MODE_VEL)
ur5_dof_props["stiffness"][:5].fill(40.0)
ur5_dof_props["damping"][:5].fill(400.0)
ur5_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
ur5_dof_props["stiffness"][4:].fill(400.0)
ur5_dof_props["damping"][4:].fill(40.0)

# Default dof states and position targets, following Franka example.
ur5_num_dofs = gym.get_asset_dof_count(robot_asset)
default_dof_pos = np.zeros(ur5_num_dofs, dtype=np.float32)
# default_dof_pos[4:] = ur5_mids[4:]
default_dof_state = np.zeros(ur5_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos
print(default_dof_pos,"default_dof_pos")
# Get link index for end-effector.
ur5_link_dict = gym.get_asset_rigid_body_dict(robot_asset)
ur5_ee_index = ur5_link_dict["tool0"]

# Set up the grid of environments
num_per_row = int(np.sqrt(num_envs))
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)


# Information to cache.
envs = []
targ_handles = []
targ_idxs = []
ee_idxs = []
eetip_handles = []
eetip_idxs = []
init_pos_list = []
init_rot_list = []

# Use sphere sampling to sample targets for IK.
target_points = sample_sphere_surface(
    center_sphere=(1.0, 0.0, 0.2), radius=0.01, n_points=num_envs,
)

for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # Actor 0: UR5, set dof properties and initial DOF states / targets.
    ur5_handle = gym.create_actor(env, robot_asset, ur5_pose, "ur5", i, 0)
    gym.set_actor_dof_properties(env, ur5_handle, ur5_dof_props)
    gym.set_actor_dof_states(env, ur5_handle, default_dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, ur5_handle, default_dof_pos)
    gym.set_actor_rigid_body_properties(env,ur5_handle,gym.get_actor_rigid_body_properties(env,ur5_handle),False)
   
    eetip_pose = gymapi.Transform()
    eetip_id = f"eetip_{i}"
    eetip_coll = num_envs + 2
   
    
    # Track the indices of the 'last' UR5 link. For the 'real tip' see `eetip_idx`.
    ee_idx = gym.find_actor_rigid_body_index(env, ur5_handle, "tool0", gymapi.DOMAIN_SIM)
    ee_idxs.append(ee_idx)

    # Get inital hand pose, this is the same across all envs.
    ee_handle = gym.find_actor_rigid_body_handle(env, ur5_handle, "tool0")
    ee_pose = gym.get_rigid_transform(env, ee_handle)
    init_pos_list.append([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z])
    init_rot_list.append([ee_pose.r.x, ee_pose.r.y, ee_pose.r.z, ee_pose.r.w])

    box_pose = gymapi.Transform()
    box_pose.p = gymapi.Vec3(0.5, 0.5, 0)
    box_pose.r = gymapi.Quat(0.0, 0, 0.0, 1)

    box_id = f"box_{i}"
    print(box_id,"box_id")
    box_coll = num_envs + 2
    box_handle = gym.create_actor(env, box_asset, box_pose, box_id, box_coll, 0)
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL, gym_BLUE)

# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[args.num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ============================== prepare tensors ===================================
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)
eetip_offsets = torch.tensor([0., 0., 0.18], device=device).repeat((num_envs,1))

# get jacobian tensor, I think tensor shape (num_envs, num_links, [pose], num_joints-dof)
_jacobian = gym.acquire_jacobian_tensor(sim, "ur5")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to ur5 ee
j_eef = jacobian[:, ur5_ee_index - 1, :, :]

# Actor root state tensor, only to control debugging sphere target if desired.
_actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
root_state_tensor = gymtorch.wrap_tensor(_actor_root_state_tensor).view(num_envs, -1, 13)

# Get rigid body state tensor, shape (num_RBs, 13).
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# Get dof state tensor, we will be querying to get current DOF state, and adding to it.
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
ur5_dof_states = dof_states.view(num_envs, -1, 2)[:,:ur5_num_dofs]
# print(ur5_dof_states,"ur5_dof_states")
dof_pos = ur5_dof_states[:,:,0]
dof_pos = torch.unsqueeze(dof_pos, 2).to(device)
# print(dof_pos,"dof_pos")

total_dofs = gym.get_sim_dof_count(sim) // num_envs
iter = 1
go = True
while not gym.query_viewer_has_closed(viewer):
    ur5_dof_targets = torch.zeros((num_envs, total_dofs), dtype=torch.float, device=device)
    husky_dof_vel = torch.zeros((num_envs, total_dofs), dtype=torch.float, device=device)
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Refresh Tensors
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    hand_pos = rb_states[ee_idxs, :3]  # this is the UR5 joint 'before' the tip
    hand_rot = rb_states[ee_idxs, 3:7]

    _pos_target = dof_pos.squeeze(-1)+0   
    _vel_target = dof_pos.squeeze(-1)+0   
    
    if iter>100: 
        _pos_target[:,4] = 0.00  
        _pos_target[:,5] = -0.20  
        _pos_target[:,6] = -0.20  
        _pos_target[:,7] = -0.20  
        _pos_target[:,8] = -0.20  
        _pos_target[:,9] = -0.20
    if iter>200 and iter <300:
        _pos_target[:,4] = 0.00  
        _pos_target[:,5] = -0.10  
        _pos_target[:,6] = -0.10  
        _pos_target[:,7] = -0.10  
        _pos_target[:,8] = -0.10  
        _pos_target[:,9] = -0.10  
    if iter>300 and iter <400:
        _pos_target[:,4] = 0.00  
        _pos_target[:,5] = -0.20  
        _pos_target[:,6] = -0.20  
        _pos_target[:,7] = -0.20  
        _pos_target[:,8] = -0.20  
        _pos_target[:,9] = -0.20  
    if iter>400 and iter <500:
        
        _pos_target[:,4] = 0.00  
        _pos_target[:,5] = -0.10  
        _pos_target[:,6] = -0.10  
        _pos_target[:,7] = -0.10  
        _pos_target[:,8] = -0.10  
        _pos_target[:,9] = -0.10  
    
    _vel_target[:,0] = -10
    _vel_target[:,1] = 10 
    _vel_target[:,2] = -10
    _vel_target[:,3] = 10 

    ur5_dof_targets[:,:ur5_num_dofs] = _pos_target
    husky_dof_vel[:,:ur5_num_dofs] = _vel_target

    gym.set_dof_position_target_tensor(
        sim, gymtorch.unwrap_tensor(ur5_dof_targets)
    )
    gym.set_dof_velocity_target_tensor(
        sim, gymtorch.unwrap_tensor(husky_dof_vel)
    )

    
    # Update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    # print(iter,"iter")
    iter = iter + 1

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)