import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 指定中文字体文件路径（请根据您的系统更改路径）
font_path = '/System/Library/Fonts/PingFang.ttc'  # macOS 上的一个中文字体
prop = fm.FontProperties(fname=font_path)

plt.rcParams['font.family'] = prop.get_name()

# 定义骨骼连接
connections = [
    (0, 1), (0, 2), (0, 3),  # 骨盆到左右髋和脊柱
    (1, 4), (4, 7), (7, 10),  # 左腿
    (2, 5), (5, 8), (8, 11),  # 右腿
    (3, 6), (6, 9), (9, 12), (12, 15),  # 脊柱到头部
    (9, 13), (9, 14),  # 脊柱到左右锁骨
    (13, 16), (16, 18), (18, 20), (20, 22),  # 左臂
    (14, 17), (17, 19), (19, 21), (21, 23)   # 右臂
]

def visualize_joints(joints):
    """
    可视化3D关节位置

    参数：
    joints: numpy数组，形状为(关节数， 3)，包含每个关节的x, y, z坐标
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制关节点
    ax.scatter(joints[:, 2], joints[:, 0], -joints[:, 1], c='r', s=50)

    # 添加关节索引标签
    for i, joint in enumerate(joints):
        ax.text(joint[2], joint[0], -joint[1], str(i), fontsize=8)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置视角
    ax.view_init(elev=20, azim=45)

    plt.title('3D Joint Positions')
    plt.show()

def animate_3d_poses(joints, transform_matrix=[2, 0, 1], scale_factor=[1, 1, -1]):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 创建45度旋转矩阵
    import numpy as np
    rotation_angle = np.pi / 2  # 90度
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])

    transformed_joints = joints[:, :, transform_matrix] * scale_factor
    transformed_joints = np.dot(transformed_joints.reshape(-1, 3), rotation_matrix).reshape(transformed_joints.shape)
    # 初始化散点
    scat = ax.scatter(transformed_joints[0, :, 0], transformed_joints[0, :, 1], transformed_joints[0, :, 2])
    # 初始化线条
    lines = [ax.plot([], [], [], 'b-')[0] for _ in connections]

    # 设置初始视角
    ax.view_init(elev=10, azim=45)

    # 设置坐标轴标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')

    # 更新函数
    def update(frame, scat, lines):
        # 更新散点位置
        transformed_joint = transformed_joints[frame, :, :]
        scat._offsets3d = (transformed_joint[:, 0], transformed_joint[:, 1], transformed_joint[:, 2])

        # 更新线条
        for line, connection in zip(lines, connections):
            start, end = connection
            line.set_data([transformed_joint[start, 0], transformed_joint[end, 0]],
                        [transformed_joint[start, 1], transformed_joint[end, 1]])
            line.set_3d_properties([transformed_joint[start, 2], transformed_joint[end, 2]])

        # 获取当前帧的坐标范围
        x_min, x_max = transformed_joint[:, 0].min(), transformed_joint[:, 0].max()
        y_min, y_max = transformed_joint[:, 1].min(), transformed_joint[:, 1].max()
        z_min, z_max = (transformed_joint[:, 2]).min(), (transformed_joint[:, 2]).max()

        # 计算坐标范围的中心和宽度
        center_x, width_x = (x_min + x_max) / 2, x_max - x_min
        center_y, width_y = (y_min + y_max) / 2, y_max - y_min
        center_z, width_z = (z_min + z_max) / 2, z_max - z_min

        # 设置新的坐标轴范围，稍微扩大一些以留出边距
        max_width = max(width_x, width_y, width_z) * 1.2
        ax.set_xlim(center_x - max_width/2, center_x + max_width/2)
        ax.set_ylim(center_y - max_width/2, center_y + max_width/2)
        ax.set_zlim(center_z - max_width/2, center_z + max_width/2)

        # 确保坐标轴标签始终可见
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')

        return scat, *lines

    # 创建动画
    import matplotlib.animation as animation
    anim = animation.FuncAnimation(fig, update, frames=len(joints), fargs=(scat, lines), interval=50, blit=False)

    plt.tight_layout()
    plt.show()

# 假设我们已经加载了SMPL参数
import os
base_dir = os.environ.get("FREEMAN_DATASET_BASE_DIR")
motion_dir = os.environ.get("FREEMAN_DATASET_MOTION_DIR")
seq_name = os.environ.get("FREEMAN_DATASET_SEQ_NAME")
model_path = os.environ.get("SMPL_MODEL_PATH")

from freeman_loader import FreeMan
smpl_poses, smpl_scaling, smpl_trans = FreeMan(base_dir=base_dir) \
                                    .load_motion(
                                        motion_dir=motion_dir,
                                        seq_name=seq_name
                                    )

# 创建SMPL模型
from smplx import SMPL
smpl = SMPL(model_path=model_path, gender='neutral', batch_size=1)

print("smpl_poses, smpl_scaling, smpl_trans", smpl_poses.shape, smpl_scaling.shape, smpl_trans.shape)

import torch
joints = smpl.forward(
    global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
    body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
    transl=torch.from_numpy(smpl_trans).float(),
    scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(),
).joints.detach().numpy()

# 可视化关节
print("joints", joints.shape)

# frame = 10
# joints = joints[frame]
# visualize_joints(joints)

# 创建动画
animate_3d_poses(joints, transform_matrix=[2, 0, 1])