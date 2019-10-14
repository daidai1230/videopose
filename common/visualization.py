# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import ipdb
import subprocess as sp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

pdb = ipdb.set_trace


def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)


def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)

    command = ['ffmpeg',
               '-i', filename,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vsync', '0',
               '-vcodec', 'rawvideo', '-']

    i = 0
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w*h*3)
            if not data:
                break
            i += 1
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))
            if i == limit:
                break


def downsample_tensor(X, factor):
    length = X.shape[0]//factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def render_animation(keypoints, poses, skeleton, fps, bitrate, azim, output, viewport,

                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    path = './data/' + output.split('.')[0]
    if not os.path.exists(path):
        os.mkdir(path)
    plt.ioff()
    # figsize = (10, 5)
    fig = plt.figure(figsize=(size*(1 + len(poses)), size + 2))
    # 2D
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)  # (1,2,1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    # 0, ('Reconstruction', 3d kp)
    for index, (title, data) in enumerate(poses.items()):
        # 3D
        ax = fig.add_subplot(1, 1 + len(poses), index+2, projection='3d')  # (1,2,2)
        ax.view_init(elev=15., azim=azim)
        # set 长度范围
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius/2, radius/2])
        ax.set_aspect('equal')
        # 坐标轴刻度
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])

        # lxy add
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # 轨迹 is base on position 0
        trajectories.append(data[:, 0, [0, 1]])  # only add x,y not z
    poses = list(poses.values())

    # ###################################################
    #  xx = trajectories[0]
    #  bad_points = []
    #  for i in  range(len(xx)-1):
    #  dis = xx[i+1] - xx[i]
    #  value = abs(dis[0]) + abs(dis[1])
    #  if value > 1e-06:
    #  bad_points.append(i)
    #  print(i, '      ',value)

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros(
            (keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        # 根据kp长度，决定帧的长度
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(
            np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(
                trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    numbers = []
    points = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    # array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    parents = skeleton.parents()
    kp_parents = [-1, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 0, 0, 11, 12, 13, 14]
    vis_enhence = True

    def update_video(i):
        nonlocal initialized, image, lines, points, numbers
        for num in numbers:
            num.remove()
        numbers.clear()
        for n, ax in enumerate(ax_3d):  # 只有1个
            ax.set_xlim3d([-radius/2 + trajectories[n][i, 0],
                           radius/2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius/2 + trajectories[n][i, 1],
                           radius/2 + trajectories[n][i, 1]])
        # Update 2D poses
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')
            # 17遍
            for j, j_parent in enumerate(parents):
                # 每个 keypoint of each frame
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    color_pink = 'pink'
                    if j % 2 == 1:
                        color_pink = 'blue'
                    # 画图2D
                    if vis_enhence:
                        lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, kp_parents[j], 0]],
                                                [keypoints[i, j, 1], keypoints[i, kp_parents[j], 1]], color=color_pink))
                        numbers.append(ax_in.text(
                            keypoints[i, j, 0], keypoints[i, j, 1], str(j), size=4))

                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    # 画图3D
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
                    if vis_enhence:
                        text = ''
                        if j == 6 or j == 3:
                            text = "%d(%.2f,%.2f,%.2f)" % (
                                j, pos[j, 0], pos[j, 1], pos[j, 2])
                        numbers.append(
                            ax.text(pos[j, 0], pos[j, 1], pos[j, 2], text, color='red'))

            # 一个 frame 的 scatter image
            points = ax_in.scatter(
                *keypoints[i].T, 8, color='red', edgecolors='white', zorder=10)
            initialized = True
        else:
            image.set_data(all_frames[i])

            # 对于 each frame
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                #  # 画图2D
                if len(parents) == keypoints.shape[1]:
                    if vis_enhence:
                        lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, kp_parents[j], 0]],
                                               [keypoints[i, j, 1], keypoints[i, kp_parents[j], 1]])
                        numbers.append(ax_in.text(
                            keypoints[i, j, 0], keypoints[i, j, 1], str(j), size=4))

                # 3D plot
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]  # one frame key points
                    lines_3d[n][j -
                                1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j -
                                1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j - 1][0].set_3d_properties(
                        [pos[j, 2], pos[j_parent, 2]], zdir='z')
                    if vis_enhence:
                        text = ''
                        if j == 10:
                            text = "%d(%2f,%2f,%2f)" % (
                                j, pos[j, 0], pos[j, 1], pos[j, 2])
                        numbers.append(
                            ax.text(pos[j, 0], pos[j, 1], pos[j, 2], text, color='red'))

            #  ax_3d.append(ax_3d[0])
            # rotate the Axes3D
            # for angle in range(0, 360):
            #     # 仰角 方位角
            #     ax_3d[0].view_init(0, 90)
            points.set_offsets(keypoints[i])

        if i % 25 == 0 and vis_enhence:
            plt.savefig(path + '/' + str(i), dpi=150, bbox_inches='tight')
        # plt.show()

        print('finish one frame\t  {}/{}      '.format(i, limit), end='\r')

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(
        0, limit), interval=1000/fps, repeat=False)

    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(path + '/' + output, writer=writer)
    elif output.endswith('.gif'):
        # anim.save(output, dpi=80, writer='imagemagick')
        anim.save(path + '/' + output, dpi=80, writer='imagemagick')
    else:
        raise ValueError(
            'Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()


def render_image(keypoints, skeleton, fps, bitrate, output, viewport, size=6, limit=-1, input_video_path=None, input_video_skip=0):
    # plt.ioff()

    figure1 = plt.figure(figsize=(size * 2, size))
    # figure1.tight_layout()
    figure1 = figure1.add_subplot(111)
    figure1.set_title('Keypoints with score')
    figure1.xaxis.set_ticks_position("top")
    # figure1.spines["left"].set_color("none")
    # figure1.spines["bottom"].set_position(('axes',0))
    # figure1.get_xaxis().set_visible(False)
    # figure1.get_yaxis().set_visible(False)
    # figure1.set_axis_off()
    # radius = 2
    # figure1.set_xlim([-1, 1])
    # figure1.set_ylim([-640/720, 640/720])
    # figure1.set_aspect('equal', 'datalim', 'S')
    # figure1.xaxis.tick_top()

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros(
            (keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    image = None
    lines = []
    points = None
    numbers = []
    # array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    parents = skeleton.parents()
    kp_parents = [-1, 0, 0, -1, -1, 0, 0, 5, 6, 7, 8, 0, 0, 11, 12, 13, 14]
    path = './data/' + output.split('.')[0]
    for i, frame in enumerate(all_frames):
        if not os.path.exists(path):
            os.mkdir(path)
        figpath = path + '/' + str(i) + '.png'

        if i == 0:
            image = figure1.imshow(frame, aspect='equal')
            for j, j_parent in enumerate(kp_parents):
                if j_parent == -1:
                    continue
                if len(parents) == keypoints.shape[1]:
                    color_pink = 'pink'
                    if j % 2 == 1:
                        color_pink = 'blue'
                lines.append(figure1.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                          [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color=color_pink))
                numbers.append(figure1.text(
                    keypoints[i, j, 0], keypoints[i, j, 1], str(j), size=4, color='red'))

            points = figure1.scatter(
                *keypoints[i].T, 8, color='red', edgecolors='white', zorder=10)
        else:
            image.set_data(frame)
            points.set_offsets(keypoints[i])
            count = 0
            for j, j_parent in enumerate(kp_parents):
                if j_parent == -1:
                    continue
                count += 1
                if len(parents) == keypoints.shape[1]:
                    lines[count-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                               [keypoints[i, j, 1], keypoints[i, j_parent, 1]])
                    numbers.append(figure1.text(
                        keypoints[i, j, 0], keypoints[i, j, 1], str(j), size=4, color='yellow'))

        if i % 20 == 0:
            plt.savefig(figpath, dpi=150, bbox_inches='tight')

        plt.show()
        for num in numbers:
            num.remove()
        numbers.clear()
        print('finish one frame\t  {}/{}      '.format(i, limit), end='\r')
