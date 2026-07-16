#!/usr/bin/env python3
"""
Monte Carlo campaign runner. ROS 2 Jazzy + Gazebo Harmonic.
Flail and ram use separate worlds (true ram model, mass-matched).
Usage: python3 run_campaign.py --runs 200
"""
import argparse
import os
import pathlib
import signal
import subprocess
import time

BASE = pathlib.Path(__file__).resolve().parents[1]
WORLD_FLAIL = str(BASE / 'worlds' / 'cuas_world.sdf')
WORLD_RAM = str(BASE / 'worlds' / 'cuas_world_ram.sdf')
RESULTS = pathlib.Path(__file__).resolve().parent / 'results'

# Widened to bracket both kill radii (ram ~0.35 m, flail ~0.80 m effective):
ERROR_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]   # m (sigma per axis)
TYPES = ['flail', 'ram']
STARTUP_SLEEP = 8.0

BRIDGE_ARGS = [
    '/model/dwarf_interceptor/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
    '/model/target_quadcopter/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
    '/model/dwarf_interceptor/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
    '/model/target_quadcopter/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
    '/model/dwarf_interceptor/joint/spin_joint/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double',
    '/dwarf_interceptor/body/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
] + [f'/dwarf_interceptor/arm_{i}/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts'
     for i in range(1, 5)]


def one_run(itype, sigma, seed, outfile, run_uid):
    env = os.environ.copy()
    env['GZ_SIM_RESOURCE_PATH'] = str(BASE / 'models')
    # HARD ISOLATION: unique transport namespaces per run.
    env['GZ_PARTITION'] = f'mc_{run_uid}'
    env['ROS_DOMAIN_ID'] = str(1 + (run_uid % 100))

    world = WORLD_FLAIL if itype == 'flail' else WORLD_RAM
    gz = subprocess.Popen(
        ['gz', 'sim', '-s', '-r', world],
        env=env, preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bridge = subprocess.Popen(
        ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge'] + BRIDGE_ARGS,
        env=env, preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(STARTUP_SLEEP)
    try:
        subprocess.run(
            ['python3', str(pathlib.Path(__file__).parent / 'guidance_node.py'),
             itype, str(sigma), str(seed), str(outfile)],
            env=env, timeout=120)
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT (counted as miss): {outfile.name}')
        outfile.write_text(
            '{"interceptor_type":"%s","error_sigma":%f,'
            '"kill_contact":false,"kill_geometric":false,'
            '"min_miss_distance":null,"timeout":true}' % (itype, sigma))
    finally:
        for p in (bridge, gz):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                continue
        time.sleep(2.0)
        for p in (bridge, gz):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=100)
    ap.add_argument('--seed-base', type=int, default=42)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    total = len(TYPES) * len(ERROR_LEVELS) * args.runs
    n = 0
    for t_idx, itype in enumerate(TYPES):
        for s_idx, sigma in enumerate(ERROR_LEVELS):
            for i in range(args.runs):
                n += 1
                out = RESULTS / f'{itype}_s{int(sigma * 100):03d}_r{i:04d}.json'
                if out.exists():
                    continue  # resumable
                seed = args.seed_base + t_idx * 1_000_000 + s_idx * 10_000 + i
                print(f'[{n}/{total}] {itype} sigma={sigma} run={i}')
                one_run(itype, sigma, seed, out, run_uid=n)


if __name__ == '__main__':
    main()