#!/usr/bin/env python3
"""
PN guidance + kill logger for one Monte Carlo run.
Kill criteria (both recorded, primary = contact):
  1. CONTACT: any interceptor collision (flail arms or body) against target,
     validated by a proximity guard against stale/cross-run contamination.
  2. MIN-DISTANCE (cross-check): closest approach < effective kill radius.
Usage: python3 guidance_node.py <flail|ram> <error_sigma> <seed> <outfile>
"""
import json
import math
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from ros_gz_interfaces.msg import Contacts

SPIN_RATE_RADS = 838.0       # ~8000 RPM
FLAIL_KILL_RADIUS = 0.80     # arm reach 0.55 + target extent ~0.25 (center-to-center)
RAM_KILL_RADIUS = 0.35       # body 0.10 + target extent ~0.25
CONTACT_GUARD_DIST = 2.0     # reject contact events when vehicles are farther apart
NAV_CONSTANT = 4.0
MAX_SPEED = 60.0
MAX_ACCEL = 300.0
TIMEOUT_S = 12.0
TARGET_SPEED = 8.0


class GuidanceNode(Node):
    def __init__(self, interceptor_type, error_sigma, seed, outfile):
        super().__init__('guidance_node')
        self.interceptor_type = interceptor_type  # 'flail' | 'ram'
        self.error_sigma = error_sigma
        self.rng = np.random.default_rng(seed)
        self.outfile = outfile

        # Per-run fixed guidance bias: N(0, sigma) per axis (independent variable)
        self.guidance_bias = self.rng.normal(0.0, error_sigma, size=3)
        hdg = math.pi + self.rng.uniform(-0.1745, 0.1745)
        self.target_vel = np.array([TARGET_SPEED * math.cos(hdg),
                                    TARGET_SPEED * math.sin(hdg), 0.0])

        self.int_pos = None
        self.int_vel = np.zeros(3)
        self.tgt_pos = None
        self.cmd_vel = np.zeros(3)
        self.min_dist = float('inf')
        self.min_dist_t = None
        self.contact_kill = False
        self.contact_detail = None
        self.t0 = None
        self.prev_los = None
        self.prev_t = None
        self.done = False

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, '/model/dwarf_interceptor/odometry',
                                 self.on_int_odom, qos)
        self.create_subscription(Odometry, '/model/target_quadcopter/odometry',
                                 self.on_tgt_odom, qos)
        # arm sensors (flail model only; topics simply silent in ram runs)
        for i in range(1, 5):
            self.create_subscription(
                Contacts, f'/dwarf_interceptor/arm_{i}/contact',
                lambda m, src=f'arm_{i}': self.on_contact(m, src), qos)
        # body sensor (both models)
        self.create_subscription(
            Contacts, '/dwarf_interceptor/body/contact',
            lambda m: self.on_contact(m, 'body'), qos)

        self.pub_int_cmd = self.create_publisher(Twist, '/model/dwarf_interceptor/cmd_vel', 10)
        self.pub_tgt_cmd = self.create_publisher(Twist, '/model/target_quadcopter/cmd_vel', 10)
        self.pub_spin = self.create_publisher(
            Float64, '/model/dwarf_interceptor/joint/spin_joint/cmd_vel', 10)

        self.timer = self.create_timer(0.005, self.tick)  # 200 Hz guidance

    def on_int_odom(self, msg):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.int_pos = np.array([p.x, p.y, p.z])
        self.int_vel = np.array([v.x, v.y, v.z])

    def on_tgt_odom(self, msg):
        p = msg.pose.pose.position
        self.tgt_pos = np.array([p.x, p.y, p.z])

    def on_contact(self, msg, source):
        # guard: reject contacts before both poses known, or when vehicles
        # are provably far apart (stale / cross-run contamination)
        if self.int_pos is None or self.tgt_pos is None:
            return
        if float(np.linalg.norm(self.tgt_pos - self.int_pos)) > CONTACT_GUARD_DIST:
            return
        for c in msg.contacts:
            names = (c.collision1.name + ' ' + c.collision2.name)
            if 'target_quadcopter' in names:
                if not self.contact_kill:
                    self.contact_kill = True
                    self.contact_detail = {'source': source, 'collisions': names,
                                           't': self.elapsed()}

    def elapsed(self):
        if self.t0 is None:
            return 0.0
        return (self.get_clock().now() - self.t0).nanoseconds * 1e-9

    def tick(self):
        if self.done or self.int_pos is None or self.tgt_pos is None:
            return
        now = self.get_clock().now()
        if self.t0 is None:
            self.t0 = now
        t = self.elapsed()

        # target flies its path; spin only in flail mode
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.linear.z = self.target_vel
        self.pub_tgt_cmd.publish(tw)
        spin = Float64()
        spin.data = SPIN_RATE_RADS if self.interceptor_type == 'flail' else 0.0
        self.pub_spin.publish(spin)

        # ---- PN guidance on the *corrupted* target estimate ----
        tgt_est = self.tgt_pos + self.guidance_bias
        r = tgt_est - self.int_pos
        dist_true = float(np.linalg.norm(self.tgt_pos - self.int_pos))
        if dist_true < self.min_dist:
            self.min_dist, self.min_dist_t = dist_true, t

        rmag = np.linalg.norm(r)
        los = r / max(rmag, 1e-6)
        tsec = now.nanoseconds * 1e-9
        if self.prev_los is not None and tsec > self.prev_t:
            dt = tsec - self.prev_t
            los_rate = (los - self.prev_los) / dt
            v_rel = np.dot(self.target_vel - self.int_vel, los)
            a_c = NAV_CONSTANT * abs(v_rel) * los_rate
            amag = np.linalg.norm(a_c)
            if amag > MAX_ACCEL:
                a_c *= MAX_ACCEL / amag
            self.cmd_vel = self.cmd_vel + a_c * dt
            closing = self.cmd_vel + los * 5.0
            speed = np.linalg.norm(closing)
            if speed > MAX_SPEED:
                closing *= MAX_SPEED / speed
            cmd = Twist()
            cmd.linear.x, cmd.linear.y, cmd.linear.z = closing
            self.pub_int_cmd.publish(cmd)
        self.prev_los, self.prev_t = los, tsec

        # ---- termination ----
        kill_radius = FLAIL_KILL_RADIUS if self.interceptor_type == 'flail' else RAM_KILL_RADIUS
        past = (self.min_dist_t is not None and t - self.min_dist_t > 2.0
                and dist_true > self.min_dist + 5.0)
        if self.contact_kill or t > TIMEOUT_S or past:
            self.finish(kill_radius, t)

    def finish(self, kill_radius, t):
        self.done = True
        result = {
            'interceptor_type': self.interceptor_type,
            'error_sigma': self.error_sigma,
            'guidance_bias': self.guidance_bias.tolist(),
            'kill_contact': self.contact_kill,
            'contact_detail': self.contact_detail,
            'min_miss_distance': self.min_dist if self.min_dist != float('inf') else None,
            'time_closest_approach': self.min_dist_t,
            'kill_geometric': self.min_dist < kill_radius,
            'duration': t,
        }
        with open(self.outfile, 'w') as f:
            json.dump(result, f, indent=2)
        self.get_logger().info(f"RESULT: {result}")
        raise SystemExit(0)


def main():
    interceptor_type, error_sigma, seed, outfile = (
        sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    rclpy.init()
    node = GuidanceNode(interceptor_type, error_sigma, seed, outfile)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()