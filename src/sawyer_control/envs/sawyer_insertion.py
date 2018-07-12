from collections import OrderedDict
import numpy as np
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable
from sawyer_control.core.eval_util import get_stat_in_paths, \
    create_stats_ordered_dict
from sawyer_control.core.multitask_env import MultitaskEnv
from sawyer_control.configs import base_config

class SawyerHumanControlEnv(SawyerEnvBase, MultitaskEnv):
    def __init__(self,
                 fixed_goal=(0.45, 0.158, -0.2),
                 reward_type='hand_distance',
                 config = base_config,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(self)
        SawyerEnvBase.__init__(self, **kwargs)
        if self.action_mode=='torque':
            self.goal_space = self.config.TORQUE_SAFETY_BOX
        else:
            self.goal_space = self.config.POSITION_SAFETY_BOX
        # self.indicator_threshold=indicator_threshold
        self.reward_type = reward_type
        self._state_goal = np.array(fixed_goal)
        self.reset_position = np.array([0.45, 0.158, 0.21])

    @property
    def goal_dim(self):
        return 3

    def _get_obs(self):
        angles, velocities, endpoint_pose = self.request_observation()
        obs = np.hstack((
            self._wrap_angles(angles), #0-6
            velocities, #7-13
            endpoint_pose, #14-20
        ))
        return obs

    def compute_rewards(self, actions, obs, goals):
        distances = np.linalg.norm(obs - goals, axis=1)
        if self.reward_type == 'hand_distance':
            r = -distances
        elif self.reward_type == 'hand_success':
            r = -(distances < self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def human_controller(self):
        goal_pose = np.array([0.53074485, 0.39027235, 0.10360944, 0.71906072, 0.69293875, 0.03858845,0.03603474])
        # reset_pose = np.array([0.4478398, 0.15977567, 0.21573293, 0.70707673, 0.70713544, -0.00071061, 0.0012021])

        # err = self.get_goal() - self._get_endeffector_pose()[:3]
        err = goal_pose - self._get_endeffector_pose()
        kp = 1.0
        u = kp * err
        return u[:3]

    # def reset(self):
    #     self.in_reset = True
    #     self._safe_move_to_neutral()
    #     self.in_reset = False
    #     observation = self._get_obs()
    #     return observation
    #
    # def _safe_move_to_neutral(self):
    #     for i in range(self.config.RESET_LENGTH):
    #         # cur_pos, cur_vel, _ = self.request_observation()
    #         # torques = self.AnglePDController._compute_pd_forces(cur_pos, cur_vel)
    #         self._position_act(self.reset_position - self._get_endeffector_pose()[:3])
    #         # self._torque_act(torques)
    #         if self._reset_complete():
    #             break
    #
    # def _reset_complete(self):
    #     close_to_desired_reset_pos = self._check_reset_angles_within_threshold()
    #     _, velocities, _ = self.request_observation()
    #     velocities = np.abs(np.array(velocities))
    #     VELOCITY_THRESHOLD = .09 * np.ones(7)
    #     no_velocity = (velocities < VELOCITY_THRESHOLD).all()
    #     return close_to_desired_reset_pos and no_velocity
    #
    # def _check_reset_angles_within_threshold(self):
    #     desired_neutral = self.AnglePDController._des_angles
    #     desired_neutral = np.array([desired_neutral[joint] for joint in self.config.JOINT_NAMES])
    #     actual_neutral = (self._get_joint_angles())
    #     errors = self.compute_angle_difference(desired_neutral, actual_neutral)
    #     is_within_threshold = (errors < self.config.RESET_ERROR_THRESHOLD).all()
    #     return is_within_threshold #Output True or False

    # def _get_info(self):
    #     hand_distance = np.linalg.norm(self._state_goal - self._get_endeffector_pose())
    #     return dict(
    #         hand_distance=hand_distance,
    #         hand_success=(hand_distance<self.indicator_threshold).astype(float)
    #     )


    #
    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'hand_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics
    #
    def set_to_goal(self, goal):
        raise NotImplementedError()
