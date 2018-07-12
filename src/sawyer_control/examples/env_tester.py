# #!/usr/bin/python3
from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from sawyer_control.envs.sawyer_insertion import SawyerHumanControlEnv
import numpy as np
# env = SawyerReachXYZEnv()
# env.reset()
#
env = SawyerHumanControlEnv(action_mode='position', position_action_scale = 1)
# env.reset()
for i in range(100):
    u = env.human_controller()
    env._act(u)
    act_endeff_pos = env._get_endeffector_pose()[:3]
    goal_pos = env.get_goal()
    dis = goal_pos - act_endeff_pos
    # print(act_endeff_pos)
    if np.linalg.norm(dis) < 0.1:
        print("Goal reached")
        break
# env.reset()

print("END")
