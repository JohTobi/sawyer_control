# #!/usr/bin/python3
from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from sawyer_control.envs.sawyer_insertion import SawyerHumanControlEnv
import numpy as np
import time
# env = SawyerReachXYZEnv(action_mode='position', position_action_scale = 1)
# obs = env._get_obs()
# env._act([0, 0.1, 0])
# env.reset()

#
env = SawyerHumanControlEnv(action_mode='position', position_action_scale = 1)
test = np.array([0.4, 0.1, 0.1])
# env.step(test)

# print(env._get_endeffector_pose())
# env._act([0, 0.1, 0])
# env._reset_robot()
for x in range(1):
    x = x + 1
    print("Iteration: ",x)
    time.sleep(1)
    for i in range(200):
        env.step(test)
        act_endeff_pos = env._get_endeffector_pose()[:3]
        # goal_pos = env.get_goal()
        goal_pos = np.array([0.45, 0.16, 0.0])
        dis = goal_pos - act_endeff_pos
        # print(goal_pos, act_endeff_pos, dis,np.linalg.norm(dis))
        if np.linalg.norm(dis) < 0.01:
            print("Goal reached")
            print(act_endeff_pos, env.get_goal())
            break
    time.sleep(2)
    env.reset()
#
env.reset()
print("END")
