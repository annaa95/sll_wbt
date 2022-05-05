from deepbots.supervisor.wrappers.tensorboard_wrapper import TensorboardLogger
from tensorboardX import SummaryWriter
#from utilities import normalizeToRange
import numpy as np

class SLL_TensorboardLogger(TensorboardLogger):
    def __init__(self, supervisor,
                 log_dir="logs/results/ddpg",
                 v_action=0,
                 v_observation=0,
                 v_reward=0,
                 windows=[10, 100, 200]):
        super().__init__(supervisor)
        
        print("------------ Tensorboard controls --------------")
        print("----- ...overwriting Tensorboard Logger ... ----")
        self.controller = supervisor

        self.step_cntr = 0
        self.step_global = 0
        self.step_reset = 0

        self.score = 0
        self.score_history = []

        self.v_action = v_action
        self.v_observation = v_observation
        self.v_reward = v_reward
        self.windows = windows
        self.file_writer = SummaryWriter(log_dir, flush_secs=30)



    def step(self, action, repeatSteps=10, iter_=0):
        """
        Overriding the default Tensorboard Logger step to add custom step function for the DeepDog problem.
        """
        observation, reward, isDone, info = self.controller.step(action, repeatSteps)

        if (self.v_action > 1):
            """
            for readability converted into their different physical meanings
            """

            #control variables : normalized to range
            """
            alphas = [normalizeToRange(action[i],-1.0, 1.0 ,70, 110 ) for i in range(4)]
            """

            # to log data to the Tensorboard Logger we use file_writer.add_scalars
            """
            self.file_writer.add_scalars(
                "Alphas/Per Global Step", {'alpha1':alphas[0],
                                            'alpha2':alphas[1],
                                            'alpha3':alphas[2],
                                            'alpha4':alphas[3]},
                global_step=self.step_global)
            """

        if (self.v_observation > 1):
            # Alternatively, we can log scalrs or histograms
            """
            acc = [normalizeToRange(observation[i],  -1.0, 1.0, -200.0, 200.0, clip = True) for i in range(3)]
            self.file_writer.add_scalars(
                "Acceleration/Per Global Step", {'ax':acc[0],
                                            'ay':acc[1],
                                            'az':acc[2]},
                global_step=self.step_global)
            self.file_writer.add_histogram(
                "Observations/Per Global Step",
                observation,
                global_step=self.step_global)
            """
        if (self.v_reward > 1):
            self.file_writer.add_scalar("Rewards/Per Global Step", reward,
                                        self.step_global)

        if (isDone):
            self.file_writer.add_scalar(
                "Is Done/Per Reset step",
                self.step_cntr,
                global_step=self.step_reset)

        self.file_writer.flush()

        self.score += reward

        self.step_cntr += 1
        self.step_global += 1

        return observation, reward, isDone, info

    def is_done(self):
        isDone = self.controller.is_done()

        self.file_writer.flush()
        return isDone

    def get_observations(self):
        obs = self.controller.get_observations()

        return obs

    def get_reward(self, action):
        reward = self.controller.get_reward(action)
        return reward

    def get_info(self):
        info = self.controller.get_info()
        return info

    def reset(self):

        observations = self.controller.reset()
        self.score_history.append(self.score)

        if (self.v_observation > 0):
            self.file_writer.add_histogram(
                "Observations/Per Reset",
                observations,
                global_step=self.step_reset)

        if (self.v_reward > 0):
            self.file_writer.add_scalar(
                "Score/Per Reset", self.score, global_step=self.step_reset)

            for window in self.windows:
                if self.step_reset > window:
                    self.file_writer.add_scalar(
                        "Score/With Window {}".format(window),
                        np.average(self.score_history[-window:]),
                        global_step=self.step_reset - window)

        self.file_writer.flush()

        self.step_reset += 1
        self.step_cntr = 0
        self.score = 0

        return observations

    def flush(self):
        if self._file_writer is not None:
            self._file_writer.flush()

    def close(self):
        if self._file_writer is not None:
            self._file_writer.close()
