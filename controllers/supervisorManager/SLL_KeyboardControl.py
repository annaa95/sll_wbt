from deepbots.supervisor.wrappers.keyboard_printer import KeyboardPrinter

class SLL_KeyboardController(KeyboardPrinter):
    def __init__(self, supervisor):
        super().__init__(supervisor)
        print("------------ Keyboard controls --------------")
        print("----- ...overwriting Keyboard Printer ... ----")

    def step(self, action, repeatSteps=10, iter_= 0):
        """
        Overriding the default KeyboardPrinter step to add custom keyboard controls for the DeepDog problem.
        """
        observation, reward, isDone, info = self.controller.step(action, repeatSteps)
        key = self.keyboard.getKey()

        if key == ord("T") and not self.controller.test:
            self.controller.test = True
            print("Training will stop and agent will be deployed after episode end.")
        if key == ord("R"):
            print("User invoked reset method.")
            self.controller.reset()

        return observation, reward, isDone, info
