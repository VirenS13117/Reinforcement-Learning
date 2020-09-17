
class Action:
    def __init__(self, action):
        self.action = action
        self.action_list = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

    def get_left_perpendicular(self):
        if self.action == "up":
            return self.action_list["left"]
        elif self.action == "left":
            return self.action_list["down"]
        elif self.action == "down":
            return self.action_list["right"]
        else:
            return self.action_list["up"]

    def get_right_perpendicular(self):
        if self.action == "up":
            return self.action_list["right"]
        elif self.action == "left":
            return self.action_list["up"]
        elif self.action == "down":
            return self.action_list["left"]
        else:
            return self.action_list["down"]

    def get_opposite(self):
        if self.action == "up":
            return self.action_list["down"]
        elif self.action == "left":
            return self.action_list["right"]
        elif self.action == "down":
            return self.action_list["up"]
        else:
            return self.action_list["left"]
