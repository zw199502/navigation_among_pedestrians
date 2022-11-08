from math import hypot
class Robot():
    def __init__(self, radius, time_step, shape=[0.3, 0.5]):
        self.radius = radius
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.v_pref = None
        self.time_step = time_step

    def set(self, px, py, gx, gy, vx, vy, theta, v_pref=1.0):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.v_pref = v_pref

    def get_position(self):
        return self.px, self.py

    def compute_pose(self, action):
        px_new = self.px + action[0] * self.time_step 
        py_new = self.py + action[1] * self.time_step 
        theta_new = self.theta
        return px_new, py_new, theta_new

    def get_goal_distance(self):
        return hypot(self.gx - self.px, self.gy- self.py)

    def update_states(self, px, py, theta, action):
        """
        Perform an action and update the state
        """
        self.px, self.py, self.theta = px, py, theta
        self.vx = action[0] 
        self.vy = action[1]
