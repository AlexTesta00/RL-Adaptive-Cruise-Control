import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import carla
import carla_utility as utility
import physics as physics

class CruiseControlEnv(gym.Env):

    MAX_STEPS = 2048
    TARGET_SPEED = physics.kmh_to_ms(110)
    SPAWN_POINT = carla.Location(x=2393, y=6000, z=167)
    SPAWN_YAW = -90.0
    VEHICLE_BP = 'vehicle.tesla.model3'
    
    def __init__(self):
        super(CruiseControlEnv, self).__init__()

        self.client, self.world = utility.setup_carla_client()
        self.action_space = self.__action_space()
        self.observation_space = self.__observation_space()
        self.spectator = self.world.get_spectator()

        self.steps = 0

        self.ego_vehicle = None
    

    def __action_space(self):
        return spaces.Box(low=-1.0, high=1.0, dtype=np.float32)
    
    def __observation_space(self):
        return spaces.Box(low=0.0, high=1.0, dtype=np.float32)
    
    def spawn_vehicle(self):
        #Spawn ego vehicle
        spawn_point = carla.Transform(self.SPAWN_POINT, carla.Rotation(yaw = self.SPAWN_YAW))
        self.ego_vehicle = utility.spawn_vehicle_bp_at(world=self.world, vehicle=self.VEHICLE_BP, spawn_point=spawn_point)
        self.ego_vehicle.set_target_velocity(carla.Vector3D(self.TARGET_SPEED*math.cos(self.SPAWN_YAW), self.TARGET_SPEED*math.sin(self.SPAWN_YAW), 0))

        #Move spectator to scene
        utility.move_spectator_to(self.spectator, self.ego_vehicle.get_transform())

    def __get_observation(self):
        return np.array([physics.map_sigmoid(self.ego_vehicle.get_velocity().length())], dtype=np.float32)
    
    def __compute_reward(self, action):
        current_speed = self.ego_vehicle.get_velocity().length()
        desired_speed = self.TARGET_SPEED
        reward = 0

        if current_speed != desired_speed:
            reward -= (desired_speed - current_speed) / desired_speed
        else:
            reward += 1

        return reward
    
    def __compute_done(self, action):
        current_speed = self.ego_vehicle.get_velocity().length()

        isEgoStopped = current_speed <= 0.1
        haveEgoReachedMaxSteps = self.steps >= self.MAX_STEPS

        if isEgoStopped:
            print("Reset: Ego stopped")
        
        if haveEgoReachedMaxSteps:
            print("Reset: Ego reached max steps")

        return isEgoStopped or haveEgoReachedMaxSteps
    
    def reset(self, *, seed = None, options = None):
        print("----RESET----")
        utility.destroy_all_vehicle_and_sensors(self.world)
        self.ego_vehicle = None
        self.spawn_vehicle()
        self.world.tick()

        self.steps = 0

        return self.__get_observation(), {}

    def step(self, action):
        is_car_accelerating = action[0] > 0.0
        action = abs(np.clip(action[0], -1.0, 1.0))
        if is_car_accelerating:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=action, brake=0.0))
            print(f"Throttle: {action}, Brake: 0.0")
        else:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=action))
            print(f"Throttle: 0.0, Brake: {action}")
        
        self.world.tick()

        self.steps += 1

        observation = self.__get_observation()
        reward = self.__compute_reward(action)
        done = self.__compute_done(action)
        truncated = False
        info = {}


        print(f"Ego_Velocity: {self.ego_vehicle.get_velocity().length()}, Observations: {observation}, Reward: {reward}, Target: {self.TARGET_SPEED}")

        return observation, reward, done, truncated, info