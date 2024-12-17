import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import carla
import carla_utility as utility
import physics as physics
import random as rnd
import debug_utility

class CruiseControlEnv(gym.Env):

    MAX_STEPS = 1024
    SPAWN_POINT = carla.Transform(carla.Location(x=2388, y=6164, z=178), carla.Rotation(yaw = -88.2))
    VEHICLE_BP = 'vehicle.tesla.model3'
    
    def __init__(self):
        super(CruiseControlEnv, self).__init__()
        
        rnd.seed(42)

        self.client, self.world = utility.setup_carla_client()
        self.action_space = self.__action_space()
        self.observation_space = self.__observation_space()
        self.spectator = self.world.get_spectator()

        self.steps = 0
        self.target_speed = None
        self.ego_vehicle = None
    

    def __action_space(self):
        return spaces.Box(low=-1.0, high=1.0, dtype=np.float32)
    
    def __observation_space(self):
        return spaces.Box(low=np.array([0.0, 25.0], dtype=np.float32), high=np.array([100.0, 40.0], dtype=np.float32), dtype=np.float32) #[ego_speed, target_speed] in m/s
    
    def set_target_speed(self, speed):
        self.target_speed = speed
    
    def spawn_vehicle(self):
        #Spawn ego vehicle
        self.ego_vehicle = utility.spawn_vehicle_bp_at(world=self.world, vehicle=self.VEHICLE_BP, spawn_point=self.SPAWN_POINT)
        random_ego_velocity = physics.kmh_to_ms(rnd.randint(90, 130)) # In m/s
        self.set_target_speed(random_ego_velocity)
        self.ego_vehicle.set_target_velocity(debug_utility.get_velocity_vector(random_ego_velocity, self.SPAWN_POINT.rotation))

        #Move spectator to scene
        utility.move_spectator_to(self.spectator, self.ego_vehicle.get_transform())

    def __get_observation(self):
        return np.array([self.ego_vehicle.get_velocity().length(), self.target_speed], dtype=np.float32)
    
    def __compute_reward(self, action):
        current_speed = self.ego_vehicle.get_velocity().length()
        desired_speed = self.target_speed
        tolerance = 0.5  # 5% tolerance around the desired speed
        reward = 0

        if not desired_speed - tolerance <= current_speed <= desired_speed + tolerance:
            reward -= abs(desired_speed - current_speed) / desired_speed
        else:
            reward += 1

        return reward
    
    def __compute_done(self, action):
        current_speed = self.ego_vehicle.get_velocity().length()

        isEgoStopped = current_speed <= 0.1
        haveEgoReachedMaxSteps = self.steps >= self.MAX_STEPS
        haveEgoReachedMaxSpeed = current_speed >= 37.0

        if isEgoStopped:
            print("Reset: Ego stopped")
        
        if haveEgoReachedMaxSteps:
            print("Reset: Ego reached max steps")
        
        if haveEgoReachedMaxSpeed:
            print("Reset: Ego reached max speed")

        return isEgoStopped or haveEgoReachedMaxSteps or haveEgoReachedMaxSpeed
    
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
            #print(f"Throttle: {action}, Brake: 0.0")
        else:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=action))
            #print(f"Throttle: 0.0, Brake: {action}")
        
        self.world.tick()

        self.steps += 1

        observation = self.__get_observation()
        reward = self.__compute_reward(action)
        done = self.__compute_done(action)
        truncated = False
        info = {}

        print(f"Ego_Velocity: {self.ego_vehicle.get_velocity().length() * 3.6} km/h, Target: {self.target_speed * 3.6} km/h, Reward: {reward}, Observations_ego_speed: {observation[0] * 3.6}, Observations_target_speed: {observation[1] * 3.6}")

        return observation, reward, done, truncated, info