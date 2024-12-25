import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import carla_utility as utility
import physics as physics
import random as rnd
import debug_utility

class AdaptiveCruiseControlEnv(gym.Env):

    MAX_STEPS = 1024
    RADAR_RANGE = 100
    DEFAULT_RELATIVE_SPEED = 0
    SPAWN_POINT = carla.Transform(carla.Location(x=2388, y=6164, z=178), carla.Rotation(yaw = -88.2))
    VEHICLE_BP = 'vehicle.tesla.cybertruck'
    
    def __init__(self):
        super(AdaptiveCruiseControlEnv, self).__init__()
        
        rnd.seed(1234)

        self.client, self.world = utility.setup_carla_client()
        self.action_space = self.__action_space()
        self.observation_space = self.__observation_space()
        self.spectator = self.world.get_spectator()

        self._reset_environment_variables()

    def _reset_environment_variables(self):
        self.steps = 0
        self.target_speed = None
        self.ego_vehicle = None
        self.leader_vehicle = None
        self.radar_sensor = None
        self.radar_data = {}
    

    def __action_space(self):
        #[brake, throttle]
        return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    def __observation_space(self):
        #[ego_speed, target_speed, distance, relative_speed, security_distance] distance in m and speed in m/s
        return spaces.Box(low=np.array([0.0, 25.0, 0.0, -100.0, 0.0], dtype=np.float32), high=np.array([100.0, 40.0, 100, 100.0, 72.0], dtype=np.float32), shape=(5,), dtype=np.float32)
    
    def set_target_speed(self, speed):
        self.target_speed = speed
    
    def set_leader_speed(self, speed):
        self.leader_vehicle.set_target_velocity(debug_utility.get_velocity_vector(speed, self.SPAWN_POINT.rotation))
    
    def spawn_vehicle(self):
        #Spawn ego vehicle
        random_offset = rnd.randint(50, 100)
        self.ego_vehicle = utility.spawn_vehicle_bp_at(world=self.world, vehicle=self.VEHICLE_BP, spawn_point=self.SPAWN_POINT)
        self.leader_vehicle = utility.spawn_vehicle_bp_in_front_of(world=self.world, vehicle=self.ego_vehicle, vehicle_bp_name=self.VEHICLE_BP, offset=random_offset)
        random_ego_velocity = physics.kmh_to_ms(rnd.randint(90, 110)) # In m/s
        random_leader_velocity = physics.kmh_to_ms(rnd.randint(50, 70))
        self.ego_vehicle.set_target_velocity(debug_utility.get_velocity_vector(random_ego_velocity, self.SPAWN_POINT.rotation))
        self.leader_vehicle.set_target_velocity(debug_utility.get_velocity_vector(random_leader_velocity, self.SPAWN_POINT.rotation))
        self.set_target_speed(random_ego_velocity)

        #Spawn radar sensor
        self._spawn_radar()

        #Move spectator to scene
        utility.move_spectator_to(self.spectator, self.ego_vehicle.get_transform())
    
    def _spawn_radar(self):
        self.radar_sensor = utility.spawn_radar(self.world, self.ego_vehicle, range=200, horizontal_fov=50, vertical_fov=50)
        self.radar_sensor.listen(self._radar_callback)
    
    def _radar_callback(self, radar_data):
        distances = []
        relative_speeds = []

        for detection in radar_data:
            if debug_utility.evaluate_point(self.radar_sensor, detection, 1, 0.8):
                debug_utility.draw_radar_point(self.radar_sensor, detection)
                distances.append(detection.depth) #Distance to detect object
                relative_speeds.append(detection.velocity) #Velocity of detect object
        
        # Use the closest detected object as the relevant one for control
        if len(distances) > 0:
            closest_distance = min(distances)
            closest_relative_speed = relative_speeds[distances.index(closest_distance)]
            self.radar_data["distance"] = closest_distance
            self.radar_data["relative_speed"] = closest_relative_speed
        else:
            self.radar_data["distance"] = self.RADAR_RANGE
            self.radar_data["relative_speed"] = self.DEFAULT_RELATIVE_SPEED

    def __get_observation(self):
        ego_speed = self.ego_vehicle.get_velocity().length()
        target_speed = self.target_speed
        distance = self.radar_data["distance"] if self.radar_data["distance"] <= self.RADAR_RANGE else self.RADAR_RANGE
        relative_speed = self.radar_data["relative_speed"]
        security_distance = physics.compute_security_distance(physics.ms_to_kmh(ego_speed)) if (distance < self.RADAR_RANGE) else 0.0
        return np.array([ego_speed, target_speed, distance, relative_speed, security_distance], dtype=np.float32)
    
    def __compute_reward(self, observation):
        current_speed, desired_speed, distance, relative_speed, security_distance = observation
        speed_tolerance = 0.1  # 1% tolerance around the desired speed
        security_distance_tolerance = 0.3 # 3% tolerance around the security distance
        isEgoMaintainSpeed = desired_speed - speed_tolerance <= current_speed <= desired_speed + speed_tolerance
        isObjectDetected = distance < self.RADAR_RANGE
        isEgoMaintainSecurityDistance = security_distance - security_distance_tolerance <= distance <= security_distance + security_distance_tolerance
        reward = 0

        if not isObjectDetected:
            #Penalty for not maintaining speed
            if not isEgoMaintainSpeed:
                reward -= abs(desired_speed - current_speed) / max(desired_speed, 0.1)
            else:
                reward += 10.0 - abs(desired_speed - current_speed)
        else:
            #Penalty for security distance
            if not isEgoMaintainSecurityDistance:
                reward -= abs(distance - security_distance) / max(security_distance, 0.1)
            else:
                reward += 10.0 - abs(distance - security_distance)
            
            #Collision detection
            if distance <= 5:
                reward -= 10.0

        return reward
    
    def __compute_done(self, observation):
        current_speed, _, distance, _, _ = observation
        isEgoStopped = current_speed <= 0.1
        haveEgoReachedMaxSpeed = current_speed >= 37.0
        haveEgoReachedMaxSteps = self.steps >= self.MAX_STEPS
        isInCollisionCase = distance <= 5.0

        if isEgoStopped:
            print("Reset: Ego stopped")
        
        if haveEgoReachedMaxSpeed:
            print("Reset: Ego reached max speed")

        if haveEgoReachedMaxSteps:
            print("Reset: Max steps reached")
        
        if isInCollisionCase:
            print("Reset: Collision detected")
        

        return isEgoStopped or  haveEgoReachedMaxSteps or haveEgoReachedMaxSpeed or isInCollisionCase #isEgoStopped or 
    
    def reset(self, *, seed = None, options = None):
        print("----RESET----")
        utility.destroy_all_vehicle_and_sensors(self.world)
        self._reset_environment_variables()
        self.spawn_vehicle()
        self.world.tick()

        return self.__get_observation(), {}

    def step(self, action):
        is_car_accelerating = action[0] > 0.0
        action = abs(np.clip(action[0], -1.0, 1.0))
        if is_car_accelerating:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=action, brake=0.0))
        else:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=action))
        
        self.steps += 1

        self.world.tick()

        observation = self.__get_observation()
        reward = self.__compute_reward(observation)
        done = bool(self.__compute_done(observation))
        truncated = False
        info = {}
        
        print(f"Reward: {reward}, Observations_ego_speed: {physics.ms_to_kmh(observation[0])} km/h, Observations_target_speed: {physics.ms_to_kmh(observation[1])} km/h, Leader speed: {physics.ms_to_kmh(self.leader_vehicle.get_velocity().length())} km/h, Distance: {observation[2]} m, Detected: {observation[2] < self.RADAR_RANGE}, Relative Speed: {physics.ms_to_kmh(observation[3])} km/h, Security Distance: {observation[4]} km/h")
        return observation, reward, done, truncated, info