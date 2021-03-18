# Parametric Parkour continuous environment
# Initially created by Clément Romac.

#region Imports

import math
import os

import Box2D
import gym
import numpy as np
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef)
from gym import spaces
from gym.utils import seeding, EzPickle

from TeachMyAgent.environments.envs.Box2D_dynamics.water_dynamics import WaterDynamics, WaterContactDetector
from TeachMyAgent.environments.envs.Box2D_dynamics.climbing_dynamics import ClimbingDynamics, ClimbingContactDetector
from TeachMyAgent.environments.envs.PCGAgents.CPPN.TanHSoftplusMixCPPN import TanHSoftplusMixCPPN
from TeachMyAgent.environments.envs.bodies.BodiesEnum import BodiesEnum
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomUserDataObjectTypes, CustomUserData


#endregion

#region Utils
class ContactDetector(WaterContactDetector, ClimbingContactDetector):
    def __init__(self, env):
        super(ContactDetector, self).__init__()
        self.env = env
    def BeginContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
        if any([body.userData.object_type == CustomUserDataObjectTypes.WATER for body in bodies]):
            WaterContactDetector.BeginContact(self, contact)
        elif any([body.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR for body in bodies]):
            ClimbingContactDetector.BeginContact(self, contact)
        else:
            if contact.fixtureA.sensor or contact.fixtureB.sensor:
                return
            for idx, body in enumerate(bodies):
                if body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT and body.userData.check_contact:
                    body.userData.has_contact = True
                    other_body = bodies[(idx + 1) % 2]
                    # Authorize climbing bodies to touch climbing parts
                    if body.userData.is_contact_critical and \
                            not (other_body.userData.object_type == CustomUserDataObjectTypes.GRIP_TERRAIN and
                                         self.env.agent_body.body_type == BodyTypesEnum.CLIMBER):
                        self.env.critical_contact = True

    def EndContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
        if any([body.userData.object_type == CustomUserDataObjectTypes.WATER for body in bodies]):
            WaterContactDetector.EndContact(self, contact)
        elif any([body.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR for body in bodies]):
            ClimbingContactDetector.EndContact(self, contact)
        else:
            for body in [contact.fixtureA.body, contact.fixtureB.body]:
                if body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT and body.userData.check_contact:
                    body.userData.has_contact = False

    def Reset(self):
        WaterContactDetector.Reset(self)
        ClimbingContactDetector.Reset(self)


class LidarCallback(Box2D.b2.rayCastCallback):
    def __init__(self, agent_mask_filter):
        Box2D.b2.rayCastCallback.__init__(self)
        self.agent_mask_filter = agent_mask_filter
        self.fixture = None
        self.is_water_detected = False
        self.is_creeper_detected = False
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & self.agent_mask_filter) == 0:
            return -1
        self.p2 = point
        self.fraction = fraction
        self.is_water_detected = True if fixture.body.userData.object_type == CustomUserDataObjectTypes.WATER else False
        self.is_creeper_detected = True if fixture.body.userData.object_type == CustomUserDataObjectTypes.SENSOR_GRIP_TERRAIN else False
        return fraction
#endregion

#region Constants

FPS    = 50
SCALE  =  30.0   # affects how fast-paced the game is, forces should be adjusted as well
VIEWPORT_W = 600
VIEWPORT_H = 400

RENDERING_VIEWER_W = VIEWPORT_W
RENDERING_VIEWER_H = VIEWPORT_H

NB_LIDAR = 10
LIDAR_RANGE   = 160/SCALE

INITIAL_RANDOM = 5

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_END    = 5 # in steps
INITIAL_TERRAIN_STARTPAD = 20 # in steps
FRICTION = 2.5
WATER_DENSITY = 1.0
NB_FIRST_STEPS_HANG = 5

#endregion

class ParametricContinuousParkour(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, agent_body_type, CPPN_weights_path = None, input_CPPN_dim=3, terrain_cppn_scale=10,
                 ceiling_offset = 200, ceiling_clip_offset = 0, lidars_type = 'full', water_clip = 20, movable_creepers=False,
                 **walker_args):
        super(ParametricContinuousParkour, self).__init__()

        # Use 'down' for walkers, 'up' for climbers and 'full' for swimmers.
        if lidars_type == "down":
            self.lidar_angle = 1.5
            self.lidar_y_offset = 0
        elif lidars_type == "up":
            self.lidar_angle = 2.3
            self.lidar_y_offset = 1.5
        elif lidars_type == "full":
            self.lidar_angle = np.pi
            self.lidar_y_offset = 0

        # Seed env and init Box2D
        self.seed()
        self.viewer = None
        self.contact_listener = ContactDetector(self)
        self.world = Box2D.b2World(contactListener=self.contact_listener)
        self.movable_creepers = movable_creepers

        # Create agent
        body_type = BodiesEnum.get_body_type(agent_body_type)
        if body_type == BodyTypesEnum.SWIMMER or body_type == BodyTypesEnum.AMPHIBIAN:
            self.agent_body = BodiesEnum[agent_body_type].value(SCALE, density=WATER_DENSITY, **walker_args)
        elif body_type == BodyTypesEnum.WALKER:
            self.agent_body = BodiesEnum[agent_body_type].value(SCALE, **walker_args,
                                                                reset_on_hull_critical_contact=False)
        else:
            self.agent_body = BodiesEnum[agent_body_type].value(SCALE, **walker_args)

        # Terrain and  dynamics
        self.terrain = []
        self.water_dynamics = WaterDynamics(self.world.gravity, max_push=water_clip)
        self.climbing_dynamics = ClimbingDynamics()
        self.prev_shaping = None
        self.episodic_reward = 0

        self.TERRAIN_STARTPAD = INITIAL_TERRAIN_STARTPAD if \
            self.agent_body.AGENT_WIDTH / TERRAIN_STEP + 5 <= INITIAL_TERRAIN_STARTPAD else \
            self.agent_body.AGENT_WIDTH / TERRAIN_STEP + 5  # in steps
        self.create_terrain_fixtures()

        self.input_CPPN_dim = input_CPPN_dim
        if CPPN_weights_path is None:
            current_path = os.path.dirname(os.path.realpath(__file__))
            CPPN_weights_path = os.path.join(current_path, "PCGAgents/CPPN/weights/same_ground_ceiling_cppn/")

        self.terrain_CPPN = TanHSoftplusMixCPPN(x_dim=TERRAIN_LENGTH,
                                                input_dim=input_CPPN_dim,
                                                weights_path=CPPN_weights_path,
                                                output_dim=2)  # ground + ceiling

        self.set_terrain_cppn_scale(terrain_cppn_scale, ceiling_offset, ceiling_clip_offset)

        # Set observation / action spaces
        self._generate_agent()  # To get state / action sizes
        agent_action_size = self.agent_body.get_action_size()
        self.action_space = spaces.Box(np.array([-1] * agent_action_size),
                                       np.array([1] * agent_action_size), dtype=np.float32)

        agent_state_size = self.agent_body.get_state_size()
        high = np.array([np.inf] * (agent_state_size +
                                    6 +  # head infos (including water) + is_dead
                                    NB_LIDAR*2))  # lidar info + if creeper for each lidar
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_terrain_cppn_scale(self, terrain_cppn_scale, ceiling_offset, ceiling_clip_offset):
        '''
        Scale the terrain generate by the CPPN to be more suited to our embodiments.
        '''
        assert terrain_cppn_scale > 1
        self.TERRAIN_CPPN_SCALE = terrain_cppn_scale
        self.CEILING_LIMIT = 1000 / self.TERRAIN_CPPN_SCALE
        self.GROUND_LIMIT = -1000 / self.TERRAIN_CPPN_SCALE
        self.ceiling_offset = ceiling_offset / self.TERRAIN_CPPN_SCALE
        self.ceiling_clip_offset = ceiling_clip_offset / self.TERRAIN_CPPN_SCALE

    def set_environment(self, input_vector, water_level, creepers_width=None, creepers_height=None,
                        creepers_spacing=0.1, terrain_cppn_scale=10):
        '''
        Set the parameters controlling the PCG algorithm to generate a task.
        Call this method before `reset()`.
        '''
        self.CPPN_input_vector = input_vector
        self.water_level = water_level.item() if isinstance(water_level, np.float32) else water_level
        self.water_level = max(0.01, self.water_level)
        self.creepers_width = creepers_width if creepers_width is not None else creepers_width
        self.creepers_height = creepers_height if creepers_height is not None else creepers_height
        self.creepers_spacing = max(0.01, creepers_spacing)
        self.set_terrain_cppn_scale(terrain_cppn_scale,
                                    self.ceiling_offset*self.TERRAIN_CPPN_SCALE,
                                    self.ceiling_clip_offset*self.TERRAIN_CPPN_SCALE)

    def _destroy(self):
        # if not self.terrain: return
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []

        self.agent_body.destroy(self.world)

    def reset(self):
        self.world.contactListener = None
        self.contact_listener.Reset()
        self._destroy()
        self.world.contactListener = self.contact_listener
        self.critical_contact = False
        self.prev_shaping = None
        self.scroll = [0.0, 0.0]
        self.lidar_render = 0
        self.water_y = self.GROUND_LIMIT
        self.nb_steps_outside_water = 0
        self.nb_steps_under_water = 0

        self.generate_game()

        self.drawlist = self.terrain + self.agent_body.get_elements_to_render()

        self.lidar = [LidarCallback(self.agent_body.reference_head_object.fixtures[0].filterData.maskBits)
                      for _ in range(NB_LIDAR)]

        actions_to_play = np.array([0] * self.action_space.shape[0])
        # If embodiment is a climber, make it start hanging on the ceiling using a few steps to let the Box2D solver handle positions.
        if self.agent_body.body_type == BodyTypesEnum.CLIMBER:
            # Init climber
            y_diff = 0
            for i in range(len(self.agent_body.sensors)):
                actions_to_play[len(actions_to_play) - i - 1] = 1
                # Hang sensor
                sensor = self.agent_body.sensors[len(self.agent_body.sensors) - i - 1]
                if y_diff == 0:
                    y_diff = TERRAIN_HEIGHT + self.ceiling_offset - sensor.position[1]
                sensor.position = (sensor.position[0],
                                   TERRAIN_HEIGHT + self.ceiling_offset)

            for body_part in self.agent_body.body_parts:
                body_part.position = (body_part.position[0],
                                      body_part.position[1] + y_diff)

            for i in range(NB_FIRST_STEPS_HANG):
                self.step(actions_to_play)

        initial_state = self.step(actions_to_play)[0]
        self.nb_steps_outside_water = 0
        self.nb_steps_under_water = 0
        self.episodic_reward = 0
        return initial_state

    def step(self, action):
        # Check if agent's dead
        if hasattr(self.agent_body, "nb_steps_can_survive_outside_water") and \
                        self.nb_steps_outside_water > self.agent_body.nb_steps_can_survive_outside_water or \
                        hasattr(self.agent_body, "nb_steps_can_survive_under_water") and \
                                self.nb_steps_under_water > self.agent_body.nb_steps_can_survive_under_water:
            is_agent_dead = True
            action = np.array([0] * self.action_space.shape[0])
        else:
            is_agent_dead = False
        self.agent_body.activate_motors(action)

        # Prepare climbing dynamics according to the actions (i.e. sensor ready to grasp or sensor release destroying joint)
        if self.agent_body.body_type == BodyTypesEnum.CLIMBER:
            self.climbing_dynamics.before_step_climbing_dynamics(action, self.agent_body, self.world)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        # Create joints between sensors ready to grasp if collision with graspable area was detected
        if self.agent_body.body_type == BodyTypesEnum.CLIMBER:
            self.climbing_dynamics.after_step_climbing_dynamics(self.world.contactListener, self.world)

        # Calculate water physics
        self.water_dynamics.calculate_forces(self.world.contactListener.fixture_pairs)

        head = self.agent_body.reference_head_object
        pos = head.position
        vel = head.linearVelocity

        for i in range(NB_LIDAR):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin((self.lidar_angle * i / NB_LIDAR + self.lidar_y_offset)) * LIDAR_RANGE,
                pos[1] - math.cos((self.lidar_angle * i / NB_LIDAR) + self.lidar_y_offset) * LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        is_under_water = pos.y <= self.water_y
        if not is_agent_dead:
            if is_under_water:
                self.nb_steps_under_water += 1
                self.nb_steps_outside_water = 0
            else:
                self.nb_steps_outside_water += 1
                self.nb_steps_under_water = 0

        state = [
            head.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * head.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            1.0 if is_under_water else 0.0,
            1.0 if is_agent_dead else 0.0
        ]

        # add leg-related state
        state.extend(self.agent_body.get_motors_state())

        # add sensor-related state
        if self.agent_body.body_type == BodyTypesEnum.CLIMBER:
            state.extend(self.agent_body.get_sensors_state())

        # add lidar-related state with distance and surface detected
        nb_of_water_detected = 0
        surface_dectected = []
        for lidar in self.lidar:
            state.append(lidar.fraction)
            if lidar.is_water_detected:
                surface_dectected.append(-1)
                nb_of_water_detected += 1
            elif lidar.is_creeper_detected:
                surface_dectected.append(1)
            else:
                surface_dectected.append(0)

        # state.append(nb_of_water_detected / NB_LIDAR)  # percentage of lidars that detect water
        state.extend(surface_dectected)

        self.scroll = [pos[0] - RENDERING_VIEWER_W / SCALE / 5,
                       pos[1] - RENDERING_VIEWER_H / SCALE / 5 - TERRAIN_HEIGHT + 1 / SCALE]  # 1 = grass

        shaping = 130 * pos[
            0] / SCALE  # moving forward is a way to receive reward (normalized to get 300 on completion)
        if not (
            hasattr(self.agent_body, "remove_reward_on_head_angle") and self.agent_body.remove_reward_on_head_angle):
            shaping -= 5.0 * abs(
                state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= self.agent_body.TORQUE_PENALTY * 80 * np.clip(np.abs(a), 0, 1)  # 80 => Original torque
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        # Ending conditions
        done = False
        if self.critical_contact or pos[0] < 0:
            reward = -100
            done = True
        if pos[0] > (TERRAIN_LENGTH + self.TERRAIN_STARTPAD - TERRAIN_END) * TERRAIN_STEP:
            done = True
        self.episodic_reward += reward

        return np.array(state), reward, done, {"success": self.episodic_reward > 230}

    def close(self):
        self.world.contactListener = None
        self.contact_listener.Reset()
        self._destroy()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # region Rendering
    # ------------------------------------------ RENDERING ------------------------------------------
    def color_agent_head(self, c1, c2):
        '''
        Color agent's head depending on its 'dying' state.
        '''
        ratio = 0
        if hasattr(self.agent_body, "nb_steps_can_survive_outside_water"):
            ratio = self.nb_steps_outside_water / self.agent_body.nb_steps_can_survive_outside_water
        elif hasattr(self.agent_body, "nb_steps_can_survive_under_water"):
            ratio = self.nb_steps_under_water / self.agent_body.nb_steps_can_survive_under_water

        color1 = (c1[0] + ratio*(1.0 - c1[0]),
                  c1[1] + ratio*(0.0 - c1[1]),
                  c1[2] + ratio*(0.0 - c1[2]))
        color2 = c2
        return color1, color2

    def render(self, mode='human', draw_lidars=True):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(RENDERING_VIEWER_W, RENDERING_VIEWER_H)
        self.viewer.set_bounds(self.scroll[0], RENDERING_VIEWER_W/SCALE + self.scroll[0], # x
                               self.scroll[1], RENDERING_VIEWER_H/SCALE + self.scroll[1]) # y

        self.viewer.draw_polygon( [
            (self.scroll[0], self.scroll[1]),
            (self.scroll[0]+RENDERING_VIEWER_W/SCALE, self.scroll[1]),
            (self.scroll[0]+RENDERING_VIEWER_W/SCALE, self.scroll[1]+RENDERING_VIEWER_H/SCALE),
            (self.scroll[0], self.scroll[1]+RENDERING_VIEWER_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )

        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll[0]/2: continue
            if x1 > self.scroll[0]/2 + RENDERING_VIEWER_W/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll[0]/2, p[1]) for p in poly], color=(1,1,1))

        for obj in self.drawlist:
            color1 = obj.color1
            color2 = obj.color2
            if obj.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR and obj.userData.has_joint: # Color sensors when attached
                color1 = (1.0, 1.0, 0.0)
                color2 = (1.0, 1.0, 0.0)
            elif obj == self.agent_body.reference_head_object:
                color1, color2 = self.color_agent_head(color1, color2)

            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=color2, linewidth=2)

        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll[0]: continue
            if poly[0][0] > self.scroll[0] + RENDERING_VIEWER_W / SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        # Draw lidars
        if draw_lidars:
            for i in range(len(self.lidar)):
                l = self.lidar[i]
                self.viewer.draw_polyline([l.p1, l.p2], color=(1, 0, 0), linewidth=1)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def _SET_RENDERING_VIEWPORT_SIZE(self, width, height=None, keep_ratio=True):
        global RENDERING_VIEWER_W, RENDERING_VIEWER_H
        RENDERING_VIEWER_W = width
        if keep_ratio or height is None:
            RENDERING_VIEWER_H = int(RENDERING_VIEWER_W / (VIEWPORT_W / VIEWPORT_H))
        else:
            RENDERING_VIEWER_H = height
    #endregion

    #region Fixtures Initialization
    # ------------------------------------------ FIXTURES INITIALIZATION ------------------------------------------

    def create_terrain_fixtures(self):
        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0),
                                (1, 0),
                                (1, -1),
                                (0, -1)]),
            friction=FRICTION,
            categoryBits=0x1,
            maskBits=0xFFFF
        )

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=
                            [(0, 0),
                             (1, 1)]),
            friction=FRICTION,
            categoryBits=0x1,
            maskBits=0xFFFF
        )

        self.fd_water = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0),
                                (1, 0),
                                (1, -1),
                                (0, -1)]),
            density=WATER_DENSITY,
            isSensor=True
        )

        self.fd_creeper = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0),
                                (1, 0),
                                (1, -1),
                                (0, -1)]),
            density=5.0,
            isSensor=True,
        )
    #endregion

    # region Game Generation
    # ------------------------------------------ GAME GENERATION ------------------------------------------

    def generate_game(self):
        self._generate_terrain()
        self._generate_clouds()
        self._generate_agent()

    def clip_ceiling_values(self, row, clip_offset):
        if row["ceiling"] >= row["ground"] + clip_offset:
            return row["ceiling"]
        else:
            return row["ground"] + clip_offset

    def _generate_terrain(self):
        y = self.terrain_CPPN.generate(self.CPPN_input_vector)
        y = y / self.TERRAIN_CPPN_SCALE
        ground_y = y[:, 0]
        ceiling_y = y[:, 1]

        # Align ground with startpad
        offset = TERRAIN_HEIGHT - ground_y[0]
        ground_y = np.add(ground_y, offset)

        # Align ceiling from startpad ceiling
        offset = TERRAIN_HEIGHT + self.ceiling_offset - ceiling_y[0]
        ceiling_y = np.add(ceiling_y, offset)

        self.terrain = []
        self.terrain_x = []
        self.terrain_ground_y = []
        self.terrain_ceiling_y = []
        terrain_creepers = []
        water_body = None
        x = 0
        max_x = TERRAIN_LENGTH * TERRAIN_STEP + self.TERRAIN_STARTPAD * TERRAIN_STEP

        # Generation of terrain
        i = 0
        while x < max_x:
            self.terrain_x.append(x)
            if i < self.TERRAIN_STARTPAD:
                self.terrain_ground_y.append(TERRAIN_HEIGHT)
                self.terrain_ceiling_y.append(TERRAIN_HEIGHT + self.ceiling_offset)
            else:
                self.terrain_ground_y.append(ground_y[i - self.TERRAIN_STARTPAD].item())

                # Clip ceiling
                if ceiling_y[i - self.TERRAIN_STARTPAD] >= ground_y[i - self.TERRAIN_STARTPAD] + self.ceiling_clip_offset:
                    ceiling_val = ceiling_y[i - self.TERRAIN_STARTPAD]
                else:
                    ceiling_val = ground_y[i - self.TERRAIN_STARTPAD] + self.ceiling_clip_offset

                self.terrain_ceiling_y.append(ceiling_val.item())

            x += TERRAIN_STEP
            i += 1

        # Draw terrain
        space_from_precedent_creeper = self.creepers_spacing
        self.terrain_poly = []
        for i in range(len(self.terrain_x) - 1):
            # Ground
            poly = [
                (self.terrain_x[i], self.terrain_ground_y[i]),
                (self.terrain_x[i + 1], self.terrain_ground_y[i + 1])
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_edge,
                userData=CustomUserData("grass", CustomUserDataObjectTypes.TERRAIN))
            color = (0.3, 1.0 if (i % 2) == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [(poly[1][0], self.GROUND_LIMIT), (poly[0][0], self.GROUND_LIMIT)]
            self.terrain_poly.append((poly, color))

            # Ceiling
            poly = [
                (self.terrain_x[i], self.terrain_ceiling_y[i]),
                (self.terrain_x[i + 1], self.terrain_ceiling_y[i + 1])
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_edge,
                userData=CustomUserData("rock", CustomUserDataObjectTypes.GRIP_TERRAIN))
            color = (0, 0.25, 0.25)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.5, 0.5, 0.5)
            poly += [(poly[1][0], self.CEILING_LIMIT), (poly[0][0], self.CEILING_LIMIT)]
            self.terrain_poly.append((poly, color))

            # Creepers
            if self.creepers_width is not None and self.creepers_height is not None:
                if space_from_precedent_creeper >= self.creepers_spacing:
                    creeper_height = max(0.2, self.np_random.normal(self.creepers_height, 0.1))
                    creeper_width = max(0.2, self.creepers_width)
                    creeper_step_size = np.floor(creeper_width / TERRAIN_STEP).astype(int)
                    creeper_step_size = max(1, creeper_step_size)
                    creeper_y_init_pos = max(self.terrain_ceiling_y[i],
                                             self.terrain_ceiling_y[min(i + creeper_step_size, len(self.terrain_x) - 1)])
                    if self.movable_creepers: # Break creepers in multiple objects linked by joints
                        previous_creeper_part = t
                        for w in range(math.ceil(creeper_height)):
                            if w == creeper_height // 1:
                                h = max(0.2, creeper_height % 1)
                            else:
                                h = 1

                            poly = [
                                (self.terrain_x[i] + creeper_width, creeper_y_init_pos - (w * 1) - h),
                                (self.terrain_x[i] + creeper_width, creeper_y_init_pos - (w * 1)),
                                (self.terrain_x[i], creeper_y_init_pos - (w * 1)),
                                (self.terrain_x[i], creeper_y_init_pos - (w * 1) - h)
                            ]
                            self.fd_creeper.shape.vertices = poly
                            t = self.world.CreateDynamicBody(
                                fixtures=self.fd_creeper,
                                userData=CustomUserData("creeper", CustomUserDataObjectTypes.SENSOR_GRIP_TERRAIN))
                            c = (0.437, 0.504, 0.375)
                            t.color1 = c
                            t.color2 = tuple([_c+0.1 for _c  in c])
                            self.terrain.append(t)

                            rjd = revoluteJointDef(
                                bodyA=previous_creeper_part,
                                bodyB=t,
                                anchor=(self.terrain_x[i] + creeper_width / 2, creeper_y_init_pos - (w * 1)),
                                enableLimit=True,
                                lowerAngle=-0.4 * np.pi,
                                upperAngle=0.4 * np.pi,
                            )
                            self.world.CreateJoint(rjd)
                            previous_creeper_part = t
                    else:
                        poly = [
                            (self.terrain_x[i], creeper_y_init_pos),
                            (self.terrain_x[i] + creeper_width, creeper_y_init_pos),
                            (self.terrain_x[i] + creeper_width, creeper_y_init_pos - creeper_height),
                            (self.terrain_x[i], creeper_y_init_pos - creeper_height),
                        ]
                        self.fd_creeper.shape.vertices = poly
                        t = self.world.CreateStaticBody(
                            fixtures=self.fd_creeper,
                            userData=CustomUserData("creeper", CustomUserDataObjectTypes.SENSOR_GRIP_TERRAIN))
                        c = (0.437, 0.504, 0.375)
                        t.color1 = c
                        t.color2 = c
                        terrain_creepers.append(t)
                    space_from_precedent_creeper = 0
                else:
                    space_from_precedent_creeper += self.terrain_x[i] - self.terrain_x[i - 1]

        # Water
        # Fill water from GROUND_LIMIT to highest point of the current ceiling
        air_max_distance = max(self.terrain_ceiling_y) - self.GROUND_LIMIT
        water_y = self.GROUND_LIMIT + self.water_level * air_max_distance
        self.water_y = water_y

        water_poly = [
            (self.terrain_x[0], self.GROUND_LIMIT),
            (self.terrain_x[0], water_y),
            (self.terrain_x[len(self.terrain_x) - 1], water_y),
            (self.terrain_x[len(self.terrain_x) - 1], self.GROUND_LIMIT)
        ]
        self.fd_water.shape.vertices = water_poly
        t = self.world.CreateStaticBody(
            fixtures=self.fd_water,
            userData=CustomUserData("water", CustomUserDataObjectTypes.WATER))
        c = (0.465, 0.676, 0.898)
        t.color1 = c
        t.color2 = c
        water_body = t

        self.terrain.extend(terrain_creepers)
        if water_body is not None:
            self.terrain.append(water_body)
        self.terrain.reverse()


    def _generate_clouds(self):
        self.cloud_poly   = []
        for i in range(TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP
            y = VIEWPORT_H/SCALE*3/4
            poly = [
                (x+15*TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP),
                 y+ 5*TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def _generate_agent(self):
        init_x = TERRAIN_STEP*self.TERRAIN_STARTPAD/2
        init_y = TERRAIN_HEIGHT + self.agent_body.AGENT_CENTER_HEIGHT # set y position according to the agent

        self.agent_body.draw(
            self.world,
            init_x,
            init_y,
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        )
    #endregion