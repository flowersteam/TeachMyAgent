# Parametric Walker continuous environment
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements. There's no coordinates
# in the state vector.
#
# Initially Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
# Modified by Rémy Portelas and licensed under TeachMyAgent/teachers/LICENSES/ALP-GMM
# Modified Clément Romac

#region Imports

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle
import math

from TeachMyAgent.environments.envs.bodies.BodiesEnum import BodiesEnum
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum
from TeachMyAgent.environments.envs.utils.custom_user_data import CustomUserDataObjectTypes, CustomUserData

#endregion

#region Utils


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        for body in [contact.fixtureA.body, contact.fixtureB.body]:
            if body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT and body.userData.check_contact:
                body.userData.has_contact = True
                if body.userData.is_contact_critical:
                    self.env.head_contact = True

    def EndContact(self, contact):
        for body in [contact.fixtureA.body, contact.fixtureB.body]:
            if body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT and body.userData.check_contact:
                body.userData.has_contact = False

def Rotate2D(pts,cnt,ang=np.pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    m1 = pts-cnt
    m2 = np.array([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]])
    return np.dot(m1,m2)+cnt


class LidarCallback(Box2D.b2.rayCastCallback):
    def __init__(self, agent_mask_filter):
        Box2D.b2.rayCastCallback.__init__(self)
        self.agent_mask_filter = agent_mask_filter
        self.fixture = None
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & self.agent_mask_filter) == 0:
            return -1
        self.p2 = point
        self.fraction = fraction
        return fraction

#endregion

#region Constants

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
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
TERRAIN_END    = 10    # in steps
INITIAL_TERRAIN_STARTPAD = 20 # in steps
FRICTION = 2.5

#endregion

class ParametricContinuousStumpTracks(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, walker_type, **walker_args):
        super(ParametricContinuousStumpTracks, self).__init__()

        # Seed env and init Box2D
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = []

        self.prev_shaping = None

        # Create agent
        body_type = BodiesEnum.get_body_type(walker_type)
        if body_type == BodyTypesEnum.SWIMMER or body_type == BodyTypesEnum.AMPHIBIAN:
            self.walker_body = BodiesEnum[walker_type].value(SCALE, density=1.0, **walker_args)
        elif body_type == BodyTypesEnum.WALKER:
            self.walker_body = BodiesEnum[walker_type].value(SCALE, **walker_args,
                                                                reset_on_hull_critical_contact=True)
        else:
            self.walker_body = BodiesEnum[walker_type].value(SCALE, **walker_args)

        # Adapt startpad to walker's width
        self.TERRAIN_STARTPAD = INITIAL_TERRAIN_STARTPAD if \
            self.walker_body.AGENT_WIDTH / TERRAIN_STEP + 5 <= INITIAL_TERRAIN_STARTPAD else \
            self.walker_body.AGENT_WIDTH / TERRAIN_STEP + 5  # in steps
        self.create_terrain_fixtures()

        # Set observation / action spaces
        self._generate_walker()  # To get state / action sizes
        agent_action_size = self.walker_body.get_action_size()
        self.action_space = spaces.Box(np.array([-1] * agent_action_size),
                                       np.array([1] * agent_action_size), dtype=np.float32)

        agent_state_size = self.walker_body.get_state_size()
        high = np.array([np.inf] * (agent_state_size +
                                    4 +  # head infos
                                    NB_LIDAR))  # lidars infos
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_environment(self, roughness=None, stump_height=None, stump_width=None, stump_rot=None,
                        obstacle_spacing=None, poly_shape=None, stump_seq=None):
        '''
        Set the parameters controlling the PCG algorithm to generate a task.
        Call this method before `reset()`.
        '''
        self.roughness = roughness if roughness else 0
        self.obstacle_spacing = max(0.01, obstacle_spacing) if obstacle_spacing is not None else 8.0
        self.stump_height = [stump_height, 0.1] if stump_height is not None else None
        self.stump_width = stump_width
        self.stump_rot = stump_rot
        self.hexa_shape = poly_shape
        self.stump_seq = stump_seq
        if poly_shape is not None:
            self.hexa_shape = np.interp(poly_shape,[0,4],[0,4]).tolist()
            assert(len(poly_shape) == 12)
            self.hexa_shape = self.hexa_shape[0:12]

    def _destroy(self):
        # if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []

        self.walker_body.destroy(self.world)

    def reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.head_contact = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        self.generate_game()

        self.drawlist = self.terrain + self.walker_body.get_elements_to_render()

        self.lidar = [LidarCallback(self.walker_body.reference_head_object.fixtures[0].filterData.maskBits)
                      for _ in range(NB_LIDAR)]
        self.episodic_reward = 0

        return self.step(np.array([0] * self.action_space.shape[0]))[0]

    def step(self, action):
        self.walker_body.activate_motors(action)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        head = self.walker_body.reference_head_object
        pos = head.position
        vel = head.linearVelocity

        for i in range(NB_LIDAR):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*i/NB_LIDAR)*LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/NB_LIDAR)*LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)
        state = [
            head.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*head.angularVelocity/FPS,
            0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(VIEWPORT_H/SCALE)/FPS]

        # add leg-related state
        state.extend(self.walker_body.get_motors_state())

        if self.walker_body.body_type == BodyTypesEnum.CLIMBER:
            state.extend(self.walker_body.get_sensors_state())

        state += [l.fraction for l in self.lidar]

        self.scroll = pos.x - RENDERING_VIEWER_W/SCALE/5

        shaping  = 130*pos[0]/SCALE  # moving forward is a way to receive reward (normalized to get 300 on completion)
        if not (hasattr(self.walker_body, "remove_reward_on_head_angle") and self.walker_body.remove_reward_on_head_angle):
            shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= self.walker_body.TORQUE_PENALTY * 80 * np.clip(np.abs(a), 0, 1) # 80 => Original torque
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.head_contact or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_END)*TERRAIN_STEP:
            done   = True
        self.episodic_reward += reward
        return np.array(state), reward, done, {"success": self.episodic_reward > 230}

    def render(self, mode='human', draw_lidars=True):
        #self.scroll = 1
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(RENDERING_VIEWER_W, RENDERING_VIEWER_H)
        self.viewer.set_bounds(self.scroll, RENDERING_VIEWER_W/SCALE + self.scroll, 0, RENDERING_VIEWER_H/SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                  0),
            (self.scroll+RENDERING_VIEWER_W/SCALE, 0),
            (self.scroll+RENDERING_VIEWER_W/SCALE, RENDERING_VIEWER_H/SCALE),
            (self.scroll,                  RENDERING_VIEWER_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + RENDERING_VIEWER_W/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + RENDERING_VIEWER_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

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

    def close(self):
        self._destroy()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

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

        # Init default hexagon fixture and shape, used only for Hexagon Tracks
        self.fd_default_hexagon = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0),
                                (1, 0),
                                (1, -1),
                                (0, -1)]),
            friction=FRICTION,
            categoryBits=0x1,
            maskBits=0xFFFF
        )
        self.default_hexagon = [(-0.5, 0), (-0.5, 0.25), (-0.25, 0.5), (0.25, 0.5), (0.5, 0.25), (0.5, 0)]

    #endregion

    # region Game Generation
    # ------------------------------------------ GAME GENERATION ------------------------------------------

    def generate_game(self):
        self._generate_terrain()
        self._generate_clouds()
        self._generate_walker()

    def _generate_terrain(self):
        GRASS, STUMP, HEXA = 0, None, None
        cpt=1
        if self.stump_height:
            STUMP = cpt
            cpt += 1
        if self.hexa_shape:
            HEXA = cpt
            cpt += 1
        if self.stump_seq is not None:
            SEQ = cpt
            cpt += 1
        _STATES_ = cpt

        state = self.np_random.randint(1, _STATES_)
        velocity = 0.0
        y = TERRAIN_HEIGHT
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        x = 0
        max_x = TERRAIN_LENGTH * TERRAIN_STEP

        # Add startpad
        max_startpad_x = self.TERRAIN_STARTPAD * TERRAIN_STEP
        self.terrain_x.append(x)
        self.terrain_y.append(y)
        x += max_startpad_x
        self.terrain_x.append(x)
        self.terrain_y.append(y)
        oneshot = True

        # Generation of terrain
        while x < max_x:
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
                if x > max_startpad_x: velocity += self.np_random.uniform(-self.roughness, self.roughness)/SCALE
                y += velocity
                x += self.obstacle_spacing

            elif state==STUMP and oneshot:
                stump_height = max(0.05, self.np_random.normal(self.stump_height[0], self.stump_height[1]))
                stump_width = TERRAIN_STEP
                if self.stump_width is not None:
                    stump_width *= max(0.05, np.random.normal(self.stump_width[0], self.stump_width[1]))
                poly = [
                    (x, y),
                    (x+stump_width, y),
                    (x+stump_width, y+stump_height * TERRAIN_STEP),
                    (x,y+stump_height * TERRAIN_STEP),
                    ]
                x += stump_width
                if self.stump_rot is not None:
                    anchor = (np.array(poly[0]) + np.array(poly[1]))/2
                    rotation = np.clip(self.np_random.normal(self.stump_rot[0], self.stump_rot[1]),0,2*np.pi)
                    poly = Rotate2D(np.array(poly), anchor, rotation).tolist()
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon,
                    userData=CustomUserData("grass", CustomUserDataObjectTypes.TERRAIN))
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
            elif state==HEXA and oneshot:
                # first point do not move
                poly = []
                delta_pos = []
                for i in range(0,len(self.hexa_shape),2):
                    delta_pos.append(tuple(np.random.normal(self.hexa_shape[i:i+2],0.1)))
                for i,(b,d) in enumerate(zip(self.default_hexagon, delta_pos)):
                    if i != 0 and i != (len(self.default_hexagon)-1):
                        poly.append((x + (b[0]*TERRAIN_STEP) + (d[0]*TERRAIN_STEP),
                                     y + (b[1]*TERRAIN_STEP) + (d[1]*TERRAIN_STEP)))
                    else:
                        poly.append((x + (b[0]*TERRAIN_STEP) + (d[0]*TERRAIN_STEP),
                                     y + (b[1]*TERRAIN_STEP)))
                x += 1
                self.fd_default_hexagon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_default_hexagon)
                t.color1, t.color2 = (1.0, np.clip(delta_pos[0][1]/3,0,1), np.clip(delta_pos[-1][1]/3,0,1)), (0.6, 0.6, 0.6)
                self.terrain.append(t)

            elif state==SEQ and oneshot:
                for height, width in zip(self.stump_seq[0::2], self.stump_seq[1::2]):
                    stump_height = max(0.05, self.np_random.normal(height, 0.1))
                    stump_width = max(0.05, self.np_random.normal(width, 0.1))
                    poly = [
                        (x, y),
                        (x + stump_width, y),
                        (x + stump_width, y + stump_height * TERRAIN_STEP),
                        (x, y + stump_height * TERRAIN_STEP),
                    ]
                    x += stump_width
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(
                        fixtures=self.fd_polygon,
                        userData=CustomUserData("grass", CustomUserDataObjectTypes.TERRAIN))
                    t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    self.terrain.append(t)

            oneshot = False
            self.terrain_y.append(y)
            if state==GRASS:
                state = self.np_random.randint(1, _STATES_)
                oneshot = True
            else:
                state = GRASS
                oneshot = False

        # Draw terrain
        self.terrain_poly = []
        assert len(self.terrain_x) == len(self.terrain_y)
        for i in range(len(self.terrain_x)-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge,
                userData=CustomUserData("grass", CustomUserDataObjectTypes.TERRAIN))
            color = (0.3, 1.0 if (i % 2) == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
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

    def _generate_walker(self):
        init_x = TERRAIN_STEP*self.TERRAIN_STARTPAD/2
        if hasattr(self.walker_body, "old_morphology") and self.walker_body.old_morphology:
            init_y = TERRAIN_HEIGHT + 2 * self.walker_body.LEG_H
        else:
            init_y = TERRAIN_HEIGHT + self.walker_body.AGENT_CENTER_HEIGHT

        self.walker_body.draw(
            self.world,
            init_x,
            init_y,
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        )

    #endregion