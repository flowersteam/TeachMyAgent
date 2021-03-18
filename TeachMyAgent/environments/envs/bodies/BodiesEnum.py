from enum import Enum
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum

from TeachMyAgent.environments.envs.bodies.back_bodies.BackChimpanzee import BackChimpanzee

from TeachMyAgent.environments.envs.bodies.climbers.ClimbingChestProfileChimpanzee import ClimbingChestProfileChimpanzee
from TeachMyAgent.environments.envs.bodies.climbers.ClimbingProfileChimpanzee import ClimbingProfileChimpanzee

from TeachMyAgent.environments.envs.bodies.swimmers.FishBody import FishBody

from TeachMyAgent.environments.envs.bodies.amphibians.AmphibiousBipedalBody import AmphibiousBipedalBody

from TeachMyAgent.environments.envs.bodies.walkers.old.OldBigQuadruBody import OldBigQuadruBody
from TeachMyAgent.environments.envs.bodies.walkers.old.OldClassicBipedalBody import OldClassicBipedalBody
from TeachMyAgent.environments.envs.bodies.walkers.SmallBipedalBody import SmallBipedalBody
from TeachMyAgent.environments.envs.bodies.walkers.BigQuadruBody import BigQuadruBody
from TeachMyAgent.environments.envs.bodies.walkers.ClassicBipedalBody import ClassicBipedalBody
from TeachMyAgent.environments.envs.bodies.walkers.HumanBody import HumanBody
from TeachMyAgent.environments.envs.bodies.walkers.MillipedeBody import MillipedeBody
from TeachMyAgent.environments.envs.bodies.walkers.ProfileChimpanzee import ProfileChimpanzee
from TeachMyAgent.environments.envs.bodies.walkers.SpiderBody import SpiderBody
from TeachMyAgent.environments.envs.bodies.walkers.WheelBody import WheelBody


class BodiesEnum(Enum):
    small_bipedal = SmallBipedalBody
    classic_bipedal = ClassicBipedalBody
    big_quadru = BigQuadruBody
    spider = SpiderBody
    millipede = MillipedeBody
    wheel = WheelBody
    human = HumanBody
    old_classic_bipedal = OldClassicBipedalBody
    profile_chimpanzee = ProfileChimpanzee
    back_chimpanzee = BackChimpanzee
    old_big_quadru = OldBigQuadruBody
    fish = FishBody
    climbing_profile_chimpanzee = ClimbingProfileChimpanzee
    climbing_chest_profile_chimpanzee = ClimbingChestProfileChimpanzee
    amphibious_bipedal = AmphibiousBipedalBody

    @classmethod
    def get_body_type(self, body_name):
        '''
        Return embodiment's type given its name.
        '''
        if body_name in ['climbing_chest_profile_chimpanzee', 'climbing_profile_chimpanzee']:
            return BodyTypesEnum.CLIMBER
        elif body_name == 'fish':
            return BodyTypesEnum.SWIMMER
        elif body_name == 'amphibious_bipedal':
            return BodyTypesEnum.AMPHIBIAN
        else:
            return BodyTypesEnum.WALKER