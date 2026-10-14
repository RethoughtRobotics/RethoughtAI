"""VITA (vision-to-action flow matching), from andnet-deboer/lerobot@vita (220c6fb,
src/lerobot/policies/vita) until a LeRobot release has it. Importing this registers the "vita"
policy type; LeRobot finds VitaPolicy and its processors by naming convention."""

from .configuration_vita import VitaConfig
from .modeling_vita import VitaPolicy
from .processor_vita import make_vita_pre_post_processors

__all__ = ["VitaConfig", "VitaPolicy", "make_vita_pre_post_processors"]
