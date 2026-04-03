from .scout import Scout
from .kg_matcher import KGMatcher
from .architect import Architect
from .trainer import Trainer
from .critic import Critic

ALL_WORKERS = {
    "scout": Scout,
    "kg_matcher": KGMatcher,
    "architect": Architect,
    "trainer": Trainer,
    "critic": Critic,
}
