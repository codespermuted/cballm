from .scout import Scout
from .engineer import Engineer
from .architect import Architect
from .trainer import Trainer
from .critic import Critic

ALL_WORKERS = {
    "scout": Scout,
    "engineer": Engineer,
    "architect": Architect,
    "trainer": Trainer,
    "critic": Critic,
}
