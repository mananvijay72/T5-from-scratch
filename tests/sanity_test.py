import numpy as np
from utils.logger import get_logger
from utils.config import load_config

logger = get_logger()
config = load_config()

logger.info("Loaded config:")
logger.info(config)

a = np.random.rand(2, 2)
b = np.random.rand(2, 2)
c = np.matmul(a, b)

logger.info("Matrix multiplication result:")
logger.info(c)